import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import ToPILImage, RandomResizedCrop, Normalize, Resize
from tqdm import tqdm
from tokenizer import tokenize


class DirectVisions:
    PERCEPTOR_SIZE = 224

    def __init__(self,
                 prompt,
                 perceptor,
                 enhancer,
                 noise_std=0.2,
                 n_images=1,
                 cycles=250,
                 cuts=1,
                 gamma=1.0,
                 gain=1.0,
                 chroma_noise_scale=1.0,
                 luma_noise_mean=0.0,
                 luma_noise_scale=2.0,
                 noise_clamp=5,
                 luma_lr=5e-2,
                 chroma_lr=3e-2
                 ):
        self.prompt = prompt
        self.perceptor = perceptor
        self.enhancer = enhancer
        self.noise_std = noise_std
        self.n_images = n_images
        self.cycles = cycles
        self.cuts = cuts
        # Contrast
        self.gamma = gamma
        # Brightness
        self.gain = gain
        # Saturation (0 to 2 is safe but you can go as high as you want)
        self.chroma_noise_scale = chroma_noise_scale
        # Brightness (-3 to 3 seems safe but around 0 seems to work better)
        self.luma_noise_mean = luma_noise_mean
        # Contrast (0 to 2 is safe but you can go as high as you want)
        self.luma_noise_scale = luma_noise_scale
        # Turn this down if you're getting persistent super bright or dark spots.
        self.noise_clamp = noise_clamp
        self.luma_lr = luma_lr
        self.chroma_lr = chroma_lr

        self.prompt_embed = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.augs = torch.nn.Sequential(
            Resize(self.PERCEPTOR_SIZE),
            RandomResizedCrop(size=(self.PERCEPTOR_SIZE, self.PERCEPTOR_SIZE),
                              scale=(0.1, 0.9)),
            Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                      std=(0.26862954, 0.26130258, 0.27577711)),
        )

        torch.autograd.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True

    @staticmethod
    def _calculate_total_variation_loss(images, strength):
        Y = (images[:, :, 1:, :] - images[:, :, :-1, :]).abs().mean()
        X = (images[:, :, :, 1:] - images[:, :, :, :-1]).abs().mean()
        loss = (X + Y) * 0.5 * strength
        return loss

    @staticmethod
    def _luma_chroma_to_images(luma, chroma):
        luma = torch.sigmoid(luma)
        chroma = torch.sigmoid(chroma) - 0.5

        Co = chroma[:, 0:1]
        Cg = chroma[:, 1:2]

        R = luma + Co - Cg
        G = luma + Cg
        B = luma - Co - Cg
        images = torch.cat((R, G, B), dim=1)
        return images

    def _embed_prompt(self):
        tokenized_prompt = tokenize(self.prompt).to(self.device)
        embedded_prompt = self.perceptor.encode_text(tokenized_prompt).detach()
        return embedded_prompt

    def _init_luma_chroma(self, dim):
        luma = torch.randn(size=(self.n_images, 1, dim, dim), device=self.device)
        luma = luma * self.luma_noise_scale + self.luma_noise_mean

        chroma = torch.randn(size=(self.n_images, 2, dim, dim), device=self.device)
        chroma = chroma * self.chroma_noise_scale

        luma = luma.clamp(-self.noise_clamp, self.noise_clamp)
        chroma = chroma.clamp(-self.noise_clamp, self.noise_clamp)

        luma = nn.Parameter(luma, requires_grad=True)
        chroma = nn.Parameter(chroma, requires_grad=True)

        return luma, chroma

    def _calculate_clip_loss(self, images):
        image_batch = self.augs(images.repeat(self.cuts, 1, 1, 1))
        image_batch += torch.randn_like(image_batch) * self.noise_std
        image_embed = self.perceptor.encode_image(image_batch)

        loss = torch.mean(-torch.cosine_similarity(self.prompt_embed, image_embed))

        return loss

    def _calculate_loss(self, images, denoise):
        loss = 0
        loss += self._calculate_clip_loss(images)
        loss += self._calculate_total_variation_loss(images, denoise)
        return loss

    def _cycle(self, luma, chroma, denoise, optimizer):
        with tqdm(total=self.cycles) as pbar:
            for _ in range(self.cycles):
                with torch.enable_grad():
                    images = self._luma_chroma_to_images(luma, chroma)
                    loss = self._calculate_loss(images, denoise)
                    pbar.update(1)
                    pbar.set_description(f'Loss:{loss:.3f}')
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

    def _save_images(self, images):
        for i in range(self.n_images):
            save_path = f'prompt={self.prompt.replace(" ", "_")}-seed={i}.jpg'
            image = self.enhancer(images[i:i + 1]).data.squeeze().float().cpu().clamp_(0, 1)
            image = ToPILImage()(image.squeeze())
            image.save(save_path, quality=95, subsampling=0)

    def run(self):
        dim = [4, 8, 16, 32, 64]
        denoise = [0.0, 0.01, 0.1, 0.2, 0.5]
        noise = [0.0, 0.75, 0.2, 0.2, 0.2]

        self.prompt_embed = self._embed_prompt()
        luma, chroma = self._init_luma_chroma(dim[0])

        for i in range(len(dim)):
            # Resize luma and chroma
            luma = F.interpolate(luma.data,
                                 size=(dim[i], dim[i]),
                                 mode='bilinear',
                                 align_corners=False)
            chroma = F.interpolate(chroma.data,
                                   size=(dim[i], dim[i]),
                                   mode='bilinear',
                                   align_corners=False)

            # Make luma and chroma parameters again
            luma = nn.Parameter(luma, requires_grad=True)
            chroma = nn.Parameter(chroma, requires_grad=True)

            # Add noise
            luma += torch.randn_like(luma) * noise[i]
            chroma += torch.randn_like(chroma) * noise[i]

            # Set the optimizer
            params = (
                {'params': luma, 'lr': self.luma_lr, 'weight_decay': 0},
                {'params': chroma, 'lr': self.chroma_lr, 'weight_decay': 0}
            )
            optimizer = torch.optim.AdamW(params)

            self._cycle(luma, chroma, denoise[i], optimizer)
            images = self._luma_chroma_to_images(luma, chroma)

            # On the last iteration, save the images
            if i == len(dim) - 1:
                self._save_images(images)
