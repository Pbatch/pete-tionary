import torch
from siren import SirenNet
from torch import nn
import kornia.augmentation as K


class DeepDaze(nn.Module):
    PERCEPTOR_SIZE = 224

    def __init__(
            self,
            perceptor,
            n_images=1,
            batch_size=32,
            loss_coef=100,
            lower_bound_cutout=0.1,
            upper_bound_cutout=1.0,
            averaging_weight=0.3,
            dim_hidden=256,
            n_layers=20,
            image_size=64,
            w0=30,
            w0_initial=30,
    ):
        super().__init__()
        self.perceptor = perceptor
        self.n_images = n_images
        self.batch_size = batch_size
        self.loss_coef = loss_coef
        self.averaging_weight = averaging_weight

        self.generators = nn.ModuleList([SirenNet(dim_in=2,
                                                  dim_hidden=dim_hidden,
                                                  dim_out=3,
                                                  n_layers=n_layers,
                                                  image_size=image_size,
                                                  w0=w0,
                                                  w0_initial=w0_initial
                                                  ) for _ in range(self.n_images)])
        self.augs = torch.nn.Sequential(
            K.RandomResizedCrop(size=(self.PERCEPTOR_SIZE, self.PERCEPTOR_SIZE),
                                scale=(lower_bound_cutout, upper_bound_cutout),
                                align_corners=False),
            K.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711))
        )

    @staticmethod
    def siren_norm(img):
        return ((img + 1) * 0.5).clamp(0.0, 1.0)

    def forward(self, text_embed, return_loss=True):
        imgs = [self.siren_norm(g()) for g in self.generators]

        if not return_loss:
            return imgs

        loss = 0
        for img in imgs:
            # Randomly crop, resize and normalize the image
            img_batch = self.augs(img.repeat(self.batch_size, 1, 1, 1))

            # Calculate the embedding of the image
            img_embed = self.perceptor.encode_image(img_batch)

            # Calculate the averaged loss and the general loss
            avg_img_embed = img_embed.mean(dim=0).unsqueeze(0)
            averaged_loss = -self.loss_coef * torch.cosine_similarity(text_embed, avg_img_embed, dim=-1).mean()
            general_loss = -self.loss_coef * torch.cosine_similarity(text_embed, img_embed, dim=-1).mean()

            # Merge the losses
            loss += averaged_loss * self.averaging_weight + general_loss * (1 - self.averaging_weight)

        return imgs, loss
