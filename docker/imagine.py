import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torchvision.transforms import ToPILImage
from tqdm.autonotebook import tqdm
from tokenizer import tokenize
import PIL


class Imagine(nn.Module):
    def __init__(self,
                 prompt,
                 model,
                 epochs=100,
                 lr=4e-5
                 ):
        super().__init__()
        self.prompt = prompt
        self.model = model
        self.epochs = epochs
        self.lr = lr

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler()
        self.optimizer = AdamW(self.model.generators.parameters(), lr=self.lr)

        self.model.to(self.device)
        self.clip_encoding = self.create_encoding()

    def create_encoding(self):
        tokenized_prompt = tokenize(self.prompt).to(self.device)
        with torch.no_grad():
            encoding = self.model.perceptor.encode_text(tokenized_prompt).detach()
        return encoding

    def forward(self):
        with tqdm(total=self.epochs) as pbar:
            for _ in range(self.epochs):
                with autocast(enabled=True):
                    _, loss = self.model(self.clip_encoding)
                pbar.update(1)
                pbar.set_description(f'Loss:{loss:.3f}')
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

    def save(self, save_paths, size=256):
        self.model.generators.eval()
        with torch.no_grad():
            imgs = self.model(self.clip_encoding, return_loss=False)
            for img, save_path in zip(imgs, save_paths):
                img = img.cpu().float().clamp(0., 1.)
                img = ToPILImage()(img.squeeze())
                img = img.resize((size, size), resample=PIL.Image.BICUBIC)
                img.save(save_path, quality=95, subsampling=0)
