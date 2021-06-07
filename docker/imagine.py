import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torchvision.transforms import ToPILImage
from tqdm import trange
from tokenizer import tokenize


class Imagine(nn.Module):
    def __init__(self,
                 prompt,
                 model,
                 save_path,
                 lr=4e-5,
                 epochs=100
                 ):
        super().__init__()
        self.prompt = prompt
        self.model = model
        self.save_path = save_path
        self.lr = lr
        self.epochs = epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler()
        self.optimizer = torch.optim.Adam(self.model.model.parameters(), lr)

        self.model.to(self.device)
        self.clip_encoding = self.create_encoding()

    def create_encoding(self):
        tokenized_prompt = tokenize(self.prompt).to(self.device)
        with torch.no_grad():
            encoding = self.model.perceptor.encode_text(tokenized_prompt).detach()
        return encoding

    def forward(self):
        for _ in trange(self.epochs, desc='epochs'):
            with autocast(enabled=True):
                out, loss = self.model(self.clip_encoding)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        # Save the image
        with torch.no_grad():
            img = self.model(self.clip_encoding, return_loss=False).cpu().float().clamp(0., 1.)
            img = ToPILImage()(img.squeeze())
            img.save(self.save_path, quality=95, subsampling=0)
