import torch
import torch.nn.functional as F
from siren import SirenNet
from torch import nn
from torch.cuda.amp import autocast


class DeepDaze(nn.Module):
    def __init__(
            self,
            perceptor,
            clip_norm,
            batch_size=64,
            loss_coef=100,
            saturate_limit=0.75,
            lower_bound_cutout=0.1,
            upper_bound_cutout=1.0,
            averaging_weight=0.3,
            dim_hidden=512,
            n_layers=20,
            image_size=128,
            w0=30,
            w0_initial=30,
    ):
        super().__init__()
        self.perceptor = perceptor
        self.clip_norm = clip_norm
        self.batch_size = batch_size
        self.image_size = image_size
        self.loss_coef = loss_coef
        self.saturate_limit = saturate_limit
        self.lower_bound_cutout = lower_bound_cutout
        self.upper_bound_cutout = upper_bound_cutout
        self.averaging_weight = averaging_weight

        self.generator = SirenNet(dim_in=2,
                                  dim_hidden=dim_hidden,
                                  dim_out=3,
                                  n_layers=n_layers,
                                  image_size=image_size,
                                  w0=w0,
                                  w0_initial=w0_initial
                                  )

    @staticmethod
    def siren_norm(img):
        return ((img + 1) * 0.5).clamp(0.0, 1.0)

    @staticmethod
    def rand_cutout(image, size):
        width = image.shape[-1]
        min_offset = 0
        max_offset = width - size
        offset_x = torch.randint(min_offset, max_offset, ())
        offset_y = torch.randint(min_offset, max_offset, ())
        cutout = image[:, :, offset_x:offset_x + size, offset_y:offset_y + size]
        return cutout

    @staticmethod
    def interpolate(image, size):
        return F.interpolate(image, (size, size), mode='bilinear', align_corners=False)

    def forward(self, text_embed, return_loss=True):
        out = self.siren_norm(self.generator())

        if not return_loss:
            return out

        # Sample cutout sizes between the lower and upper bounds
        sizes = torch.randint(int(self.lower_bound_cutout * self.image_size),
                              int(self.upper_bound_cutout * self.image_size),
                              (self.batch_size,))

        # Create normalized random cutouts
        image_pieces = [self.interpolate(self.rand_cutout(out, size), 224) for size in sizes]

        # Normalize the image pieces
        image_pieces = torch.cat([self.clip_norm(piece) for piece in image_pieces])

        # Calculate the embedding of the image
        with autocast(enabled=False):
            image_embed = self.perceptor.encode_image(image_pieces)

        # Calculate the loss
        # Loss over averaged features of cutouts
        avg_image_embed = image_embed.mean(dim=0).unsqueeze(0)
        averaged_loss = -self.loss_coef * torch.cosine_similarity(text_embed, avg_image_embed, dim=-1).mean()
        # Loss over all cutouts
        general_loss = -self.loss_coef * torch.cosine_similarity(text_embed, image_embed, dim=-1).mean()
        # Merge the losses
        loss = averaged_loss * self.averaging_weight + general_loss * (1 - self.averaging_weight)

        return out, loss
