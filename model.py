from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gan import Discriminator, Generator


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvModel:
    def __init__(self,
            latent_dim : int = 100, 
            lbl        : int = 0,
            img_shape  : Union[List[int], Tuple[int]] = [1,28,28],
            lamda      : float = 0.1,
            lr         : float = 2e-4,
            momentum   : Union[List[float], Tuple[float]] = (0.5, 0.999),
            device     : Union[str, torch.device] = 'cuda',
        ):

        self._latent_dim = latent_dim
        self._lamda  = lamda
        self._device = device
        self._lbl    = lbl
        
        self.generator = Generator(latent_dim=latent_dim, img_shape=img_shape).to(device)
        self.discriminator = Discriminator(latent_dim=latent_dim, img_shape=img_shape).to(device)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)


        self.adv_loss = nn.BCELoss().to(device)
        self._optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=momentum)
        self._optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=momentum)


    def train_step(self, real_imgs: torch.Tensor, target_model : nn.Module):
        # target model should be a pretrained model
        # in eval mode, with all parameters freezed

        B,C,H,W = real_imgs.shape
        
        valid = torch.Tensor(B, 1, device=self._device).fill_(1.0)
        fake = torch.Tensor(B, 1, device=self._device).fill_(0.0)
        z = torch.normal(0.0, 1.0, size=[B, self._latent_dim], device=self._device)
        
        # generator
        self._optimizer_G.zero_grad()
        gen_imgs : torch.Tensor = self.generator(z)
        g_loss : torch.Tensor = self.adv_loss(self.discriminator(gen_imgs), valid)
        
        # minimize the gap between the true label and other labels
        logits : torch.Tensor = target_model(gen_imgs)
        probs_model = F.softmax(logits, dim=1)
        onehot_label = torch.tensor([1 if idx == self._lbl else 0 for idx in range(logits.shape[1])], dtype=torch.float32, device=self._device)
        real = torch.sum(onehot_label * probs_model, dim=1)
        other, _ = torch.max((1 - onehot_label) * probs_model - onehot_label * 10000, dim=1)
        zeros = torch.zeros_like(other)
        loss_adv = torch.max(real - other, zeros)
        loss_adv = torch.sum(loss_adv)
        
        compound_loss = g_loss + self._lamda * loss_adv
        compound_loss.backward()
        self._optimizer_G.step()
        
        # discriminator
        self._optimizer_D.zero_grad()
        real_loss = self.adv_loss(self.discriminator(real_imgs), valid)
        fake_loss = self.adv_loss(self.discriminator(gen_imgs.detach()), fake)
        d_loss : torch.Tensor = (real_loss + fake_loss) / 2
        d_loss.backward()
        self._optimizer_D.step()

        return compound_loss.item(), d_loss.item()