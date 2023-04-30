# Adversarial Attacks with AdvGAN and AdvRaGAN
# Copyright(C) 2020 Georgios (Giorgos) Karantonis
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Modified from https://github.com/mathcbc/advGAN_pytorch/blob/master/advGAN.py

import matplotlib

matplotlib.use('Agg')

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import gan

from SafeBench.safebench.agent.object_detection.utils.general import non_max_suppression

models_path = './checkpoints/AdvGAN/'
losses_path = './results/losses/'



def init_weights(m):
    '''
        Custom weights initialization called on G and D
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN_Attack:
    def __init__(
                self, 
                device, 
                target_model, 
                n_channels,
                target_lbl,
                lr, 
                l_inf_bound, 
                alpha, 
                beta, 
                gamma,
                n_steps_D, 
                n_steps_G,
                C,
                is_relativistic=False
            ):
        self.device = device
        self.target_model_class = target_model
        self.target_model = self.target_model_class.model
        
        self.target_lbl = target_lbl

        self.lr = lr

        self.l_inf_bound = l_inf_bound

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.n_steps_D = n_steps_D
        self.n_steps_G = n_steps_G

        self.is_relativistic = is_relativistic

        self.c = C

        self.G = gan.Generator(n_channels, n_channels).to(device)
        self.D = gan.Discriminator(n_channels).to(device)

        # initialize all weights
        self.G.apply(init_weights)
        self.D.apply(init_weights)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.lr)
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.lr)

        self.up_sample = nn.Upsample(scale_factor=2)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

        if not os.path.exists('{}/'.format(losses_path)):
            os.makedirs('{}/'.format(losses_path))


    def train_batch(self, x):
        # optimize D
        for i in range(self.n_steps_D):
            perturbation = self.G(x)

            adv_images = torch.clamp(perturbation, -self.l_inf_bound, self.l_inf_bound) + x
            adv_images = torch.clamp(adv_images, 0, 1)

            self.D.zero_grad()
            
            logits_real, pred_real = self.D(x)
            logits_fake, pred_fake = self.D(adv_images.detach())

            real = torch.ones_like(pred_real, device=self.device)
            fake = torch.zeros_like(pred_fake, device=self.device)



            if self.is_relativistic:
                # loss_D = F.binary_cross_entropy_with_logits(torch.squeeze(logits_real - logits_fake), real)
                loss_D_real = torch.mean((logits_real - torch.mean(logits_fake) - real)**2)
                loss_D_fake = torch.mean((logits_fake - torch.mean(logits_real) + real)**2)

                loss_D = (loss_D_fake + loss_D_real) / 2
            else:
                loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
                loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
                loss_D = loss_D_fake + loss_D_real

            loss_D.backward()
            self.optimizer_D.step()

        # optimize G
        for i in range(self.n_steps_G):
            self.G.zero_grad()

            # the Hinge Loss part of L
            perturbation_norm = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
            loss_hinge = torch.max(torch.zeros(1, device=self.device), perturbation_norm - self.c)

            # the Adv Loss part of L
            logits_model = self.target_model(self.up_sample(adv_images))
            ##TODO: add detach ??

            # batch_size, numOfAnchor, 4 + 1 + 80
            logits_model = logits_model[:, :, self.target_lbl + 5]
            loss_adv = torch.log(logits_model).mean(1).sum()


            # the GAN Loss part of L
            logits_real, pred_real = self.D(x)
            logits_fake, pred_fake = self.D(adv_images)

            if self.is_relativistic:
                # loss_G_gan = F.binary_cross_entropy_with_logits(torch.squeeze(logits_fake - logits_real), real)
                loss_G_real = torch.mean((logits_real - torch.mean(logits_fake) + real)**2)
                loss_G_fake = torch.mean((logits_fake - torch.mean(logits_real) - real)**2)
                
                loss_G_gan = (loss_G_real + loss_G_fake) / 2
            else:
                loss_G_gan = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))


            loss_G = self.gamma * loss_adv + self.alpha * loss_G_gan + self.beta * loss_hinge
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D.item(), loss_G.item(), loss_G_gan.item(), loss_hinge.item(), loss_adv.item()


    def train(self, target_img, epochs):
        loss_D, loss_G, loss_G_gan, loss_hinge, loss_adv = [], [], [], [], []
        
        for epoch in range(1, epochs+1):
            loss_D_sum, loss_G_sum, loss_G_gan_sum, loss_hinge_sum, loss_adv_sum = 0, 0, 0, 0, 0
            # for i, data in enumerate(train_dataloader, start=0):
            target_img = target_img.to(self.device)
            loss_D_batch, loss_G_batch, loss_G_fake_batch, loss_hinge_batch, loss_adv_batch = self.train_batch(target_img)

            loss_D_sum += loss_D_batch
            loss_G_sum += loss_G_batch
            loss_adv_sum += loss_adv_batch
            loss_G_gan_sum += loss_G_fake_batch
            loss_hinge_sum += loss_hinge_batch

            # print statistics
            print('Epoch {}: \nLoss D: {}, \nLoss G: {}, \n\t-Loss Adv: {}, \n\t-Loss G GAN: {}, \n\t-Loss Hinge: {}, \n'.format(
                epoch, loss_D_sum, loss_G_sum, loss_adv_sum, loss_G_gan_sum, loss_hinge_sum, 
            ))

            loss_D.append(loss_D_sum)
            loss_G.append(loss_G_sum)
            loss_adv.append(loss_adv_sum)
            loss_G_gan.append(loss_G_gan_sum)
            loss_hinge.append(loss_hinge_sum)

            # save generator
            torch.save(self.G.state_dict(), '{}G_epoch_{}.pth'.format(models_path, str(epoch)))

        plt.figure()
        plt.plot(loss_D)
        plt.savefig(losses_path + 'loss_D.png')

        plt.figure()
        plt.plot(loss_G)
        plt.savefig(losses_path + 'loss_G.png')

        plt.figure()
        plt.plot(loss_adv)
        plt.savefig(losses_path + 'loss_adv.png')

        plt.figure()
        plt.plot(loss_G_gan)
        plt.savefig(losses_path + 'loss_G_gan.png')

        plt.figure()
        plt.plot(loss_hinge)
        plt.savefig(losses_path + 'loss_hinge.png')


    def evaluate(self, target_img, conf_thres=0.0, iou_thres=0.0):
        #generate adv images and evaluate by target model
        self.G.eval()
        target_img = target_img.to(self.device)
        with torch.no_grad():
            perturbation = self.G(target_img)
            adv_images = torch.clamp(perturbation, -self.l_inf_bound, self.l_inf_bound) + target_img
            adv_images = torch.clamp(adv_images, 0, 1)
        #TODO: evaluate
        return {
            "adv_image":adv_images, 
            "evaluation": self.target_model_class.get_inference(self.up_sample(adv_images.detach()),conf_thres,iou_thres)
        }