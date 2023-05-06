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

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import gan

from config import LABELS, LOSSES_PATH, MODELS_PATH




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
                target_model,
                device='cuda',
                n_channels=3,
                target_img_lbl=11,
                lr=5e-5, 
                l_inf_bound=0.05, 
                alpha=10, 
                beta=1, 
                gamma=1000,
                n_steps_D=1, 
                n_steps_G=1,
                C=0.1,
                attack_mode=-1,
                is_relativistic=False
            ):
        self.device = device
        self.target_model_class = target_model
        self.target_model = self.target_model_class.model
        
        self.target_img_lbl = target_img_lbl

        self.lr = lr

        self.l_inf_bound = l_inf_bound

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.n_steps_D = n_steps_D
        self.n_steps_G = n_steps_G

        self.is_relativistic = is_relativistic

        self.c = C

        self.attack_mode = attack_mode

        self.G = gan.Generator(n_channels, n_channels).to(device)
        self.D = gan.Discriminator(n_channels).to(device)

        # initialize all weights
        self.G.apply(init_weights)
        self.D.apply(init_weights)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.lr)
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.lr)

        self.up_sample = nn.Upsample(scale_factor=2)


    def train_batch(self, x):
        self.G.train()
        self.D.train()
        # optimize D
        for i in range(self.n_steps_D):
            perturbation = self.G(x)
            perturbation = torch.clamp(perturbation, -self.l_inf_bound, self.l_inf_bound)

            adv_images = perturbation + x
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
            # batch_size, numOfAnchor, 4 + 1 + 80
            if self.attack_mode == -2:
                # attack for object existence
                logits1 = logits_model[:, :, 4]
                logits2 = logits_model[:, :, self.target_img_lbl + 5]
                loss_adv = torch.log(logits1).max(1)[0].sum() + torch.log(logits2).max(1)[0].sum()
            elif self.attack_mode == -1:
                # untargeted attack
                logits1 = logits_model[:, :, self.target_img_lbl + 5]
                loss_adv = torch.log(logits1).max(1)[0].sum()
                # loss_adv = torch.max(torch.zeros_like(loss_adv), loss_adv - math.log(0.5))
            else:
                # targeted attack with target label = attack_mode (0 - 79)
                logits1 = logits_model[:, :, self.target_img_lbl + 5]
                logits2 = logits_model[:, :, self.attack_mode + 5]
                loss_adv = torch.log(logits1).max(1)[0].sum()
                loss_adv = 1.25*torch.max(torch.zeros_like(loss_adv), loss_adv - math.log(0.5))
                loss_adv -= torch.log(logits2).max(1)[0].sum()

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
        best_epoch = -1
        best_norm = float('inf')
        best_labels = []
        best_image = None
        best_obj_existence = float('inf')
        best_score = None
        
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

            """save generator"""
            # torch.save(self.G.state_dict(), '{}G_epoch_{}.pth'.format(MODELS_PATH, str(epoch)))

            if epoch%2 == 0:
                res = self.evaluate(target_img)
                labels = res["evaluation"][1][0]['labels'][:5]
                scores = res["evaluation"][1][0]['scores'][:5]
                norm = res["perturbation_norm"].detach().clone()
                obj_existence = res["evaluation"][0][:, :, 4].max(1)[0].item()
                print("Check labels",labels)
                print("Check scores",scores)
                print("Check norm",norm)
                print("Check obj_existence",obj_existence)
                if self.attack_mode == -2: # attack for object existence
                    if obj_existence <= 0.5 and norm < best_norm:
                        best_norm = norm
                        best_labels = labels
                        best_epoch = epoch
                        best_score = scores
                        best_image = res["adv_image"].detach().clone()
                        best_obj_existence = obj_existence
                elif self.attack_mode == -1: # untargeted attack
                    if labels[0] != 'stopsign' and norm < best_norm:
                        best_norm = norm
                        best_labels = labels
                        best_epoch = epoch
                        best_score = scores
                        best_image = res["adv_image"].detach().clone()
                else: # targeted attack
                    if labels[0] == LABELS[self.attack_mode] and norm < best_norm:
                        if 'stopsign' in labels and scores[labels.index('stopsign')] > 0.5:
                            continue
                        best_norm = norm
                        best_labels = labels
                        best_epoch = epoch
                        best_score = scores
                        best_image = res["adv_image"].detach().clone()

        """plot losses"""
        # plt.figure()
        # plt.plot(loss_D)
        # plt.savefig(LOSSES_PATH + f'loss_D_{self.attack_mode}_{self.alpha}_{self.beta}_{self.gamma}.png')

        # plt.figure()
        # plt.plot(loss_G)
        # plt.savefig(LOSSES_PATH + f'loss_G_{self.attack_mode}_{self.alpha}_{self.beta}_{self.gamma}.png')

        # plt.figure()
        # plt.plot(loss_adv)
        # plt.savefig(LOSSES_PATH + f'loss_adv_{self.attack_mode}_{self.alpha}_{self.beta}_{self.gamma}.png')

        # plt.figure()
        # plt.plot(loss_G_gan)
        # plt.savefig(LOSSES_PATH + f'loss_G_gan_{self.attack_mode}_{self.alpha}_{self.beta}_{self.gamma}.png')

        # plt.figure()
        # plt.plot(loss_hinge)
        # plt.savefig(LOSSES_PATH + f'loss_hinge_{self.attack_mode}_{self.alpha}_{self.beta}_{self.gamma}.png')

        return best_norm, best_epoch, best_labels, best_image, best_score, best_obj_existence


    def evaluate(self, target_img, conf_thres=0.0, iou_thres=0.0, max_det=50, preturb=True):
        """generate adv images and evaluate by target model"""
        self.G.eval()
        target_img = target_img.to(self.device)
        if preturb:
            with torch.no_grad():
                perturbation = self.G(target_img).detach()
                perturbation = torch.clamp(perturbation, -self.l_inf_bound, self.l_inf_bound)
                adv_images = perturbation + target_img
                adv_images = torch.clamp(adv_images, 0, 1)
                return {
                    "adv_image": adv_images, 
                    "evaluation": self.target_model_class.get_inference(self.up_sample(adv_images.detach()),conf_thres,iou_thres, max_det),
                    "perturbation_norm": torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
                }
        else:
            adv_images = target_img
            return {
                "evaluation": self.target_model_class.get_inference(self.up_sample(adv_images.detach()),conf_thres,iou_thres, max_det)
            }
        