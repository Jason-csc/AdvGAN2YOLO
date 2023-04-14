import os

import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import AdvModel


BATCH_SIZE = 128
EPOCH = 60

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "./model/"
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
train_dataloader = DataLoader(mnist_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)


advmodel = AdvModel(device=DEVICE)


for epoch in range(EPOCH):
    batch_compound_loss = 0
    batch_d_loss = 0
    for i, (imgs, _) in enumerate(train_dataloader):
        imgs = imgs.to(DEVICE)
        compound_loss, d_loss = advmodel.train_step(imgs, target_model=)
        batch_compound_loss += compound_loss
        batch_d_loss += batch_d_loss
    
    # print statistics
    print(f"Epoch {epoch}: \
          Average Compound g_loss: {batch_compound_loss/len(train_dataloader)} \
          Average d_loss: {batch_d_loss/len(train_dataloader)}")
    
    # save model
    if epoch%20 == 0:
        filename = MODEL_PATH  + f'advGenerator_epoch_{epoch}.pth'
        torch.save(advmodel.generator.state_dict(), filename)
        

         







def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):

            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.0001)
            if epoch == 80:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.00001)
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = \
                    self.train_batch(images, labels)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch))

            # save generator
            if epoch%20==0:
                netG_file_name = MODEL_PATH + 'netG_epoch_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)
