import argparse
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset import MINISTLableWiseInMemoDataset
from model import AdvModel
from resnet18 import Resnet18

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--lbl", type=int, default=0, help="label of images used for training")
parser.add_argument("--lamda", type=float, default=0.1, help="default weight of adv loss")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SHAPE = (opt.channels, opt.img_size, opt.img_size)

MODEL_PATH = "./model/"
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

GRAYSCALE = True
NUM_CLASSES = 10


TARGET_MODEL_PATH = "./target_model/resnet18_minst_best.pth"
print(f"Loading target model from {TARGET_MODEL_PATH}")
target_model =  Resnet18(NUM_CLASSES,GRAYSCALE)
target_model.load_state_dict(torch.load(TARGET_MODEL_PATH))
target_model.to(DEVICE)
target_model.eval()


mnist_dataset = MINISTLableWiseInMemoDataset('./data', lbl = opt.lbl, train=True, transform=transforms.ToTensor())
train_dataloader = DataLoader(mnist_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)


advmodel = AdvModel(latent_dim=opt.latent_dim, lbl=opt.lbl, img_shape=IMAGE_SHAPE, \
                    lamda=opt.lamda, lr=opt.lr, momentum=(opt.b1, opt.b2), device=DEVICE)


for epoch in range(opt.n_epochs):
    batch_compound_loss = 0
    batch_d_loss = 0
    for i, (imgs, _) in enumerate(train_dataloader):
        imgs = imgs.to(DEVICE)
        compound_loss, d_loss = advmodel.train_step(imgs, target_model=target_model)
        batch_compound_loss += compound_loss
        batch_d_loss += d_loss
    
    # print statistics
    print(f"Epoch {epoch}: \
          Average Compound g_loss: {batch_compound_loss/len(train_dataloader)} \
          Average d_loss: {batch_d_loss/len(train_dataloader)}")
    
    # save model
    if epoch%20 == 0:
        filename = MODEL_PATH  + f'advGenerator_lbl={opt.lbl}_lam={opt.lamda}_lr={opt.lr}_bz={opt.batch_size}_epoch={epoch}.pth'
        torch.save(advmodel.generator.state_dict(), filename)