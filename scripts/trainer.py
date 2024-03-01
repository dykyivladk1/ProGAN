import torch
import torch.nn as nn

from generator import Generator
from discriminator import Discriminator
from utils import *


import torchvision.transforms as tr
from torchvision.utils import save_image

import albumentations as A
from albumentations.pytorch import ToTensorV2


from math import log2


from ds import CustomImageDataset

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")


device = torch.device("mps")


z_dim = 100
in_features = 256

gen = Generator(z_dim, in_features).to(device)
dis = Discriminator(z_dim, in_features).to(device)


lr = 1e-3
lambda_gp = 10

batch_sizes = [32, 32, 32, 16, 16, 16, 16, 8, 4]
progressive_epochs = [30] * len(batch_sizes)

scaler_dis = torch.cuda.amp.GradScaler()
scaler_gen = torch.cuda.amp.GradScaler()

opt_gen = torch.optim.Adam(gen.parameters(),
                           lr = lr, betas = (0.5, 0.999))

opt_dis = torch.optim.Adam(dis.parameters(),
                           lr = lr, betas = (0.5, 0.999))


start_train_at_img_size = 128
fixed_noise = torch.randn(8, z_dim, 1, 1).to(device)

step = int(log2(start_train_at_img_size // 4))

gen.train()
dis.train()


import argparse

parser = argparse.ArgumentParser(description = "Train a ProGAN")
parser.add_argument("--train_dir", type = str, default = "./data/")
args = parser.parse_args()




if __name__ == "__main__":
    for num_epochs in progressive_epochs[step:]:
        alpha = 1e-5
        image_size = 4 * 2 ** step
        transform = tr.Compose([
            tr.Resize((image_size, image_size)),
            tr.ToTensor(),
            tr.RandomHorizontalFlip(p = 0.5),
            tr.Normalize([0.5 for _ in range(3)], [0.5 for _ in range(3)])
            
        ])
        
        b_size = batch_sizes[int(log2(image_size / 4))]
        train_ds = CustomImageDataset(
            image_dir = args.train_dir,
            transform = transform
        )
        train_dl = torch.utils.data.DataLoader(
            train_ds, 32, shuffle = True, pin_memory = True, num_workers = 4
        )
        pbar = tqdm(train_dl, total=len(train_dl))
        for batch_idx, (real) in enumerate(pbar, start=1):  
            real = real.to(device)
            cur_batch_size = real.shape[0]
            noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
            with torch.cuda.amp.autocast():
                fake = gen(noise, alpha, step)
                critic_real = dis(real, alpha, step)
                critic_fake = dis(fake.detach(), alpha, step)
                gp = gradient_penalty(dis, real, fake, alpha, step, device=device)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp + (0.001 * torch.mean(critic_real ** 2))

            opt_dis.zero_grad()
            scaler_dis.scale(loss_critic).backward()
            scaler_dis.step(opt_dis)
            scaler_dis.update()

            with torch.cuda.amp.autocast():
                gen_fake = dis(fake, alpha, step)
                loss_gen = -torch.mean(gen_fake)

            opt_gen.zero_grad()
            scaler_gen.scale(loss_gen).backward()
            scaler_gen.step(opt_gen)
            scaler_gen.update()

            alpha += cur_batch_size / ((progressive_epochs[step] * 0.5) * len(train_ds))
            alpha = min(alpha, 1)

            if batch_idx % 500 == 0:
                with torch.no_grad():
                    fixed_fakes = gen(fixed_noise, alpha, step) * 0.5 + 0.5
            
            pbar.set_postfix(gp=gp.item(), loss_critic=loss_critic.item())
            
    step += 1