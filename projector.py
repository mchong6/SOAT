import argparse
import math
import os

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
import pickle
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import lpips
from model import Generator
from op import fused_leaky_relu
import numpy as np
from id_loss import IDLoss

import torchvision
from util import ensure_checkpoint_exists

style2list_len = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 
                  512, 512, 512, 512, 512, 256, 256, 256, 128, 128, 
                  128, 64, 64, 64, 32, 32]


def style2list(s):
    output = []
    count = 0 
    for size in style2list_len:
        output.append(s[:, count:count+size])
        count += size
    return output

def list2style(s):
    return torch.cat(s, 1)

def gaussian_loss(v, gt_mean, gt_cov_inv):
    # [B, 9088]
    loss = (v-gt_mean) @ gt_cov_inv @ (v-gt_mean).transpose(1,0)
    return loss.mean()

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8: break 
            noise = noise.reshape([1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to('cpu')
        .numpy()
    )


def projection(img, name, size=1024, lr=0.05, step=1000, device='cuda'):
    # img: PIL images

    # download all requires files
    ensure_checkpoint_exists('inversion_stats.npz')
    ensure_checkpoint_exists('stylegan2-ffhq-config-f.pt')
    ensure_checkpoint_exists('model_ir_se50.pt')

    data = np.load('inversion_stats.npz')
    gt_mean = torch.tensor(data['mean']).to(device).view(1,-1).float()
    gt_cov_inv = torch.tensor(data['cov']).to(device)
    
    # Only take diagonals
    mask = torch.eye(*gt_cov_inv.size()).to(device)
    gt_cov_inv = torch.inverse(gt_cov_inv*mask).float()

    resize = min(size, 256)

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    imgs = transform(img).unsqueeze(0).to(device)

    g_ema = Generator(size, 512, 8)
    g_ema.load_state_dict(torch.load('stylegan2-ffhq-config-f.pt')['g_ema'], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)
    
    id_criterion = IDLoss().to(device).eval()

    with torch.no_grad():
        latent_mean = g_ema.mean_latent(50000)
        latent_in = list2style(latent_mean)

    percept = lpips.LPIPS(net='vgg').to(device)
    latent_in.requires_grad = True

    optimizer = optim.Adam([latent_in], lr=lr, betas=(0.9, 0.999))

    pbar = tqdm(range(step))
    latent_path = []

    for i in pbar:
        t = i / step
        lr = get_lr(t, lr)
        latent_n = latent_in

        img_gen, _ = g_ema(style2list(latent_n))

        batch, channel, height, width = img_gen.shape

        if height > 256:
            img_gen = F.interpolate(img_gen, size=(256,256), mode='bilinear')

        p_loss = 10*percept(img_gen, imgs).mean()
        mse_loss = 1*F.mse_loss(img_gen, imgs)
        g_loss = 1e-3*gaussian_loss(latent_n, gt_mean, gt_cov_inv)
        id_loss = 1*id_criterion(img_gen, imgs)

        loss = p_loss + mse_loss + g_loss + id_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            latent_path.append(latent_in.detach().clone())

        pbar.set_description(
            (
                f'perceptual: {p_loss.item():.4f};'
                f' mse: {mse_loss.item():.4f}; gaussian: {g_loss.item():.4f} id: {id_loss:.4f}'
            )
        )

    result_file = {}

    latent_path.append(latent_in.detach().clone())
    img_gen, _ = g_ema(style2list(latent_path[-1]))

    filename = './inversion_codes/' + name + '.pt'

    img_ar = make_image(img_gen)
    result_file['latent'] = latent_in[0]

    torch.save(result_file, filename)


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--step', type=int, default=2000)
    parser.add_argument('files', metavar='FILES', nargs='+')

    args = parser.parse_args()


    img = Image.open(args.files[0])
    name = os.path.splitext(os.path.basename(args.files[0]))[0]
    projection(img, name, args.size, args.lr, args.step, device)
