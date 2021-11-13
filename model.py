import math
import numpy as np
import random
import functools
import operator

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import kornia.augmentation as K
import kornia.filters as k
import kornia.geometry.transform as ktf

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class To4d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.view(*input.size(),1,1)

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )
    def get_latent(self, style):
        style = self.modulation(style)
        return style

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        #style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        style = style.view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        device = image.device
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        try:
            output = image + self.weight * noise
        except:
            # reproducible noise when differeing spatial
            old_width = noise.size(3)
            batch, _, height, width = image.shape
            gen = torch.Generator()
            gen = gen.manual_seed(width)
            #noise = torch.randn(*image.shape, generator=gen).cuda()
            out_noise = torch.zeros([batch,1,height,width]).normal_(generator=gen).to(device)
            out_noise[..., :old_width] = noise
            out_noise[..., -old_width:] = noise
            #noise = image.new_empty(batch, 1, height, width).normal_(generator=gen)
            output = image + self.weight * out_noise
        return output

    #def forward(self, image, noise=None):
    #    if noise is None:
    #        batch, _, height, width = image.shape
    #        noise = image.new_empty(batch, 1, height, width).normal_()

    #    output = image + self.weight * noise
    #    return output

class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def get_latent(self, style):
        return self.conv.get_latent(style)
    def forward(self, input, style, noise=None):
        out_t = self.conv(input, style)
        out = self.noise(out_t, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out, out_t


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def get_latent(self, style):
        return self.conv.get_latent(style)
    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn( n_latent, self.style_dim, device=self.input.input.device)
        latent = self.get_latent(latent_in)#.mean(0, keepdim=True)
        latent = [latent[i].mean(0, keepdim=True) for i in range(len(latent))]

        return latent

    def get_w(self, input):
        device = self.input.input.device
        latent = self.style(input)
        latent = fused_leaky_relu(latent, torch.zeros_like(latent).to(device), 5.)
        return latent

    def get_latent(self, input, is_latent=False, truncation=1, mean_latent=None):
        output = []
        if not is_latent:
            latent = self.style(input)
            latent = latent.unsqueeze(1).repeat(1, self.n_latent, 1) #[B, 14, 512]
        else:
            latent = input
        output.append(self.conv1.get_latent(latent[:, 0]))
        output.append(self.to_rgb1.get_latent(latent[:, 1]))

        i = 1
        for conv1, conv2, to_rgb in zip(self.convs[::2], self.convs[1::2], self.to_rgbs):
            output.append(conv1.get_latent(latent[:, i]))
            output.append(conv2.get_latent(latent[:, i+1]))
            output.append(to_rgb.get_latent(latent[:, i+2]))
            i += 2

        if truncation < 1 and mean_latent is not None:
            output = [mean_latent[i] + truncation * (output[i] - mean_latent[i]) for i in range(len(output))]
            
        return output

    def patch_swap(self, latent1, latent2, coord, swap=True):
        noise = [getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)]

        coord = torch.tensor(coord).long()
        # this is the case of single image patch swap.
        if coord.ndim == 3:
            coord = coord[0]
        
        out = self.input(latent1[0])

        out1, _ = self.conv1(out, latent1[0], noise=noise[0])
        out2, _ = self.conv1(out, latent2[0], noise=noise[0])

        skip1 = self.to_rgb1(out1, latent1[1])
        skip2 = self.to_rgb1(out2, latent2[1])

        i = 2
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):

            out1, _ = conv1(out1, latent1[i], noise=noise1)
            out2, _ = conv1(out2, latent2[i], noise=noise1)

            out1, _ = conv2(out1, latent1[i+1], noise=noise2)
            out2, _ = conv2(out2, latent2[i+1], noise=noise2)

            skip1 = to_rgb(out1, latent1[i+2], skip1)
            skip2 = to_rgb(out2, latent2[i+2], skip2)

            # 2 for more blending less texture accuracy, 5 for more exact transfer
            if i == 5:
            #if i < 20:
            #if i == 2:
                h,w = out1.size()[2:]
                scale = 256 // h
                scaled_coord = torch.floor(coord / scale).long()
                x1, y1, w1, h1 = scaled_coord[0]
                x2, y2, w2, h2 = scaled_coord[1]

                w = max(w1,w2)
                h = max(h1,h2)

                out1[:,:,y1:y1+h,x1:x1+w] = out2[:,:,y2:y2+h,x2:x2+w]
                if swap:
                    out1[:,:,y2:y2+h,x2:x2+w] = out2[:,:,y1:y1+h,x1:x1+w]

            i += 3

        image = skip1.clamp(-1,1)
        return image

    def singan(self, latent, mode):
        noise = [None] * self.num_layers
#         noise = [getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)]
        
        out = self.input(latent[0])        
        out, _ = self.conv1(out, latent[0], noise=noise[0])
        skip = self.to_rgb1(out, latent[1])


        i = 2
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):

            out, _ = conv1(out, latent[i], noise=noise1)
            out, _ = conv2(out, latent[i+1], noise=noise2)
            skip = to_rgb(out, latent[i+2], skip)
            
            if i == 2:
                h,w = out.size()[2:]
                if mode == 'horizontal': 
                    out_h = h
                    out_w = w*2
                    out = F.interpolate(out, size=(out_h, out_w), mode='nearest')
                    skip = F.interpolate(skip, size=(out_h, out_w), mode='nearest')

                elif mode == 'vertical':
                    out_h = h*2
                    out_w = w
                    out = F.interpolate(out, size=(out_h, out_w), mode='nearest')
                    skip = F.interpolate(skip, size=(out_h, out_w), mode='nearest')
                else:
                    pad_type = mode
                    npad = w // 2
                    out = F.pad(out, (npad,npad,0,0), pad_type)
                    skip = F.pad(skip, (npad,npad,0,0), pad_type)

                    #out = torch.roll(out, npad, 3)
                    #skip = torch.roll(skip, npad, 3)
            i += 3

        image = skip.clamp(-1,1)
        return image



    def blend_bbox(self, latent1, latent2, coord, model_type, num_blend=99):
        noise = [getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)]

        if model_type == 'face':
            pose_align = True
            pose_num = 4
        else:
            pose_align = False

        x, y, w, h = coord[0]

        device = self.input.input.device
        mask = torch.zeros([1,1,256,256]).to(device)
        mask[..., y:y+h, x:x+w] = 1
        k_h = h//2
        k_h += k_h%2+1
        k_h = int(k_h)
        k_w = w//2
        k_w += k_w%2+1
        k_w = int(k_w)

        mask = k.gaussian_blur2d(mask, (k_h, k_w), sigma=(k_h, k_w))
        #mask = k.gaussian_blur2d(mask, (53,125), sigma=(53,125))
        
        out = self.input(latent1[0])
        
        out1, _ = self.conv1(out, latent1[0], noise=noise[0])
        out2, _ = self.conv1(out, latent2[0], noise=noise[0])
        alpha = F.interpolate(mask, size=out1.size()[2:], mode='bilinear')
        out = (1-alpha)*out1 + alpha*out2

        skip1 = self.to_rgb1(out, latent1[1])
        skip2 = self.to_rgb1(out, latent2[1])
        alpha = F.interpolate(mask, size=skip1.size()[2:], mode='bilinear')
        if pose_align: alpha.zero_()
        skip = (1-alpha)*skip1 + alpha*skip2


        i = 2
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):

            out1, _ = conv1(out, latent1[i], noise=noise1)
            out2, _ = conv1(out, latent2[i], noise=noise1)
            alpha = F.interpolate(mask, size=out1.size()[2:], mode='bilinear')
            if i > num_blend or (pose_align and i < pose_num): alpha.zero_()
            out = (1-alpha)*out1 + alpha*out2
            i += 1

            out1, _ = conv2(out, latent1[i], noise=noise2)
            out2, _ = conv2(out, latent2[i], noise=noise2)
            alpha = F.interpolate(mask, size=out1.size()[2:], mode='bilinear')
            if i > num_blend or (pose_align and i < pose_num): alpha.zero_()
            out = (1-alpha)*out1 + alpha*out2
            i += 1

            skip1 = to_rgb(out, latent1[i], skip)
            skip2 = to_rgb(out, latent2[i], skip)
            alpha = F.interpolate(mask, size=skip1.size()[2:], mode='bilinear')
            if i > num_blend or (pose_align and i < pose_num): alpha.zero_()
            skip = (1-alpha)*skip1 + alpha*skip2
            i += 1

        image = skip.clamp(-1,1)
        return image
    
    def blend(self, latent1, latent2, mode):
        noise = [getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)]
        device = self.input.input.device

        assert mode in ('vertical', 'horizontal')
        if mode == 'vertical':
            view_size = (1,1,-1,1)
        else:
            view_size = (1,1,1,-1)
            
        out = self.input(latent1[0])
        
        out1, _ = self.conv1(out, latent1[0], noise=noise[0])
        out2, _ = self.conv1(out, latent2[0], noise=noise[0])
        alpha = torch.zeros([out1.size(2)])
        pad = out1.size(2)//4
        alpha[-pad:] = 1
        alpha[pad:-pad] = torch.linspace(0,1,alpha.size(0)-2*pad)
        alpha = alpha.view(*view_size).expand_as(out1).to(device)
        out = (1-alpha)*out1 + alpha*out2

        skip1 = self.to_rgb1(out, latent1[1])
        skip2 = self.to_rgb1(out, latent2[1])
        alpha = torch.zeros([skip1.size(2)])
        pad = skip1.size(2)//4
        alpha[-pad:] = 1
        alpha[pad:-pad] = torch.linspace(0,1,alpha.size(0)-2*pad)
        alpha = alpha.view(*view_size).expand_as(skip1).to(device)
        skip = (1-alpha)*skip1 + alpha*skip2


        i = 2
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):

            out1, _ = conv1(out, latent1[i], noise=noise1)
            out2, _ = conv1(out, latent2[i], noise=noise1)
            alpha = torch.zeros([out1.size(2)])
            pad = out1.size(2)//4
            alpha[-pad:] = 1
            alpha[pad:-pad] = torch.linspace(0,1,alpha.size(0)-2*pad)
            alpha = alpha.view(*view_size).expand_as(out1).to(device)
            out = (1-alpha)*out1 + alpha*out2

            out1, _ = conv2(out, latent1[i+1], noise=noise2)
            out2, _ = conv2(out, latent2[i+1], noise=noise2)
            alpha = torch.zeros([out1.size(2)])
            pad = out1.size(2)//4
            alpha[-pad:] = 1
            alpha[pad:-pad] = torch.linspace(0,1,alpha.size(0)-2*pad)
            alpha = alpha.view(*view_size).expand_as(out1).to(device)
            out = (1-alpha)*out1 + alpha*out2

            skip1 = to_rgb(out, latent1[i+2], skip)
            skip2 = to_rgb(out, latent2[i+2], skip)
            alpha = torch.zeros([skip1.size(2)])
            pad = skip1.size(2)//4
            alpha[-pad:] = 1
            alpha[pad:-pad] = torch.linspace(0,1,alpha.size(0)-2*pad)
            alpha = alpha.view(*view_size).expand_as(skip1).to(device)
            skip = (1-alpha)*skip1 + alpha*skip2

            i += 3

        image = skip.clamp(-1,1)
        return image
    
    
    def merge_extension(self, latent1, latent2):
        noise = [getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)]
        device = self.input.input.device
        
        out = self.input(latent1[0])
        out1, _ = self.conv1(out, latent1[0], noise=noise[0])
        out2, _ = self.conv1(out, latent2[0], noise=noise[0])
        out = torch.cat([out1, out2], 3)

        skip1 = self.to_rgb1(out, latent1[1])
        skip2 = self.to_rgb1(out, latent2[1])

        alpha = torch.zeros([skip1.size(3)])
        pad = skip1.size(3)//4
        alpha[-pad:] = 1
        alpha[pad:-pad] = torch.linspace(0,1,alpha.size(0)-2*pad)
        alpha = alpha.view(1,1,1,-1).expand_as(skip1).to(device)
        skip = (1-alpha)*skip1 + alpha*skip2


        i = 2
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):


            out1, _ = conv1(out, latent1[i], noise=noise1)
            out2, _ = conv1(out, latent2[i], noise=noise1)
            alpha = torch.zeros([out1.size(3)])
            pad = out1.size(3)//4
            alpha[-pad:] = 1
            alpha[pad:-pad] = torch.linspace(0,1,alpha.size(0)-2*pad)
            alpha = alpha.view(1,1,1,-1).expand_as(out1).to(device)
            out = (1-alpha)*out1 + alpha*out2

            out1, _ = conv2(out, latent1[i+1], noise=noise2)
            out2, _ = conv2(out, latent2[i+1], noise=noise2)
            alpha = torch.zeros([out1.size(3)])
            pad = out1.size(3)//4
            alpha[-pad:] = 1
            alpha[pad:-pad] = torch.linspace(0,1,alpha.size(0)-2*pad)
            alpha = alpha.view(1,1,1,-1).expand_as(out1).to(device)
            out = (1-alpha)*out1 + alpha*out2

            skip1 = to_rgb(out, latent1[i+2], skip)
            skip2 = to_rgb(out, latent2[i+2], skip)
            alpha = torch.zeros([skip1.size(3)])
            pad = skip1.size(3)//4
            alpha[-pad:] = 1
            alpha[pad:-pad] = torch.linspace(0,1,alpha.size(0)-2*pad)
            alpha = alpha.view(1,1,1,-1).expand_as(skip1).to(device)
            skip = (1-alpha)*skip1 + alpha*skip2

            i += 3

        image = skip.clamp(-1,1)
        return image

    def merge(self, latent1, latent2):
        noise = [getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)]
        device = self.input.input.device
        
        out = self.input(latent1[0])
        out1, _ = self.conv1(out, latent1[0], noise=noise[0])
        out2, _ = self.conv1(out, latent2[0], noise=noise[0])
        out = torch.cat([out1, out2], 3)

        skip1 = self.to_rgb1(out, latent1[1])
        skip2 = self.to_rgb1(out, latent2[1])
        alpha = torch.linspace(0,1, skip1.size(3)).view(1,1,1,-1).expand_as(skip1).to(device)
        skip = (1-alpha)*skip1 + alpha*skip2


        i = 2
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):


            out1, _ = conv1(out, latent1[i], noise=noise1)
            out2, _ = conv1(out, latent2[i], noise=noise1)
            alpha = torch.linspace(0,1, out1.size(3)).view(1,1,1,-1).expand_as(out1).to(device)
            out = (1-alpha)*out1 + alpha*out2

            out1, _ = conv2(out, latent1[i+1], noise=noise2)
            out2, _ = conv2(out, latent2[i+1], noise=noise2)
            alpha = torch.linspace(0,1, out1.size(3)).view(1,1,1,-1).expand_as(out1).to(device)
            out = (1-alpha)*out1 + alpha*out2

            skip1 = to_rgb(out, latent1[i+2], skip)
            skip2 = to_rgb(out, latent2[i+2], skip)
            alpha = torch.linspace(0,1, skip1.size(3)).view(1,1,1,-1).expand_as(skip1).to(device)
            skip = (1-alpha)*skip1 + alpha*skip2

            i += 3

        image = skip.clamp(-1,1)
        return image
    
    def forward(
        self,
        styles,
        stop_idx=99,
        is_cluster=False,
        noise=None,
        randomize_noise=False,
    ):

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]
                
        outputs = []
        idx_count = 0
        
        latent = styles
        out = self.input(latent[0])
        outputs.append([out, out])
        if idx_count == stop_idx:
            return outputs
        
        out, out_t = self.conv1(out, latent[idx_count], noise=noise[0])
        outputs.append([out_t, out])
        idx_count += 1
        if idx_count == stop_idx:
            return outputs

        skip = self.to_rgb1(out, latent[idx_count])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            outputs.append([out_t, out])
            idx_count += 1
            if idx_count == stop_idx:
                return outputs

            out, out_t = conv1(out, latent[idx_count], noise=noise1)
            outputs.append([out_t, out])
            idx_count += 1
            if idx_count == stop_idx:
                return outputs

            out, out_t = conv2(out, latent[idx_count], noise=noise2)
            outputs.append([out_t, out])
            idx_count += 1
            if idx_count == stop_idx:
                return outputs

            skip = to_rgb(out, latent[idx_count], skip)

            i += 2

        image = skip.clamp(-1,1)
        return image, outputs


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)



class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        #group = batch
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out
