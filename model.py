import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)


'''class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=7):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))'''


def add_sn(m):
    for name, c in m.named_children():
        m.add_module(name, add_sn(c))

    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Embedding)):
        return nn.utils.spectral_norm(m)
    else:
        return m


def _downsample(x):
    return F.avg_pool2d(x, kernel_size=2)


class OptimizedBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out, downsample=False):
        super(OptimizedBlock, self).__init__()
        self.downsample = downsample

        self.resi = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.learnable_sc = (dim_in != dim_out) or downsample
        if self.learnable_sc:
            self.sc = nn.Conv2d(dim_in, dim_out, kernel_size=1, padding=0, bias=True)

    def residual(self, x):
        h = x
        h = self.resi(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        h = x
        if self.downsample:
            h = _downsample(x)
        return self.sc(h)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResidualBlock_D(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out, downsample=False):
        super(ResidualBlock_D, self).__init__()
        self.downsample = downsample

        self.resi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True))

        self.learnable_sc = (dim_in != dim_out) or downsample

        if self.learnable_sc:
            self.sc = nn.Conv2d(dim_in, dim_out, kernel_size=1, padding=0, bias=True)

    def residual(self, x):
        h = x
        h = self.resi(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        output = self.residual(x) + self.shortcut(x)
        return output


class ImageDiscriminator(nn.Module):
    def __init__(self, conv_dim=64):
        super(ImageDiscriminator, self).__init__()
        self.ch = conv_dim
        self.relu = nn.ReLU(inplace=True)
        self.main = nn.Sequential(
            #(3, 256, 256) -> (3, 128, 128)
            ResidualBlock_D(3, 3, downsample=True),
            #(3, 128, 128) -> (3, 64, 64)
            ResidualBlock_D(3, 3, downsample=True),
            # (3, 64, 64) -> (64, 32, 32)
            OptimizedBlock(3, self.ch, downsample=True),
            # (64, 32, 32) -> (128, 16, 16)
            ResidualBlock_D(self.ch, self.ch * 2, downsample=True),
            # (128, 16, 16) -> (256, 8, 8)
            ResidualBlock_D(self.ch * 2, self.ch * 4, downsample=True),
            # (256, 8, 8) -> (512, 4, 4)
            ResidualBlock_D(self.ch * 4, self.ch * 8, downsample=True),
            # (512, 4, 4) -> (1024, 2, 2)
            ResidualBlock_D(self.ch * 8, self.ch * 16, downsample=True),
        )
        kernel_size = int(256/ np.power(2,7))
        self.classifier = nn.Linear(self.ch * 16, 1, bias=False)
        self.conv2 = nn.Conv2d(self.ch *16, 4, kernel_size=kernel_size, bias=False)

        # self.apply(weights_init)

    def forward(self, x):
        h = self.main(x)
        h = self.relu(h)
        out_cls = self.conv2(h)
        # (1024, 2, 2) -> (1024,)
        h = torch.sum(h, dim=(2, 3))
        #curr_dim = c.size(1)
        #out_cls = self.conv2(h)
        output = self.classifier(h)

        return output.view(-1),out_cls

