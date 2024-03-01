import torch
import torch.nn as nn

from utils import *


factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]


class Discriminator(nn.Module):
    def __init__(self, z_dim, in_features, img_channels = 3):
        super().__init__()
        
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2, True)

        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(in_features * factors[i])
            conv_out = int(in_features * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in, conv_out, use_pixelnorm = True))
            self.rgb_layers.append(WSConv2D(img_channels, conv_in, kernel_size = 1, stride = 1, padding = 0))
        self.initial_rgb = WSConv2D(img_channels, in_features, kernel_size = 1, stride = 1, padding = 0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(2, 2)
        
        self.final_block = nn.Sequential(
            WSConv2D(in_features + 1, in_features, kernel_size = 2, padding = 1),
            nn.LeakyReLU(0.2),
            WSConv2D(in_features, in_features, kernel_size = 4, padding = 0, stride = 1),
            nn.LeakyReLU(0.2),
            WSConv2D(in_features, 1, kernel_size = 1, padding = 0, stride = 1)
        )
    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled
    
    def minibatch_std(self, x):
        batch_stats = (torch.std(x, dim = 0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3]))
        return torch.cat([x, batch_stats], dim = 1)
    
    def forward(self, x, alpha, steps):
        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](x))
        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)
        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)
        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)
    
        