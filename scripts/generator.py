import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
            
            
class Generator(nn.Module):
    def __init__(self, z_dim, in_features, img_channels = 3):
        super(Generator, self).__init__()
        
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_features, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2D(in_features, in_features, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        self.initial_rgb = WSConv2D(in_features, img_channels, kernel_size = 1, stride = 1, padding = 0)
        self.prog_blocks, self.rgb_layers = (nn.ModuleList([]), nn.ModuleList([self.initial_rgb]))
        
        for i in range(len(factors) - 1):
            conv_in_c = int(in_features * factors[i])
            conv_out_c = int(in_features * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(WSConv2D(conv_out_c, img_channels, kernel_size = 1, stride = 1, padding = 0))
          
          
    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)
        
    def forward(self, x, alpha, steps):
        out = self.initial(x)
        if steps == 0:
            return self.initial_rgb(out)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor = 2, mode = "nearest")
            out = self.prog_blocks[step](upscaled)
        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)