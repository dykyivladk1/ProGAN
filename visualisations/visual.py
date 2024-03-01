
import torch
import torch.nn as nn
import torch.nn.functional as F


factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]


class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size = 3, stride = 1, padding = 1, gain = 2):
        super(WSConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5

        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)


    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim = 1, keepdim = True) + self.epsilon)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm = True):
        super(ConvBlock, self).__init__()

        self.use_pn = use_pixelnorm
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)

        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x
    
    
    
class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels = 3):
        super(Generator, self).__init__()

        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size = 1, stride = 1, padding = 0)
        self.prog_blocks, self.rgb_layers = (nn.ModuleList([]), nn.ModuleList([self.initial_rgb]))

        for i in range(len(factors) - 1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(WSConv2d(conv_out_c, img_channels, kernel_size = 1, stride = 1, padding = 0))


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
    
    
class Discriminator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels = 3):
        super(Discriminator, self).__init__()

        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors) - 1, 0, -1):

            conv_in = int(in_channels * factors[i])
            conv_out = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in, conv_out, use_pixelnorm = False))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in, kernel_size = 1, stride = 1, padding = 0))
        self.initial_rgb = WSConv2d(img_channels, in_channels, kernel_size = 1, stride = 1, padding = 0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(2, 2)

        self.final_block = nn.Sequential(
            WSConv2d(in_channels +1 , in_channels, kernel_size = 2, padding = 1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size = 4, padding = 0, stride = 1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size = 1, padding = 0, stride = 1)
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
    
    
class GeneratorWrapper(nn.Module):
    def __init__(self, generator, alpha, steps):
        super().__init__()
        self.generator = generator
        self.alpha = alpha
        self.steps = steps

    def forward(self, x):
        return self.generator(x, self.alpha, self.steps)

class DiscriminatorWrapper(nn.Module):
    def __init__(self, discriminator, alpha, steps):
        super().__init__()
        self.discriminator = discriminator
        self.alpha = alpha
        self.steps = steps

    def forward(self, x):
        return self.discriminator(x, self.alpha, self.steps)



wsconv_model = WSConv2d(in_channels=3, out_channels=64)
wsconv_model.eval()

x = torch.randn(1, 3, 224, 224)

torch.onnx.export(wsconv_model, x, "wsconv2d.onnx", export_params=True, do_constant_folding=True,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})



pixelnorm_model = PixelNorm()
pixelnorm_model.eval()

x = torch.randn(1, 3, 224, 224)

torch.onnx.export(pixelnorm_model, x, "pixelnorm.onnx", export_params=True, do_constant_folding=True,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})



convblock_model = ConvBlock(in_channels=3, out_channels=64)
convblock_model.eval()

x = torch.randn(1, 3, 224, 224)

torch.onnx.export(convblock_model, x, "convblock.onnx", export_params=True, do_constant_folding=True,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})


