import torch 
import torch.nn as nn


class WSConv2D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size = 3, stride = 1, padding = 1, gain = 2):
        '''
        Weight-Scaled Convolutional 2D Class Module,
        stabilizes the training of GAN by normalizing the weights, 
        which helps in controlling the scale of gradients
        '''
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        
        self.bias = self.conv.bias
        self.conv.bias = None
        
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class PixelNorm(nn.Module):
    '''
    This class module is designed to perform pixel-wise
    feature extraction normalization.
    '''
    def __init__(self):
        super().__init__()
        
        self.epsilon = 1e-8
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim = 1, keepdim = True) + self.epsilon)


class ConvBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 use_pixelnorm = True):
        '''
        Convolutional Block, that performs Weight Scaling and Pixel Normalization 
        on the input tensor
        '''
        super().__init__()
        
        self.use_pn = use_pixelnorm
        
        self.conv1 = WSConv2D(in_channels, out_channels)
        self.conv2 = WSConv2D(out_channels, out_channels)
        
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()
        
    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x
    

    
def gradient_penalty(critic, real, fake, alpha, train_step, device = torch.device("mps")):
    batch_size, c, h, w, = real.shape
    beta = torch.rand((batch_size, 1, 1, 1)).repeat(1, c, h, w).to(device)
    
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)
    mixed_scores = critic(interpolated_images, alpha, train_step)
    
    gradient = torch.autograd.grad(
        inputs = interpolated_images,
        outputs = mixed_scores,
        grad_outputs = torch.ones_like(mixed_scores),
        create_graph = True,
        retain_graph = True
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim = 1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty
    