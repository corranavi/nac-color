'''Script containing methods and class to build the Deep Colorization NN'''
import torch
import torch.nn as nn

def bn_weight_init(m):
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def _make_res_layers(nl, ni, kernel=3, stride=1, padding=1):
    layers = []
    for i in range(nl):
        layers.append(ResBlock(ni, kernel=kernel, stride=stride, padding=padding))

    return nn.Sequential(*layers)

def conv_layer(in_layer, out_layer, kernel=3, stride=1, padding=1, instanceNorm=False):
    """Building block for PixelShuffle convolutional layers"""
    return nn.Sequential(
        nn.Conv2d(in_layer, out_layer, kernel_size=kernel, stride=stride, padding=padding),
        nn.BatchNorm2d(out_layer) if not instanceNorm else nn.InstanceNorm2d(out_layer),
        nn.LeakyReLU(inplace=True)
    )

def icnr(x, scale=4, init=nn.init.kaiming_normal_):
    """ ICNR init of `x`, with `scale` and `init` function.

        Checkerboard artifact free sub-pixel convolution: https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)

class ResBlock(nn.Module):
    def __init__(self, ni, nf=None, kernel=3, stride=1, padding=1):
        super().__init__()
        if nf is None:
            nf = ni
        self.conv1 = conv_layer(ni, nf, kernel=kernel, stride=stride, padding=padding)
        self.conv2 = conv_layer(nf, nf, kernel=kernel, stride=stride, padding=padding)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))
    

class ColorizationModule(pl.LightningModule):
    
    def __init__(self, num_input_channels:int, name="unique_colorizer"):
        super().__init__()
        self.colorization_name = name
        self.model = PixelShuffle(num_input_channels, scale=2)

    def forward(self,x):
        return self.model(x)
    
    
class BaseDECO(pl.LightningModule):
    def __init__(self, out=224, init=None):
        super().__init__()
        self.out_s = out
        self.init = init
    
    def init_weights(self):
        if self.init == None:
            pass
        elif self.init == 1:
            self.apply(bn_weight_init)


class PixelShuffle(BaseDECO):
    """
        Custom implementation of PixelShuffle 
    """

    def __init__(self, num_input_channels: int, out:int=224, init:int=1, scale:int=4, lrelu:bool=False):
        super().__init__(out, init)
        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks = _make_res_layers(8,64)
        self.pixel_shuffle = PixelShuffle_ICNR(ni=64, nf=3, scale=scale, lrelu=lrelu)
        self.init_weights()

    def forward(self, xb):
        """
        @:param xb : Tensor "x batch"
          Batch of input images

        @:return tensor
          A batch of output images
        """
        _xb = self.maxpool(self.act1(self.bn1(self.conv1(xb))))
        _xb = self.resblocks(_xb)

        return self.pixel_shuffle(_xb)


class PixelShuffle_ICNR(pl.LightningModule):
    """ Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init,
        and `weight_norm`.

        "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts":
        https://arxiv.org/abs/1806.02658
    """

    def __init__(self, ni: int, nf: int = None, scale: int = 4, icnr_init=True, blur_k=2, blur_s=1,
                 blur_pad=(1, 0, 1, 0), lrelu=True):
        super().__init__()
        nf = ni if nf is None else nf
        self.conv = conv_layer(ni, nf * (scale ** 2), kernel=1, padding=0, stride=1) if lrelu else nn.Sequential(
            nn.Conv2d(64, 3 * (scale ** 2), 1, 1, 0), nn.BatchNorm2d(3 * (scale ** 2)))
        if icnr_init:
            icnr(self.conv[0].weight, scale=scale)
        self.act = nn.LeakyReLU(inplace=False) if lrelu else nn.Hardtanh(-10000, 10000)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        self.pad = nn.ReplicationPad2d(blur_pad)
        self.blur = nn.AvgPool2d(blur_k, stride=blur_s)

    def forward(self, x):
        x = self.shuf(self.act(self.conv(x)))
        return self.blur(self.pad(x))