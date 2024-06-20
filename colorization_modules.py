
class BaseDECO(nn.Module):
    def __init__(self, out=224, init=None):
        """
        :param out: int
            size of output of the DECO module
        """
        super().__init__()
        self.out_s = out
        self.init = init

    def set_output_size(self, out_s):
        self.out_s = out_s

    def init_weights(self):
        if self.init is None:
            pass
        elif self.init == 0:
            self.apply(default_deco__weight_init)
        elif self.init == 1:
            self.apply(bn_weight_init)

class StandardDECO(BaseDECO):
    """
    Standard DECO Module
    """

    def __init__(self, out=224, init=0, deconv=False, norm=False):
        super().__init__(out, init)
        self.norm = norm
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        # ReLU
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks = _make_res_layers(8, 64)
        self.conv_last = nn.Conv2d(64, 3, kernel_size=1)
        self.deconv = deconv
        if deconv:
            # TODO: Check if use "groups = 1"
            self.deconv = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=8, padding=2, stride=4,
                                             groups=3, bias=False)
        else:
            self.pad = nn.ReflectionPad2d(1)
            self.conv_up = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=0, stride=1)

        self.init_weights()

    def forward(self, xb):
        """
        @:param xb : Tensor
          Batch of input images

        @:return tensor
          A batch of output images
        """
        _xb = self.maxpool(F.leaky_relu(self.bn1(self.conv1(xb))))
        _xb = self.resblocks(_xb)
        _xb = self.conv_last(_xb)
        if self.deconv:
            _xb = self.deconv(_xb, output_size=xb.shape)
        else:
            _xb = self.conv_up(self.pad(F.interpolate(_xb, scale_factor=4, mode='nearest')))
        return normalize_channels(_xb) if self.norm else _xb
    
class SubPixelConvDECO(BaseDECO):
    """
    ESPCN DECO Module
    https://github.com/leftthomas/ESPCN/blob/master/model.py
    """

    def __init__(self, out=224, init=0):
        super().__init__(out, init)
        # Which value should I use for stride and padding?
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        # ReLU
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks = _make_res_layers(8, 64)
        upscale_factor = 4
        channel = 3
        self.conv_last = nn.Conv2d(64, channel * (upscale_factor ** 2), kernel_size=3, padding=1, stride=1)
        icnr(self.conv_last.weight, scale=upscale_factor)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, xb):
        """
        @:param xb : Tensor
          Batch of input images

        @:return tensor
          A batch of output images
        """
        _xb = self.maxpool(F.leaky_relu(self.bn1(self.conv1(xb))))
        _xb = F.leaky_relu(self.resblocks(_xb))
        return self.pixel_shuffle(self.conv_last(_xb))
    
class PixelShuffle_ICNR(nn.Module):
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
    
class PixelShuffle_ICNR_DECO(BaseDECO):
    """
    PixelShuffle DECO Module
    """

    def __init__(self, out=224, init=1, scale=4, lrelu=True):
        super().__init__(out, init)
        # Which value should I use for stride and padding?
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks = _make_res_layers(8, 64)
        self.pixel_shuffle = PixelShuffle_ICNR(ni=64, nf=3, scale=scale, lrelu=lrelu)
        self.init_weights()

    def forward(self, xb):
        """
        @:param xb : Tensor
          Batch of input images

        @:return tensor
          A batch of output images
        """
        _xb = self.maxpool(self.act1(self.bn1(self.conv1(xb))))
        _xb = self.resblocks(_xb)

        return self.pixel_shuffle(_xb)
    
class PixelShuffle_ICNR_DECO_TEST(BaseDECO):
    """
    PixelShuffle DECO Module
    """

    def __init__(self, out=224, init=0, scale=8):
        super().__init__(out, init)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2)
        self.in1 = nn.InstanceNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, padding=1)
        self.conv = conv_layer(64, 3 * (scale ** 2), kernel=1, padding=0, stride=1)
        icnr(self.conv[0].weight, scale=scale)
        self.in2 = nn.InstanceNorm2d(3 * (scale ** 2))
        self.shuf = nn.PixelShuffle(scale)
        self.pad = nn.ReplicationPad2d(1)
        self.avg_pool = nn.AvgPool2d(4, 2)

    def forward(self, xb):
        """
        @:param xb : Tensor
          Batch of input images

        @:return tensor
          A batch of output images
        """
        _xb = self.maxpool(F.leaky_relu(self.in1(self.conv1(xb))))
        _xb = self.in2(self.conv(_xb))
        _xb = self.avg_pool(self.pad(self.shuf(_xb)))
        return normalize_channels(_xb, every=False)
    
   