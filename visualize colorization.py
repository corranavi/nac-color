"""
Questo file serve per scopi di visualizzazione delle trasformazioni applicate
"""
import torch
import torchvision
from torchvision.transforms import v2 as T
from dataset_lib import MRIDataset
from utils.train_utils_old import retrieve_folders_list, Kfold_split
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from PIL import Image

import lightning.pytorch as pl
import torch.nn as nn

import logging
logging.basicConfig(level=logging.INFO)

torch.manual_seed(42)
np.random.seed(42)
# -------------------------------------------------------colorization
def conv_layer(in_layer, out_layer, kernel=3, stride=1, padding=1, instanceNorm=False):
    """ Per costruire i conv layer di pixel shuffle """
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

def _make_res_layers(nl, ni, kernel=3, stride=1, padding=1):
    layers = []
    for i in range(nl):
        layers.append(ResBlock(ni, kernel=kernel, stride=stride, padding=padding))

    return nn.Sequential(*layers)

def bn_weight_init(m):
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class ResBlock(nn.Module):
    def __init__(self, ni, nf=None, kernel=3, stride=1, padding=1):
        super().__init__()
        if nf is None:
            nf = ni
        self.conv1 = conv_layer(ni, nf, kernel=kernel, stride=stride, padding=padding)
        self.conv2 = conv_layer(nf, nf, kernel=kernel, stride=stride, padding=padding)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))
    
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

class PixelShuffle(BaseDECO):
    """
        Modello PixelShuffle, che è quello che lavora sulla risoluzione dei checkboard artifacts
    """

    def __init__(self, out=224, init=1, scale=4, lrelu=False):
        super().__init__(out, init)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2)
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


def get_transformations(choice: int = 1) -> "torchvision.transforms":
    if choice == 1:
        transforms = T.Compose([
            T.RandomResizedCrop(size=(224, 224), scale=(1,1), antialias=True),
            T.RandomHorizontalFlip( p = 0.5),
            T.ColorJitter( brightness = (0.5 , 2)), #brightness troppo alta considerando la compressione bit -> provare a riddure max value a 2
            T.RandomAffine(
                degrees = 0,
                scale =(0.7, 1.3),
                shear = 0.3
            ),
            #T.ToTensor()
        ])
    else:
        transforms = None
    return transforms              #per visualizzare post norm devi riscalare tra min max e poi x 255

def save_images(images, output_name, save=True):
    grid = torchvision.utils.make_grid(images.view(-1,3,224,224))
    ndarr = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(10,10))
    plt.imshow(ndarr)
    plt.axis('off')  # Turn off axis
    if save:
        plt.savefig(output_name, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    index_to_print = [9] #,22,27] #[9,22,148]
    time_run = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    slices = 3
    full_path_to_dataset = "C:\\Users\\c.navilli\\Desktop\\Prova\\dataset_mini"
    preprocess_type="percentile"                        #choices for preprocess_type = min_max, norm, 10bit, 12bit, 16bit, norm_and_scale. Defaults to norm without scale
    
    if preprocess_type in ["16bit"]:
        multiply_255=True
    else:
        multiply_255=False

    folders_list = retrieve_folders_list(full_path_to_dataset)
    datasets_list = Kfold_split(folders_list, 1)
    
    dataset = MRIDataset(datasets_list[0][0], slices, preprocess_type=preprocess_type) 

    transformations = get_transformations() 

    for index in index_to_print:
        print(index)
        image_file_name = f'2605_2_image_scale_{preprocess_type}_{index}.png'
        images, _ = dataset.__getitem__(index)
        if multiply_255:
            images = images *255
            image_file_name = image_file_name.split(".")[0]+"_255"+".png"
        
        print(images.shape)
        for x in images:
            one_layer = x[0].unsqueeze(0)
            print(one_layer.shape)

        #TODO riattivare
        # images_for_visualization = images #/(2**16-1)*255 #---> perchè così saranno compresi tra 0 e 1, come si aspetta la visualizzazione per dati di tipo float
        # # if index==9:
        # #     print(images)
        # transformed_images = transformations(images)
        # print(f"Min pre transformation: {images.min()} - Max pre transformation: {images.max()}")
        # print(f"Min post transformation: {transformed_images.min()} - Max post transformation: {transformed_images.max()}")

        # comparison_images = torch.cat((images_for_visualization.unsqueeze(0), transformed_images.unsqueeze(0)))

        # save_images(comparison_images, image_file_name)

        model = PixelShuffle()
        model.eval()
        
        image_path = "C:\\Users\\c.navilli\\Desktop\\Prova\\particolare.png"  # Replace with the path to your image
        image = Image.open(image_path).convert('RGB')
        x = T.ToTensor()(image)
        print(x.shape)
        one_layer = x[0].unsqueeze(0)
        plt.figure(figsize=(10,10))
        plt.imshow(x.permute(1, 2, 0))
        plt.show()
        colorized = model(one_layer.unsqueeze(0))
        colorized = colorized.squeeze()
        print(colorized.shape)
        colorized = colorized.detach().permute(1, 2, 0).numpy()
        print(colorized)
        print(f"Details: {colorized.dtype}")
        plt.figure(figsize=(10,10))
        plt.imshow(colorized)
        plt.show()

        
