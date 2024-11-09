# Import all the packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from torch.nn import init
# # Build one of the main components - DoubleConv - for UNet
# class DoubleConv(nn.Module):
#   def __init__(self, in_channels, out_channels):
#     super().__init__()
#     self.conv = nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True)
#     )
#   def forward(self, x):
#     return self.conv(x)

# # Build UNet from scrach
# class UNet(nn.Module):
#   def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
#     super().__init__()
#     self.downs = nn.ModuleList()
#     self.ups = nn.ModuleList()

#     # Downsampling layers
#     for feature in features:
#       self.downs.append(DoubleConv(in_channels, feature))
#       in_channels = feature

#     # Bottleneck layer
#     self.bottleneck = DoubleConv(features[-1], features[-1]*2)

#     # Upsampling layers
#     for feature in reversed(features):
#       self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
#       self.ups.append(DoubleConv(feature*2, feature))

#     # Output layer
#     self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

#   def forward(self, x):
#     skip_connections = []

#     # Downsampling path
#     for down in self.downs:
#       x = down(x)
#       skip_connections.append(x)
#       x = F.max_pool2d(x,(2, 2))

#     # Bottleneck
#     x = self.bottleneck(x)
#     skip_connections.reverse()

#     # Upsampling path
#     for i in range(0, len(self.ups), 2):
#       x = self.ups[i](x)
#       skip_connection = skip_connections[i//2]
#       concat = torch.cat((skip_connection, x), dim=1)
#       x = self.ups[i+1](concat)
#     return  self.final_conv(x)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, multi_stage=False):
        super(UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.activation = nn.Sequential(nn.Sigmoid())
        # init_weights(self)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        init_type = "normal"
        gain = 0.02
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.activation(d1)
        return d1
    
# Build CustomDataset for loading data
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform= transform
        self.images = os.listdir(image_dir)

        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '_mask.jpg'))
        
        image = np.array(Image.open(img_path).convert('RGB'))
        image = self.transform(image)
        # image = image * 2 - 1
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = self.transform(mask)
        
        return image, mask, self.images[idx]
