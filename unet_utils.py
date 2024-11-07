# Import all the packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

# Build one of the main components - DoubleConv - for UNet
class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
  def forward(self, x):
    return self.conv(x)

# Build UNet from scrach
class UNet(nn.Module):
  def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
    super().__init__()
    self.downs = nn.ModuleList()
    self.ups = nn.ModuleList()

    # Downsampling layers
    for feature in features:
      self.downs.append(DoubleConv(in_channels, feature))
      in_channels = feature

    # Bottleneck layer
    self.bottleneck = DoubleConv(features[-1], features[-1]*2)

    # Upsampling layers
    for feature in reversed(features):
      self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
      self.ups.append(DoubleConv(feature*2, feature))

    # Output layer
    self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

  def forward(self, x):
    skip_connections = []

    # Downsampling path
    for down in self.downs:
      x = down(x)
      skip_connections.append(x)
      x = F.max_pool2d(x,(2, 2))

    # Bottleneck
    x = self.bottleneck(x)
    skip_connections.reverse()

    # Upsampling path
    for i in range(0, len(self.ups), 2):
      x = self.ups[i](x)
      skip_connection = skip_connections[i//2]
      concat = torch.cat((skip_connection, x), dim=1)
      x = self.ups[i+1](concat)
    return  self.final_conv(x)


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
        image = image * 2 - 1
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = self.transform(mask)
        
        return image, mask, self.images[idx]
