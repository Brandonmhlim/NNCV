import torch
import torch.nn as nn


class Model(nn.Module):
    """ 
    A simple U-Net architecture for image segmentation.
    Based on the U-Net architecture from the original paper:
    Olaf Ronneberger et al. (2015), "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf

    Adapt this model as needed for your problem-specific requirements. You can make multiple model classes and compare them,
    however, the CodaLab server requires the model class to be named "Model". Also, it will use the default values of the constructor
    to create the model, so make sure to set the default values of the constructor to the ones you want to use for your submission.
    """
    def __init__(
        self, 
        in_channels=3, 
        n_classes=19
    ):
        """
        Args:
            in_channels (int): Number of input channels. Default is 3 for RGB images.
            n_classes (int): Number of output classes. Default is 19 for the Cityscapes dataset.
        """
        
        super().__init__()

        # Encoding path
        self.in_channels = in_channels
        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (ResidualDown(64, 128))
        self.down2 = (ResidualDown(128, 256))
        self.down3 = (ResidualDown(256, 512))
        self.down4 = (ResidualDown(512, 1024))

        # Decoding path, ResidualUp
        #self.up1 = (ResidualUp(1024, 256))
        #self.up2 = (ResidualUp(512, 128))
        #self.up3 = (ResidualUp(256, 64))
        #self.up4 = (ResidualUp(128, 64))
        #self.outc = (OutConv(64, n_classes))
        
        # Decoding path, ResidualUpAttention
        self.up1 = (ResidualUpAttention(1024, 512, 512))
        self.up2 = (ResidualUpAttention(512, 256, 256))
        self.up3 = (ResidualUpAttention(256, 128, 128))
        self.up4 = (ResidualUpAttention(128, 64, 64))
        self.outc = (OutConv(64, n_classes))    
        

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        """
        # Check if the input tensor has the expected number of channels
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")
        
        # Encoding path
        x1 = self.inc(x)
        x2 = self.down1(x1) 
        x3 = self.down2(x2) 
        x4 = self.down3(x3) 
        x5 = self.down4(x4)  

        # Decoding path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits
        

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class ResidualDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(out_channels),
        )
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) # this is the connection that will be added later 
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool(x)
        shortcut = self.shortcut(x) # after the result of the pooled input
        x = self.double_conv(x) # do the double conv
        x = x + shortcut # add the result after the two conv 
        x = self.relu(x)
        return x 
        
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class ResidualUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) # this is the connection that will be added later
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x1, x2):
        x1 = self.up(x1) # decoder
        x = torch.cat([x2, x1], dim=1) # add in the encoder skip connection 
        shortcut = self.shortcut(x) # after the result of the concatenated input
        x = self.double_conv(x) # do the double conv 
        x = x + shortcut # add the result after the two conv 
        x = self.relu(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class AttentionGate(nn.Module):
    def __init__(self, enc_channels, dec_channels, inter_channels):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(dec_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(enc_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, enc_map, dec_map):
        Wg = self.Wg(dec_map)
        Ws = self.Ws(enc_map)
        out = self.relu(Wg + Ws)
        out = self.conv(out)
        out = self.sigmoid(out)
        out = out * enc_map
        return out

class ResidualUpAttention(nn.Module):
    def __init__(self, decoder_channels, encoder_channels, out_channels):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        inter_channels = encoder_channels // 2
        
        self.attention_gate = AttentionGate(encoder_channels, decoder_channels, inter_channels)
        
        in_channels = decoder_channels + encoder_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) # this is the connection that will be added later
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x1, x2 ):
        x1 = self.up(x1) # decoder
        x2 = self.attention_gate(x2, x1) # apply attention gate to
        x = torch.cat([x2, x1], dim=1) # add in the encoder skip connection
        shortcut = self.shortcut(x) # after the result of the concatenated input
        x = self.double_conv(x) # do the double conv
        x = x + shortcut # add the result after the two conv
        x = self.relu(x)

        return x
