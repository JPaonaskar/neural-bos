'''
PIX2PIX

pix2pix Image-to-Image Translation GAN

Resources:
    https://medium.com/@Skpd/pix2pix-gan-for-generating-map-given-satellite-images-using-pytorch-6e50c318673a
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''
import torch
import torch.nn as nn
    
class Down_Block(nn.Module):
    '''
    Down sampling block for U-Net

    Args:
        in_channels (int) : input channels
        out_channels (int) : output channels
        negitive_slope (float) : activation function (default=0.0)
        dropout (float) : dropout layer value (default=0.0)
    '''
    def __init__(self, in_channels:int, out_channels:int, negitive_slope:float=0.0, dropout:float=0.0):
        super().__init__()

        # create sequence
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negitive_slope)
        )

        # add dropout
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        Feed forward

        Args:
            x (torch.Tensor) : input

        Returns:
            x (torch.Tensor) : output
        '''
        # apply convolution
        x = self.conv(x)

        # apply dropout
        if self.dropout:
            x = self.dropout(x)

        # output
        return x

class Up_Block(nn.Module):
    '''
    Down sampling block for U-Net

    Args:
        in_channels (int) : input channels
        out_channels (int) : output channels
        negitive_slope (float) : activation function (default=0.0)
        dropout (float) : dropout layer value (default=0.0)
    '''
    def __init__(self, in_channels:int, out_channels:int, negitive_slope:float=0.0, dropout:float=0.0):
        super().__init__()

        # create sequence
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negitive_slope)
        )

        # add dropout
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        Feed forward

        Args:
            x (torch.Tensor) : input

        Returns:
            x (torch.Tensor) : output
        '''
        # apply convolution
        x = self.conv(x)

        # apply dropout
        if self.dropout:
            x = self.dropout(x)

        # output
        return x

class Generator(nn.Module):
    '''
    Image-to-Image generator

    Args:
        in_channels (int) : input channels
        features (int) : 
    '''
    def __init__(self, in_channels:int=3, features:int=64):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        ) # 128 x 128

        # encoder
        self.down1 = Down_Block(features, features*2, negitive_slope=0.2)
        self.down2 = Down_Block(features*2, features*4, negitive_slope=0.2)
        self.down3 = Down_Block(features*4, features*8, negitive_slope=0.2)
        self.down4 = Down_Block(features*8, features*8, negitive_slope=0.2)
        self.down5 = Down_Block(features*8, features*8, negitive_slope=0.2)
        self.down6 = Down_Block(features*8, features*8, negitive_slope=0.2)

        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode='reflect')
        )

        # decoder
        self.up1 = Up_Block(features*8, features*8, dropout=0.5)
        self.up2 = Up_Block(features*16, features*8, dropout=0.5)
        self.up3 = Up_Block(features*16, features*8, dropout=0.5)
        self.up4 = Up_Block(features*16, features*8)
        self.up5 = Up_Block(features*16, features*4)
        self.up6 = Up_Block(features*8, features*2)
        self.up7 = Up_Block(features*4, features)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        Feed forward

        Args:
            x (torch.Tensor) : input

        Returns:
            out (torch.Tensor) : output
        '''
        # encode
        d1 = self.down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        
        # bottleneck
        bottleneck = self.bottleneck(d7)
        
        # skip and decode
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))

        out = self.up(torch.cat([up7, d1],1))
        
        # output
        return out
    
class Descriminator(nn.Module):
    def __init__(self, in_channels:int=3, features:list[int]=[64, 128, 256, 512]):
        super().__init__()
        # add initial layer
        layers = [
            nn.Sequential(
                nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(0.2)
            )
        ]

        for feature in features[1:]:
            # determine stride
            if feature == features[-1]:
                stride = 2
            else:
                stride=1

            # add CNN block
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, 4, stride, bias=False, padding_mode='reflect'),
                    nn.BatchNorm2d(feature),
                    nn.LeakyReLU(0.2)
                )
            )

        # add final layer
        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, bias=False, padding_mode='reflect')
        )

        # join
        self.model = nn.Sequential(*layers)

    
    def forward(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        '''
        Feed forward

        Args:
            x (torch.Tensor) : Real input image
            y (torch.Tensor) : Real/Fake output image

        Returns:
            out (torch.Tensor) : output
        '''
        # concatinate and feed
        out = torch.cat([x, y], dim=1)
        out = self.model(out)

        # output
        return out