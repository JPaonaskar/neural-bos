'''
PIX2PIX

pix2pix Image-to-Image Translation GAN

Referances:
    https://medium.com/@Skpd/pix2pix-gan-for-generating-map-given-satellite-images-using-pytorch-6e50c318673a
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import utils
    
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
    
class Discriminator(nn.Module):
    def __init__(self, in_channels:int=3, features:list[int]=[64, 128, 256, 512]):
        super().__init__()
        # add initial layer
        layers = [
            nn.Sequential(
                nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(0.2)
            )
        ]

        in_channels = features[0]
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

            # store in channels
            in_channels = feature

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
    
class Image2Image():
    '''
    Image-to-Image class

    Args:
        learning_rate (float) : optimizer learning rate (default=2e-4)
        beta1 (float) : optimizer beta1 (default=0.5)
    
    '''
    def __init__(self, learning_rate:float=2e-4, beta1:float=0.5):
        # create models
        self.net_dis = Discriminator(in_channels=3).cuda()
        self.net_gen = Generator(in_channels=3).cuda()

        # store hyperparameters
        self.learning_rate = learning_rate
        self.beta1 = beta1

        # setup optimizer
        self.opt_dis = torch.optim.Adam(self.net_dis.parameters(), lr=learning_rate, betas=(beta1, 0.999))
        self.opt_gen = torch.optim.Adam(self.net_gen.parameters(), lr=learning_rate, betas=(beta1, 0.999))

        # setup loss
        self.BCE_Loss = nn.BCEWithLogitsLoss()
        self.L1_Loss = nn.L1Loss()

        # lost history
        self.history = {
            'dis' : [],
            'gen' : []
        }

    def epoch(self, loader:DataLoader, l1_lambda:float=100.0) -> None:
        '''
        Train for an epoch

        Args:
            loader (Dataloader) : dataloader with data
            l1_lambda (float) : L1 loss scale (default=100.0)

        Returns:
            None
        '''
        # loop
        for x, y in tqdm(loader):
            # GPU acceleration
            x = x.cuda()
            y = y.cuda()

            # generate image
            y_fake = self.net_gen(x)

            # real image
            dis_real = self.net_dis(x, y)
            dis_real_loss = self.BCE_Loss(dis_real, torch.ones_like(dis_real))

            # fake image
            dis_fake = self.net_dis(x, y_fake.detach())
            dis_fake_loss = self.BCE_Loss(dis_fake, torch.zeros_like(dis_fake))

            # average loss
            dis_loss = (dis_real_loss + dis_fake_loss) / 2

            # store loss
            self.history['dis'].append(dis_loss.cpu().detach())

            # train discriminator
            self.net_dis.zero_grad()
            dis_loss.backward()
            self.opt_dis.step()
            
            # compute loss
            dis_fake = self.net_dis(x, y_fake)
            gen_fake_loss = self.BCE_Loss(dis_fake, torch.ones_like(dis_fake))

            gen_loss = gen_fake_loss + self.L1_Loss(y_fake,y) * l1_lambda
            
            # store loss
            self.history['gen'].append(gen_loss.cpu().detach())

            # train generator
            self.opt_gen.zero_grad()
            gen_loss.backward()
            self.opt_gen.step()

    def train(self, dataset:Dataset, epochs:int=800, batch_size:int=16, load:int=0, checkpoint:bool=True) -> None:
        '''
        Train Image-to-Image model

        Args:
            dataset (Dataset) : torch dataset to train on
            epochs (int) : number of epochs (default=800)
            batch_size (int) : size of batch of data (default=16)
            load (int) : load checkpoint from specified epoch
            checkpoint (bool) : save checkpoints

        Returns:
            None
        '''
        # create dataloader
        loader = DataLoader(dataset, batch_size, shuffle=True)

        # load model
        if load:
            utils.load_checkpoint(self.net_dis, self.opt_dis, self.learning_rate, 'cuda:0', f'checkpoints\\checkpoint_dis-{load}.pt')
            utils.load_checkpoint(self.net_dis, self.opt_dis, self.learning_rate, 'cuda:0', f'checkpoints\\checkpoint_gen-{load}.pt')

        # train at each epoch
        for epoch in range(epochs):
            # train for an epoch
            self.epoch(loader)

            # save checkpoint
            if checkpoint:
                utils.save_checkpoint(self.net_dis, self.opt_dis, f'checkpoints\\checkpoint_dis-{epoch+1}.pt')
                utils.save_checkpoint(self.net_dis, self.opt_dis, f'checkpoints\\checkpoint_gen-{epoch+1}.pt')

    def plot_history(self) -> None:
        '''
        Plot loss history

        Args:
            None

        Returns:
            None
        '''
        utils.plot_loss(self.history)

    def plot_predictions(self, dataset:Dataset) -> None:
        '''
        Plot some predictions

        Args:
            dataset (Dataset) : dataset to predict with

        Returns:
            None
        '''
        loader = DataLoader(dataset, batch_size=64)

        x, y = next(iter(loader))

        with torch.no_grad():
            pred = self.net_gen(x.cuda()).cpu().detach()

        # plot
        utils.plot_images(x * 0.5 + 0.5, 'Input Images', show=False)
        utils.plot_images(y * 0.5 + 0.5, 'Target Images', show=False)
        utils.plot_images(pred * 0.5 + 0.5, 'Pred Images')

if __name__ == '__main__':
    def test_disc():
        x = torch.randn((1, 3, 256, 256))
        y = torch.randn((1, 3, 256, 256))
        Model = Discriminator()
        print(Model.model)
        pred = Model(x, y)
        print(pred.shape)
    test_disc()

    def test_gen():
        x = torch.randn((1, 3, 256, 256))
        model = Generator(in_channels=3, features=64)
        preds = model(x)
        print(preds.shape)
    test_gen()