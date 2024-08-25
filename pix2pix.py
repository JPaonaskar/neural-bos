'''
PIX2PIX

hardcoded pix2pix implementation following the paper and associated resources

Referances:
    https://arxiv.org/pdf/1611.07004
'''
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import utils

LAMBDA = 100

# weight init function
def weight_init(module:nn.Module) -> None:
    '''
    Initialize weights in a module with mean = 0.0 and stdev = 0.02

    Args:
        module (nn.Module) : module to change weight for
    '''
    classname = module.__class__.__name__

    # check convolutional layer
    if 'Conv' in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)

    # check batchnorm
    elif 'BatchNorm' in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)

class DownSample(nn.Module):
    '''
    Down sampling block

    Args:
        in_channels (int) : input channels
        out_channels (int) : output channels
        size (int) : kernal size (default=4)
        batchnorm (bool) : apply batchnorm (default=True)
    '''
    def __init__(self, in_channels:int, out_channels:int, size:int=4, batchnorm:bool=True):
        super().__init__()

        # create sequence
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, size, 2, 1, bias=False, padding_mode='reflect')
        )

        # add batch norm
        if batchnorm:
            self.conv.add_module('BatchNorm2d', nn.BatchNorm2d(out_channels))

        # activation function
        self.conv.add_module('LeakyReLU', nn.LeakyReLU(0.2))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        Feed forward

        Args:
            x (torch.Tensor) : input

        Returns:
            (torch.Tensor) : output
        '''
        return self.conv(x)

class UpSample(nn.Module):
    '''
    Up sampling block

    Args:
        in_channels (int) : input channels
        out_channels (int) : output channels
        size (int) : kernal size (default=4)
        dropout (bool) : add dropout layer (default=False)
    '''
    def __init__(self, in_channels:int, out_channels:int, size:int=4, dropout:bool=False):
        super().__init__()

        # create sequence
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, size, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # add dropout
        if dropout:
            self.conv.add_module('Dropout', nn.Dropout(0.5))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        Feed forward

        Args:
            x (torch.Tensor) : input

        Returns:
            (torch.Tensor) : output
        '''
        return self.conv(x)

class Generator(nn.Module):
    '''
    Pix2pix gererator module

    Args:
        in_channels (int) : input image channels (default=3)
        out_channels (int) : output image channels (default=3)
    '''
    def __init__(self, in_channels:int=3, out_channels:int=3):
        super().__init__()

        # architecture
        self.encoder = nn.ModuleList([
            DownSample(in_channels, 64, batchnorm=False),
            DownSample(64, 128),
            DownSample(128, 256),
            DownSample(256, 512),
            DownSample(512, 512),
            DownSample(512, 512),
            DownSample(512, 512),
        ])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder = nn.ModuleList([
            UpSample(512, 512, dropout=True),
            UpSample(1024, 512, dropout=True),
            UpSample(1024, 512, dropout=True),
            UpSample(1024, 512),
            UpSample(1024, 256),
            UpSample(512, 128),
            UpSample(256, 64),
        ])

        self.output = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

        # loss functions
        self.BCE_loss = nn.BCEWithLogitsLoss()
        self.L1_loss = nn.L1Loss()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        Pass forward through network

        Args:
            x (torch.Tensor) : input image

        Returns:
            (torch.Tensor) : output image
        '''
        # encode
        skips = []
        for module in self.encoder:
            x = module(x)
            skips.append(x)

        # bottleneck
        x = self.bottleneck(x)

        # orgainize skips
        skips = reversed(skips)

        # decode
        for module, skip in zip(self.decoder, skips):
            x = module(x)
            x = torch.cat([x, skip], dim=1)

        # output
        return self.output(x)
    
    def predict(self, x:torch.Tensor) -> torch.Tensor:
        '''
        Predict target image

        Args:
            x (torch.Tensor) : input image

        Returns:
            pred (torch.Tensor) : generated prediction
        '''
        # no gradients
        with torch.no_grad():
            # predict
            pred = self.forward(x)

        # output
        return pred
    
    def loss(self, dis_generated_output:torch.Tensor, gen_output:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        '''
        Compute the gerator loss

        Args:
            disc_generated_output (torch.Tensor) : discriminator output for gernated image
            gen_out (torch.Tensor) : generator output
            target (torch.Tensor) : target image

        Returns:
            loss (torch.Tensor) : generator loss
        '''
        gan_loss = self.BCE_loss(dis_generated_output, torch.ones_like(dis_generated_output))
        loss = gan_loss + (LAMBDA * self.L1_loss(gen_output, target))

        return loss
    
class Discriminator(nn.Module):
    '''
    Pix2pix discriminator module

    Args:
        in_channels (int) : generator input image channels (default=3)
        out_channels (int) : generator output image channels (default=3)
    '''
    def __init__(self, in_channels:int=3, out_channels:int=3):
        super().__init__()

        # architecture
        self.sequence = nn.Sequential(
            DownSample(in_channels + out_channels, 64, batchnorm=False),
            DownSample(64, 128, batchnorm=False),
            DownSample(128, 256, batchnorm=False),
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 512, 4, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 1, 4, 1)
        )

        # loss functions
        self.BCE_loss = nn.BCEWithLogitsLoss()

    def forward(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        '''
        Pass forward through network

        Args:
            x (torch.Tensor) : input image

        Returns:
            (torch.Tensor) : output image
        '''
        return self.sequence(torch.cat([x, y], dim=1))
    
    def loss(self, dis_real_output:torch.Tensor, dis_generated_output:torch.Tensor) -> torch.Tensor:
        '''
        Compute the gerator loss

        Args:
            disc_real_output (torch.Tensor) : discriminator output for real image
            disc_generated_output (torch.Tensor) : discriminator output for gernated image

        Returns:
            loss (torch.Tensor) : generator loss
        '''
        real_loss = self.BCE_loss(dis_real_output, torch.ones_like(dis_real_output))
        generated_loss = self.BCE_loss(dis_generated_output, torch.zeros_like(dis_generated_output))

        loss = real_loss + generated_loss

        return loss

class Pix2PixModel(nn.Module):
    '''
    Pix2Pix implementation

    Args:
        device (torce.device) : device to use (default=torch.device('cuda'))
        in_channels (int) : generator input image channels (default=3)
        out_channels (int) : generator output image channels (default=3)
        learning_rate (float) : model learing rate (default=2e-4)
        beta_1 (float) : optimizer beta 1 (default=0.5)
    '''
    def __init__(self, device:torch.device=torch.device('cuda'), in_channels:int=3, out_channels:int=3, learning_rate:float=2e-4, beta_1:float=0.5):
        super().__init__()

        # store device
        self.device = device

        # create models
        self.gen = Generator(in_channels, out_channels).to(device)
        self.dis = Discriminator(in_channels, out_channels).to(device)

        # assign weight distributions
        self.gen.apply(weight_init)
        self.dis.apply(weight_init)

        # define optimizers
        self.opt_gen = torch.optim.Adam(self.gen.parameters(), lr=learning_rate, betas=(beta_1, 0.999))
        self.opt_dis = torch.optim.Adam(self.dis.parameters(), lr=learning_rate, betas=(beta_1, 0.999))

        # store learning reate
        self.learning_rate = learning_rate

        # loss history
        self.history = {
            'Discriminator' : [],
            'Generator' : []
        }

    def step(self, x:torch.Tensor, y:torch.Tensor) -> None:
        '''
        Training step

        Args:
            x (torch.Tensor) : input images batch
            y (torch.Tensor) : target images batch

        Returns:
            None
        '''
        # convert to same device space
        x = x.to(self.device)
        y = y.to(self.device)

        # generate image
        gen_output = self.gen(x)

        # analyze images
        dis_real_output = self.dis(x, y)
        dis_generated_output = self.dis(x, gen_output.detach())

        # compute discriminator loss
        dis_loss = self.dis.loss(dis_real_output, dis_generated_output)

        # train discriminator
        self.dis.zero_grad()
        dis_loss.backward()
        self.opt_dis.step()

        # analyze generated image
        dis_generated_output = self.dis(x, gen_output)
        
        # compute generator loss
        gen_loss = self.gen.loss(dis_generated_output, gen_output, y)

        # train generator
        self.opt_gen.zero_grad()
        gen_loss.backward()
        self.opt_gen.step()

        # store history
        self.history['Discriminator'].append(dis_loss.cpu().detach().numpy())
        self.history['Generator'].append(gen_loss.cpu().detach().numpy())

    def learn(self, dataset:Dataset, epochs:int=800, batch_size:int=16, checkpoints:int=10, last_checkpoint:str=None) -> None:
        '''
        Train Image-to-Image model

        Args:
            dataset (Dataset) : torch dataset to train on
            epochs (int) : number of epochs (default=800)
            batch_size (int) : size of batch of data (default=16)
            checkpoints (int) : number of epochs per checkpoint
            last_checkpoint (str) : path to last checkpoint

        Returns:
            None
        '''
        # set to training
        #self.train()
        
        # create dataloader
        loader = DataLoader(dataset, batch_size, shuffle=True)

        # load checkpoint
        epoch0 = 0
        if last_checkpoint:
            epoch0 = self.load_checkpoint(last_checkpoint)
            print(f'Loaded checkpoint {os.path.basename(last_checkpoint)}')

        # loop through epochs
        print('Training')
        for epoch in tqdm(range(epoch0, epochs)):
            # loop through batches
            for x, y in tqdm(loader, desc=f'Epoch {epoch+1}', leave=False):
                # train
                self.step(x, y)

            # checkpoint
            if (epoch + 1) % checkpoints == 0:
                name = f'epoch_{epoch+1}'
                self.save_checkpoint(epoch=epoch+1, name=name)

                # save sample
                x, y, pred = self.predict(dataset)

                utils.save_sample(x.cpu(), y, pred.cpu(), name=name)


    def predict(self, dataset:Dataset, batch_size:int=16, shuffle:bool=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Predict a batch

        Args:
            dataset (Dataset) : torch dataset to train on
            batch_size (int) : size of batch of data (default=16)
            shuffle (bool) : shuffle dataset (default=False)

        Returns:
            x (torch.Tensor) : input images
            y (torch.Tensor) : target images
            pred (torch.Tensor) : prediction
        '''
        # set to eval
        #self.eval()

        # create dataloader
        loader = DataLoader(dataset, batch_size, shuffle=shuffle)

        # pull
        x, y = next(iter(loader))

        # predict
        pred = self.gen.predict(x.to(self.device))

        # return prediction
        return x.cpu().detach(), y, pred.cpu().detach()
    
    def save_checkpoint(self, epoch:int=0, dirname:str='checkpoints', name:str='checkpoint') -> None:
        '''
        Save a checkpoint

        Args:
            epoch (int) : current epoch
            dirname (str) : directory to save to
            name (str) : checkpoint save name

        Returns:
            None
        '''
        # make path
        path = os.path.abspath(dirname)
        path = os.path.join(path, f'{name}.pt')

        # create checkpoint
        checkpoint = {
            'epoch' : epoch,
            'history' : self.history,
            'gen_state_dict': self.gen.state_dict(),
            'dis_state_dict': self.dis.state_dict(),
            'opt_gen_state_dict': self.opt_gen.state_dict(),
            'opt_dis_state_dict': self.opt_dis.state_dict(),
        }

        # save
        torch.save(checkpoint, path)

    def load_checkpoint(self, filename:str='checkpoints') -> int:
        '''
        Load a checkpoint

        Args:
            filename (str) : checkpoint filename / path

        Returns:
            epoch (int) : last epoch
        '''
        # make path
        path = os.path.abspath(filename)

        # load checkpoint
        checkpoint = torch.load(path, map_location=self.device)

        # assign state
        self.gen.load_state_dict(checkpoint["gen_state_dict"])
        self.dis.load_state_dict(checkpoint["dis_state_dict"])

        self.opt_gen.load_state_dict(checkpoint["opt_gen_state_dict"])
        self.opt_dis.load_state_dict(checkpoint["opt_dis_state_dict"])

        # set learning rates
        for param_group in self.opt_gen.param_groups:
            param_group["lr"] = self.learning_rate

        for param_group in self.opt_dis.param_groups:
            param_group["lr"] = self.learning_rate

        # get history
        self.history = checkpoint['history']

        # return epoch
        return checkpoint['epoch']
    
if __name__ == '__main__':
    # create sample input
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    
    # test discriminator
    dis = Discriminator()
    pred = dis(x, y)
    print(pred.shape)

    # test generator
    gen = Generator()
    preds = gen(x)
    print(preds.shape)