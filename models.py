'''
MODELS

Model reading and training

Referances:

'''
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import utils

CONFIG_ASSIGNMENT = '='
CONFIG_COMMENT = '#'
CONFIG_BLOCK = '['

CONFIG_NET = 'net'
CONFIG_CONVOLUTIONAL = 'convolutional'

BLOCK_TYPE = 'type'

BLOCK_NET_CHANNELS = 'channels'

BLOCK_TRANSPOSE = 'transpose'
BLOCK_FROM = 'from'

BLOCK_BATCH_NORM = 'batch_normalize'
BLOCK_FEATURE = 'feature'

BLOCK_SIZE = 'size'
BLOCK_STRIDE = 'stride'
BLOCK_PAD = 'pad'
BLOCK_PADDING_MODE = 'padding_mode'

BLOCK_ACTIVATION = 'activation'
BLOCK_NEGITIVE_SLOPE = 'negitive_slope'

BLOCK_DROPOUT = 'dropout'

BLOCK_BIAS = 'bias'

ACTIVATION_LINEAR = 'linear'
ACTIVATION_RELU = 'relu'
ACTIVATION_LEAKY = 'leaky'
ACTIVATION_TANH = 'tanh'

LAYER = 'layer'
LAYER_FROM = 'from'
LAYER_NAME = 'name'

TAB = '    '

INFO_LEARNING_RATE = 'learning_rate'
INFO_BETA1 = 'beta1'
INFO_L1_LAMBDA = 'l1_lambda'

NET_INIT_WEIGHT_MEAN = 'init_weight_mean'
NET_INIT_WEIGHT_STDEV = 'init_weight_stdev'

def parse_config(filename:str) -> list[dict]:
    '''
    Read configuration file and return as a list of blocks

    Args:
        filename (str) : file to read

    Returns:
        blocks (list[dict]) : list of blocks
    '''
    filename = os.path.abspath(filename)

    # output
    blocks = []

    # open file
    with open(filename, 'r') as f:
        # create empty block
        block = {}

        # read file
        line = f.readline()
        while line:
            # clean line
            line = line.strip()

            # skip empty lines and comments
            if (len(line) == 0) or (line[0] == CONFIG_COMMENT):
                pass

            # check for new block
            elif line[0] == CONFIG_BLOCK:
                if len(block) > 0:
                    # add block
                    blocks.append(block)

                # create new block
                block = {BLOCK_TYPE : line[1:-1]}

            # add key value pair
            elif CONFIG_ASSIGNMENT in line:
                key, value = tuple(line.split(CONFIG_ASSIGNMENT))

                # store key value pair
                block[key.rstrip()] = value.lstrip()

            # bad line
            else:
                print(f'Cannot interperet line: {line}')

            # read next line
            line = f.readline()

        # add last block
        blocks.append(block)
    
    # output
    return blocks

def create_model(blocks:list[dict]) -> tuple[dict, list[dict], nn.ModuleList]:
    '''
    Create a the optimizer and list of modules from a list of blocks

    Args:
        blocks (list[dict]) : list of blocks

    Returns:
        net (dict) : network information
        modules (list[dict]) : list of modules
    '''
    layer_info = []
    layers = nn.ModuleList()
    
    # store net information
    net = blocks[0]

    # check for net
    if net[BLOCK_TYPE] != CONFIG_NET:
        raise TypeError(f'Expected [{CONFIG_NET}] as the first block but got [{net[BLOCK_TYPE]}]')

    # get initial channels
    features = [int(net[BLOCK_NET_CHANNELS])]

    # itterate
    for block in blocks[1:]:
        # get block feature
        feature = int(block[BLOCK_FEATURE])

        # create convolutional layer
        if block[BLOCK_TYPE] == CONFIG_CONVOLUTIONAL:
            layer = []

            # get values
            try:
                transpose = int(block[BLOCK_TRANSPOSE])
            except KeyError:
                transpose = 0

            try:
                shortcut = int(block[BLOCK_FROM])
            except KeyError:
                shortcut = None

            size = int(block[BLOCK_SIZE])
            stride = int(block[BLOCK_STRIDE])
            pad = int(block[BLOCK_PAD])

            try:
                padding_mode = block[BLOCK_PADDING_MODE]
            except KeyError:
                padding_mode = None

            try:
                batch_norm = int(block[BLOCK_BATCH_NORM])
            except KeyError:
                batch_norm = 0

            activation = block[BLOCK_ACTIVATION]

            try:
                negitive_slope = float(block[BLOCK_NEGITIVE_SLOPE])
            except KeyError:
                negitive_slope = 0.01

            try:
                dropout = float(block[BLOCK_DROPOUT])
            except KeyError:
                dropout = 0.0

            # check for bias
            try:
                bias = eval(block[BLOCK_BIAS]) ##### Not the safest of conversions
            except KeyError:
                if batch_norm:
                    bias = False
                else:
                    bias = True

            # set input size
            in_feature = features[-1]
            if shortcut:
                in_feature += features[shortcut] # include shortcut size

            # add conv2d / convtranspose2d
            if transpose:
                layer.append(nn.ConvTranspose2d(in_feature, feature, size, stride, pad, bias=bias))
            elif padding_mode:
                layer.append(nn.Conv2d(in_feature, feature, size, stride, pad, bias=bias, padding_mode=padding_mode))
            else:
                layer.append(nn.Conv2d(in_feature, feature, size, stride, pad, bias=bias))

            # add batch norm
            if batch_norm:
                layer.append(nn.BatchNorm2d(feature))
                
            # add activation
            if activation == ACTIVATION_RELU:
                layer.append(nn.ReLU())

            elif activation == ACTIVATION_LEAKY:
                layer.append(nn.LeakyReLU(negitive_slope))

            elif activation == ACTIVATION_TANH:
                layer.append(nn.Tanh())

            elif activation != ACTIVATION_LINEAR:
                raise ValueError(f'Activation \'{activation}\' is not recongized')

            # add dropout
            if dropout:
                layer.append(nn.Dropout(dropout))

            # store layer information
            layer_info.append({
                LAYER_FROM : shortcut,
                LAYER_NAME : block[BLOCK_TYPE]
            })

            # add layer
            layers.append(nn.Sequential(*layer))

        # create convolutional layer
        else:
            raise TypeError(f'Block type [{block[BLOCK_TYPE]}] is not a valid type or is not implemented yet')

        # store feature size
        features.append(feature)

    # check for weight initialization
    try:
        mean = float(net[NET_INIT_WEIGHT_MEAN])
        stdev = float(net[NET_INIT_WEIGHT_STDEV])

        # weight init function
        def weight_init(module:nn.Module):
            classname = module.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(module.weight.data, mean, stdev)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(module.weight.data, 1.0, stdev)
                nn.init.constant_(module.bias.data, 0)

        # apply init weights
        layers.apply(weight_init)
    except KeyError:
        pass

    # output
    return net, layer_info, layers

class Model(nn.Module):
    '''
    Class for loading and traning a model

    Args:
        filename (str) : config file to read (default=None)
    '''
    def __init__(self, filename:str=''):
        super().__init__()

        # layers
        self.info = None
        self.layer_info = None
        self.layers = None

        # configure model
        if filename:
            self.configure(filename)

    def configure(self, filename:str) -> None:
        '''
        Configure model from config file

        Args:
            filename

        Returns:
            None
        '''
        # read config file
        blocks = parse_config(filename)

        # create model
        self.info, self.layer_info, self.layers = create_model(blocks)
    
    def forward(self, *inputs:torch.Tensor) -> torch.Tensor:
        '''
        Feed through network

        Args:
            *inputs (torch.Tensor) : function inputs

        Returns:
            out (torch.Tensor) : network output
        '''
        # concat inputs
        x = torch.cat(inputs, dim=1)

        # store activations for skips
        activations = [x]

        # itterate through layers
        for info, layer in zip(self.layer_info, self.layers):
            # skip
            if info[LAYER_FROM]:
                x = torch.cat([x, activations[info[LAYER_FROM]]], dim=1)

            # feed
            x = layer(x)

            # store
            activations.append(x)

        # output
        return activations[-1]
    
    def predict(self, *inputs:torch.Tensor) -> torch.Tensor:
        '''
        Predict from inputs

        Args:
            *inputs (torch.Tensor) : function inputs

        Returns:
            pred (torch.Tensor) : network prediction
        '''
        # no gradients
        with torch.no_grad():
            # predict
            pred = self.forward(*inputs)

        # output
        return pred
    
class Image2Image(nn.Module):
    '''
    Image-to-Image class

    Args:
        device (torch.device) : device
        generator_config (str) : generator config file (default='configs\\generator.cfg')
        discriminator_config (str) : discriminator config file (default='configs\\discriminator.cfg')
    
    '''
    def __init__(self, device:torch.device, generator_config:str='configs\\generator.cfg', discriminator_config:str='configs\\discriminator.cfg'):
        super().__init__()
        
        # store device
        self.device = device
        
        # create models
        self.gen = Model(generator_config).to(device)
        self.dis = Model(discriminator_config).to(device)

        # setup optimizers
        self.opt_gen = torch.optim.Adam(
            self.gen.parameters(),
            lr=float(self.gen.info[INFO_LEARNING_RATE]),
            betas=(float(self.gen.info[INFO_BETA1]), 0.999)
        )
        self.opt_dis = torch.optim.Adam(
            self.dis.parameters(),
            lr=float(self.dis.info[INFO_LEARNING_RATE]),
            betas=(float(self.dis.info[INFO_BETA1]), 0.999)
        )

        # setup loss
        self.BCE_Loss = nn.BCEWithLogitsLoss()
        self.L1_Loss = nn.L1Loss()

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
        y_pred = self.gen(x)

        # real image loss
        real_pred = self.dis(x, y)
        real_loss = self.BCE_Loss(real_pred, torch.ones_like(real_pred))
        
        fake_pred = self.dis(x, y_pred.detach())
        fake_loss = self.BCE_Loss(fake_pred, torch.zeros_like(fake_pred))

        # average loss
        dis_loss = 0.5 * (real_loss + fake_loss)

        # train discriminator
        self.dis.zero_grad()
        dis_loss.backward()
        self.opt_dis.step()
            
        # compute loss
        fake_pred = self.dis(x, y_pred)
        gen_fake_loss = self.BCE_Loss(fake_pred, torch.ones_like(fake_pred))

        gen_loss = gen_fake_loss + self.L1_Loss(y_pred, y) * float(self.gen.info[INFO_L1_LAMBDA])

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
            param_group["lr"] = float(self.gen.info[INFO_LEARNING_RATE])

        for param_group in self.opt_dis.param_groups:
            param_group["lr"] = float(self.dis.info[INFO_LEARNING_RATE])

        # get history
        self.history = checkpoint['history']

        # return epoch
        return checkpoint['epoch']

if __name__ == '__main__':
    gen = Model('configs\\generator.cfg')
    print(gen)
    print(gen(torch.zeros(1, 3, 256, 256)).shape) # torch.Size([1, 3, 256, 256])

    dis = Model('configs\\discriminator.cfg')
    print(dis)
    print(dis(torch.zeros(1, 3, 256, 256), torch.zeros(1, 3, 256, 256)).shape) # torch.Size([1, 1, 59, 59])