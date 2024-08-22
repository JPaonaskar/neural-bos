'''
MODELS

Model reading and training

Referances:

'''
import os
import torch
import torch.nn as nn

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

def create_model(blocks:list[dict]) -> tuple[dict, list[dict]]:
    '''
    Create a the optimizer and list of modules from a list of blocks

    Args:
        blocks (list[dict]) : list of blocks

    Returns:
        net (dict) : network information
        modules (list[dict]) : list of modules
    '''
    layers = []

    # check for net
    if blocks[0][BLOCK_TYPE] != CONFIG_NET:
        raise TypeError(f'Expected [{CONFIG_NET}] as the first block but got [{blocks[0][BLOCK_TYPE]}]')

    # store net information
    net = blocks[0]

    # read config
    features = [int(blocks[0][BLOCK_NET_CHANNELS])]

    # itterate
    for block in blocks[1:]:
        # get block feature
        feature = int(block[BLOCK_FEATURE])

        # create convolutional layer
        if block[BLOCK_TYPE] == CONFIG_CONVOLUTIONAL:
            sequence = []

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
                sequence.append(nn.ConvTranspose2d(in_feature, feature, size, stride, pad, bias=bias))
            elif padding_mode:
                sequence.append(nn.Conv2d(in_feature, feature, size, stride, pad, bias=bias, padding_mode=padding_mode))
            else:
                sequence.append(nn.Conv2d(in_feature, feature, size, stride, pad, bias=bias))

            # add batch norm
            if batch_norm:
                sequence.append(nn.BatchNorm2d(feature))
                
            # add activation
            if activation == ACTIVATION_RELU:
                sequence.append(nn.ReLU())

            if activation == ACTIVATION_LEAKY:
                sequence.append(nn.LeakyReLU(negitive_slope))

            if activation == ACTIVATION_TANH:
                sequence.append(nn.Tanh())

            # add dropout
            if dropout:
                sequence.append(nn.Dropout(dropout))

            # store layer
            layers.append({
                LAYER : nn.Sequential(*sequence),
                LAYER_FROM : shortcut,
                LAYER_NAME : block[BLOCK_TYPE]
            })

        # create convolutional layer
        else:
            raise TypeError(f'Block type [{block[BLOCK_TYPE]}] is not a valid type or is not implemented yet')

        # store feature size
        features.append(feature)

    # output
    return net, layers

class Model(nn.Module):
    '''
    Class for loading and traning a model

    Args:
        filename (str) : config file to read (default=None)
    '''
    def __init__(self, filename:str=None):
        super().__init__()

        # layers
        self.info = None
        self.layers = []

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
        self.info, self.layers = create_model(blocks)

    def __str__(self) -> str:
        '''
        Convert to string

        Args:
            None

        Returns:
            string (str)
        '''
        string = ''

        # add layers
        for i, layer in enumerate(self.layers):
            # add layer
            string += f'({i}) {layer[LAYER_NAME]}\n'

            # add concat
            if layer[LAYER_FROM]:
                string += f'{TAB}<- ({layer[LAYER_FROM]})\n'

            # add sequential
            string += TAB + str(layer[LAYER]).replace('\n', '\n' + TAB) + '\n'

        # output
        return string
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        Feed through network

        Args:
            x (torch.Tensor) : input

        Returns:
            out (torch.Tensor) : network output
        '''
        # store activations for skips
        activations = [x]

        # itterate through layers
        for i, layer in enumerate(self.layers):
            # skip
            if layer[LAYER_FROM]:
                x = torch.cat([x, activations[layer[LAYER_FROM]]], dim=1)

            # feed
            x = layer[LAYER](x)

            # store
            activations.append(x)

        # output
        return activations[-1]

if __name__ == '__main__':
    gen = Model('configs\\generator.cfg')
    #print(gen)
    print(gen(torch.zeros(1, 3, 256, 256)).shape) # torch.Size([1, 3, 256, 256])

    dis = Model('configs\\discriminator.cfg')
    #print(dis)
    print(dis(torch.zeros(1, 6, 256, 256)).shape) # torch.Size([1, 1, 59, 59])