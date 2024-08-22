'''
SYNTH GEN

Synthetic image generation

Referances:

'''
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from typing import Union

class Synth_Dataset(Dataset):
    '''
    Synthetic BOS dataset 

    Args:
        length (int) : dataset length
        width (int) : image width
        height (int) : image height
        detail (int) : upscaling

    '''
    def __init__(self, length:int=1600, width:int=256, height:int=256, detail:int=10):
        self.length = length

        self.w = width
        self.h = height
        self.n = detail

        # create images
        self.input_images = []
        self.target_images = []

        # build dataset
        print('Generating Synthetic Dataset')
        for i in tqdm(range(length)):
            self._gen()

    def _gen(self) -> None:
        '''
        Generate a single synthetic image pair

        Args:
            None

        Returns:
            None
        '''
        # create background
        background = np.random.random((3, self.h, self.w), dtype=np.float16)

        mask = background > 0.5
        background[mask] = 255.0
        background[~mask] = 0.0

        background = cv2.resize(background, (self.w * self.n, self.h * self.n), interpolation=cv2.INTER_NEAREST)

        # create density map
        ##### PERLIN / SIMPLEX #####
    
    def __len__(self) -> int:
        '''
        Length of dataset

        Args:
            None

        Returns:
            len (int) : length
        '''
        return self.length

    def __getitem__(self, idx:Union[list, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Get dataset item at index

        Args:
            idx (list, torch.Tensor) : indexes

        Returns:
            input_image (torch.Tensor) : Input image
            target_image (torch.Tensor) : Target image
        '''
        return None, None