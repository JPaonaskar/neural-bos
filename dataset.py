'''
DATASET

Dataset reader

Resources:

'''
import os
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from typing import Union

ROOT = 'datasets'

transform_input = A.Compose([
    A.Resize(width=256, height=256),
    ToTensorV2()
])

transform_target = A.Compose([
    A.Resize(width=256, height=256),
    ToTensorV2()
])

class I2I_Dataset(Dataset):
    '''
    Image-to-Image dataset

    Args:
        root (str) : path to root directory

    '''
    def __init__(self, root:str):
        # store root directory and get file contents
        self.root = os.path.abspath(root)
        self.files = os.listdir(self.root)

    def __len__(self) -> int:
        '''
        Length of dataset

        Args:
            None

        Returns:
            len (int) : length
        '''
        return len(self.files)

    def __getitem__(self, idx:Union[list, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Length of dataset

        Args:
            idx (list, torch.Tensor) : indexes

        Returns:
            input_image (torch.Tensor) : Input image
            target_image (torch.Tensor) : Target image
        '''
        # convert to index
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image
        path = os.path.join(self.root, self.files[idx])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # slice image
        _, w, _ = image.shape

        cut = w // 2

        input_image = image[:, :cut, :]
        target_image = image[:, cut:, :]

        # augmentations
        input_image = transform_input(image=input_image)['image']
        target_image = transform_target(image=target_image)['image']

        # output
        return input_image, target_image

if __name__ == '__main__':
    # show part of the dataset
    dataset = I2I_Dataset('datasets\\maps\\train')
    loader = DataLoader(dataset, batch_size=64)

    x, y = next(iter(loader))

    # plot 
    for name, data in [('Input Images', x), ('Target Images', y)]:
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle(name, fontsize=16)
        for i in range(25):
            plt.subplot(5, 5, i+1)

            plt.xticks([])
            plt.yticks([])
            plt.grid(False)

            plt.imshow(data[i].permute(1,2,0), cmap=plt.cm.binary)
    plt.show()