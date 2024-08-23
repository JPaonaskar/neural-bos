'''
SYNTH GEN

Synthetic image generation

Referances:

'''
import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

from perlin_noise import PerlinNoise

import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from typing import Union

import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.RandomCrop(width=256, height=256)
], additional_targets={'target' : 'image'})

transform_input = A.Compose([
    #A.RandomBrightnessContrast(brightness_limit=(-1, 0.1), contrast_limit=(-0.1, 0.1)),
    A.Blur((3, 5), p=0.1),
    ToTensorV2()
])

transform_target = A.Compose([
    ToTensorV2()
])

class Map():
    '''
    Create a sythetic density map

    Args:
        width (int) : map width
        height (int) : map height
        displacement (list[float]) : displacement range

    '''
    def __init__(self, width:int=256, height:int=256, displacement:list[float]=[5.0, 10.0]):
        self.w = width
        self.h = height

        self.d = displacement

    def create(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Create map

        Args:
            None

        Returns:
            mask (np.ndarray) : object mask
            dx (np.ndarray) : x displacement
            dy (np.ndarray) : y displacement
        '''
        mask = np.ones((self.h, self.w))
        dx = np.zeros((self.h, self.w))
        dy = np.zeros((self.h, self.w))

        return mask, dx, dy

class Perlin(Map):
    '''
    Create perlin like density map

    Args:
        width (int) : map width
        height (int) : map height
        octaves (list[int]) : list of octaves
        displacement (list[float]) : displacement range

    '''
    def __init__(self, width:int=256, height:int=256, octaves:list[int]=[6], displacement:list[float]=[5.0, 10.0]):
        self.w = width
        self.h = height

        self.octaves = octaves

        self.d = displacement

    def _sample(self, noise:PerlinNoise):
        '''
        Sample a noise map

        Args:
            noise (PerlinNoise) : noise map

        Returns:
            sample (np.ndarray) : noise sample
        '''
        sample = np.array([[noise([i / self.h, j / self.w]) for i in range(self.h)] for j in range(self.h)])

        return sample

    def create(self):
        # create outputs
        mask = np.ones((self.h, self.w))
        dx = np.zeros((self.h, self.w))
        dy = np.zeros((self.h, self.w))

        # sample noise
        for i, octave in enumerate(self.octaves):
            scale = pow(0.5, i)

            # add dx noise map
            noise = PerlinNoise(octave)
            dx += self._sample(noise) * scale

            # add dy noise map
            noise = PerlinNoise(octave)
            dy += self._sample(noise) * scale

        # get displacement magnitude
        d = np.random.rand()
        d = (self.d[1] - self.d[0]) * d + self.d[0]

        # convert to displacement
        dx = dx * d
        dy = dy * d

        # output
        return mask, dx, dy

class Mach(Map):
    '''
    Create machflow like density map

    Args:
        None

    '''

class Laminar_Candle(Map):
    '''
    Create laminar candle like denisty map

    Args:
        None

    '''

class Density_Maps(Map):
    '''
    Class to create random density maps

    Args:
        *maps (list[ tuple[float, Map] ]) : list of maps and their probabilities

    '''
    def __init__(self, *maps:list[tuple[float, Map]]):
        self.maps = maps

        # get displacement
        self.d = self.maps[0][1].d

        for _, d_map in self.maps[1:]:
            self.d = [min(self.d[0], d_map.d[0]), max(self.d[1], d_map.d[1])]

    def create(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Create a random density map

        Args:
            None

        Returns:
            mask (np.ndarray) : object mask
            dx (np.ndarray) : x displacement
            dy (np.ndarray) : y displacement
        '''
        x = np.random.rand()
        p = 0

        # get map
        for prob, d_map in self.maps:
            p += prob

            # x is within probability
            if p > x:
                # output map
                return d_map.create()
            
        # probabilities don't add up
        raise ValueError(f'Expected probabilitys to add up to 1.0 but got {p}')


class BOS_Dataset_Generator():
    '''
    Generate a BOS dataset

    Args:
        density_map (Map) : density map gernator
        length (int) : dataset length (defualt=1600)
        width (int) : image width (defualt=256)
        height (int) : image height (defualt=256)
        detail (int) : subpixel level (default=10)
        displacement (int) : maximum displacements (default=10)
        octaves (int) : noise octives (default=4)

    '''
    def __init__(self, density_map:Map, length:int=1600, width:int=256, height:int=256, detail:int=10):
        self.length = length

        self.w = width
        self.h = height
        self.n = detail

        self.d_map = density_map
        self.d = int(np.ceil(self.d_map.d[1]))

        # create image storage
        self.input_images = []
        self.target_images = []

    def generate(self) -> None:
        '''
        Generate dataset

        Args:
            None

        Returns:
            None
        '''
        # build dataset
        print('Generating Synthetic Dataset')
        for i in tqdm(range(self.length)):
            # gernate image
            input_image, target_image = self._gen()

            # store 
            self.input_images.append(input_image)
            self.target_images.append(target_image)

    def _trace(self, background:np.ndarray, dx:np.ndarray, dy:np.ndarray) -> np.ndarray:
        '''
        Trace 'rays' through density map

        Args:
            background (np.ndarray) : background image
            dx (np.ndarray) : x displacement
            dy (np.ndarray) : y displacement

        Returns:
            displaced (np.ndarray) : displaced background
        '''
        # resize
        background = cv2.resize(background, (background.shape[1] * self.n, background.shape[0] * self.n), interpolation=cv2.INTER_NEAREST)

        # build displaced image
        displaced = np.zeros((self.h, self.w))

        # indexes
        row, col = np.meshgrid(np.arange(self.h), np.arange(self.w))
        row = row.flatten()
        col = col.flatten()

        # pull
        row_bg = np.around((self.d + row + dy.flatten()) * self.n).astype(np.uint16)
        col_bg = np.around((self.d + col + dx.flatten()) * self.n).astype(np.uint16)

        displaced[row, col] = background[row_bg, col_bg]

        # output
        return displaced

    def _gen(self) -> None:
        '''
        Generate a single synthetic image pair

        Args:
            None

        Returns:
            None
        '''
        # create background
        background = np.random.random((self.h + 2 * self.d, self.w + 2 * self.d))

        mask = background > 0.5
        background[mask] = 1.0
        background[~mask] = -1.0

        # create density map
        mask, dx, dy = self.d_map.create()

        # build displaced image
        displaced = self._trace(background, dx, dy)

        # crop background
        background = background[self.d:-self.d, self.d:-self.d]

        # mask backgound
        background[mask == 0] = -1.0

        # build images
        input_image = np.dstack([background, displaced, np.zeros_like(background)])
        target_image = np.dstack([dx, dy, np.sqrt(np.square(dx) + np.square(dy))])

        # return image
        return input_image, target_image
    
    def _build_i2i(self, input_image:np.ndarray, target_image:np.ndarray) -> np.ndarray:
        '''
        Create an I2I pair to save

        Args:
            input_image (np.ndarray) : input image
            target_image (np.ndarray) : target image

        Returns:
            image (np.ndarray) : I2I pair
        '''
        # reformat input image
        input_image = (input_image * 127 + 127).astype(np.uint8)

        # reformat displacements
        norm = np.sqrt(2) * self.d
        target_image = (target_image / norm * 127 + 127).astype(np.uint8)

        # stack together
        image = np.hstack([input_image, target_image])

        # to bgr
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # output
        return image

    def save(self, dirname:str, name:str='') -> None:
        '''
        Save current dataset images to directory

        Args:
            dirname (str) : path to save directory
            name (str) : naming prefix (default='')

        Returns:
            None
        '''
        # get full path and number of existing files
        dirname = os.path.abspath(dirname)
        imgs = len(os.listdir(dirname))

        # loop through dataset
        print('Saving Dataset')
        for i in tqdm(range(self.length)):
            input_image = self.input_images[i]
            target_image = self.target_images[i]

            # reformat
            image = self._build_i2i(input_image, target_image)

            # save
            cv2.imwrite(os.path.join(dirname, f'{name}{i+imgs:04d}_d={self.d}.png'), image)

    def build(self, dirname:str, name:str='', workers:int=16) -> None:
        '''
        Generate and save image by image

        Args:
            dirname (str) : path to save directory
            name (str) : naming prefix (default='')
            workers (int) : number of worker threads

        Returns:
            None
        '''
        # get full path and number of existing files
        dirname = os.path.abspath(dirname)
        imgs = len(os.listdir(dirname))
        
        # multiprocessing for speed
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []

            # itterate though sections
            print('Starting Dataset Build')
            for i in tqdm(range(self.length)):
                # start process
                futures.append(executor.submit(self._gen))

                # distribute threads
                time.sleep(0.1)

            # get frames
            print('Building Dataset')
            for i, future in enumerate(tqdm(futures)):
                # create image
                input_image, target_image = future.result()
                image = self._build_i2i(input_image, target_image)
                
                # save image
                cv2.imwrite(os.path.join(dirname, f'{name}{i+imgs:04d}_d={self.d}.tif'), image)

class BOS_Dataset(Dataset):
    '''
    BOS Dataset

    Args:
        dirname (str) : path to saved BOS I2I pairs
        clamped (bool) : values are percentages not desplacements (default=True)
    '''
    def __init__(self, dirname:str, clamped:bool=True):
        # store root directory and get file contents
        self.dirname = os.path.abspath(dirname)
        self.files = os.listdir(self.dirname)

        self.clamped = clamped

    def __len__(self) -> int:
        '''
        Length of dataset

        Args:
            None

        Returns:
            len (int) : length
        '''
        return len(self.files)

    def __getitem__(self, idx:Union[int, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Get dataset item at index

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
        path = os.path.join(self.dirname, self.files[idx])
        image = cv2.imread(path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # slice image
        _, w, _ = image.shape

        cut = w // 2

        input_image = image[:, :cut, :].astype(np.float32)
        target_image = image[:, cut:, :].astype(np.float32)

        # normalize input image
        input_image = (input_image - 127.5) / 127.5

        # normailize target and convert to displacements
        target_image = (target_image - 127.5) / 127.5

        # convert to displacements
        if not self.clamped:
            # get displacement size
            d = os.path.basename(path)
            d = os.path.splitext(d)[0]
            d = d.split('=')[1].strip()
            d = float(d)

            # scale
            target_image = target_image * d

        # augmentations
        transformed = transform(image=input_image, target=target_image)
        input_image = transformed['image']
        target_image = transformed['target']

        input_image = transform_input(image=input_image)['image']
        target_image = transform_target(image=target_image)['image']

        # output
        return input_image, target_image

if __name__ == '__main__':
    import utils
    from torch.utils.data import DataLoader

    # create density maps
    map1 = Perlin(width=512, height=512, octaves=[6])
    map2 = Perlin(width=512, height=512, octaves=[6, 12])
    map3 = Perlin(width=512, height=512, octaves=[6, 12, 24])
    map4 = Perlin(width=512, height=512, octaves=[8])
    map5 = Perlin(width=512, height=512, octaves=[8, 16])
    map6 = Perlin(width=512, height=512, octaves=[8, 16, 32])
    map7 = Perlin(width=512, height=512, octaves=[4])
    map8 = Perlin(width=512, height=512, octaves=[4, 8])

    d_map = Density_Maps((0.125, map1), (0.125, map2), (0.125, map3), (0.125, map4), (0.125, map5), (0.125, map6), (0.125, map7), (0.125, map8))

    # create dataset
    data = BOS_Dataset_Generator(d_map, length=800, width=512, height=512)
    data.build('datasets\\bos\\train')

    # show part of the dataset
    dataset = BOS_Dataset('datasets\\bos\\train')
    loader = DataLoader(dataset, batch_size=25)

    x, y = next(iter(loader))

    # plot
    utils.plot_bos_images(x)
    utils.plot_images(x, 'Input Images', rows=2, cols=2, show=False)
    utils.plot_images(y, 'Target Images', rows=2, cols=2)