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
    #A.RandomShadow((0, 0, 1, 1), num_shadows_upper=2, shadow_dimension=5, shadow_intensity_range=(0.1, 0.3), p=0.1),
    A.Blur((3, 5), p=0.1),
    ToTensorV2()
])

transform_target = A.Compose([
    ToTensorV2()
])

transform_backgound = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.1), contrast_limit=(-0.2, 0.0)),
    A.RandomScale((0.0, 0.2), interpolation=cv2.INTER_LINEAR, always_apply=True),
    A.RandomRotate90(p=0.2),
    A.Flip(p=0.7),
    A.Blur(3, p=0.5)
])

class Map():
    '''
    Map template class

    Args:
        width (int) : map width
        height (int) : map height
        displacement (list[float]) : displacement range

    '''
    def __init__(self, width:int=256, height:int=256, displacement:list[float]=[5.0, 10.0]):
        self.w = width
        self.h = height

        self.d = displacement

    def create(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Create map

        Args:
            None

        Returns:
            bg_mask (np.ndarray) : background object mask
            dp_mask (np.ndarray) : displaced object mask
            dx (np.ndarray) : x displacement
            dy (np.ndarray) : y displacement
        '''
        bg_mask = np.ones((self.h, self.w))
        dp_mask = np.ones((self.h, self.w))
        dx = np.zeros((self.h, self.w))
        dy = np.zeros((self.h, self.w))

        return bg_mask, dp_mask, dx, dy

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

    def _sample(self, noise:PerlinNoise) -> np.ndarray:
        '''
        Sample a noise map

        Args:
            noise (PerlinNoise) : noise map

        Returns:
            sample (np.ndarray) : noise sample
        '''
        sample = np.array([[noise([i / self.h, j / self.w]) for i in range(self.h)] for j in range(self.h)])

        return sample

    def create(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Create map

        Args:
            None

        Returns:
            bg_mask (np.ndarray) : background object mask
            dp_mask (np.ndarray) : displaced object mask
            dx (np.ndarray) : x displacement
            dy (np.ndarray) : y displacement
        '''
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
        return mask, mask.copy(), dx, dy

class Mach(Map):
    '''
    Create machflow like density map

    Args:
        None

    '''

class Candle(Map):
    '''
    Create candle like denisty map

    Args:
        width (int) : map width
        height (int) : map height
        displacement (list[float]) : displacement range

    '''
    def __init__(self, width:int=256, height:int=256, displacement:list[float]=[5.0, 10.0], angle:float=30, a:int=20, b:int=10, thickness:int=10, radius:int=40, blur:int=13):
        self.w = width
        self.h = height

        self.d = displacement
        
        self.a = a
        self.b = b

        self.ang = angle
        self.r = radius
        self.t = thickness

        self.blur = blur

    def create(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Create map

        Args:
            None

        Returns:
            bg_mask (np.ndarray) : background object mask
            dp_mask (np.ndarray) : displaced object mask
            dx (np.ndarray) : x displacement
            dy (np.ndarray) : y displacement
        '''
        # create outputs
        mask = np.ones((self.h, self.w))
        dx = np.zeros((self.h, self.w))
        dy = np.zeros((self.h, self.w))

        # random values
        ang = self.ang * (2 * np.random.rand() - 1)
        d = np.random.rand() * (self.d[1] - self.d[0]) + self.d[0]

        # center
        cy = self.h - self.a
        cx = self.w // 2

        # place candle
        cv2.ellipse(mask, (cx, cy), (self.b, self.a), ang, 0, 360, 0, -1)
        ang = -np.deg2rad(ang)
        
        # draw laminar lines
        for img in [dx, dy]:
            for x in [cx - self.b, cx + self.b]:
                # displacement value
                val = (2 * np.random.rand() - 1) * d

                # end points
                vx = round(np.sin(ang) * self.h / 2 + np.random.randn())
                vy = round(np.cos(ang) * self.h / 2 + np.random.randn())

                # draw lines
                cv2.line(img, (x, cy), (x - vx, cy - vy), val, self.t)

        # draw turbulance
        for img in [dx, dy]:
            x = self.w * (1 - np.sin(ang)) * 0.5
            y = self.h // 2

            # draw circles
            while y > 0:
                # displacement value
                val = (2 * np.random.rand() - 1) * d

                # random arc
                r = (self.r - self.t) * np.random.rand() + self.t
                a = np.random.randint(0, 360)
                b = np.random.randint(60, 360)

                # draw arc
                cv2.ellipse(img, (round(x), round(y)), (round(r), round(r)), a, 0, b, val, self.t)

                # take radius size step
                x -= np.sin(ang) * r
                y -= np.cos(ang) * r

                # go crazy
                x -= r * np.random.randn() * 0.5
                y -= r * np.random.randn() * 0.5

        # blur displacements
        ksize = (self.blur, self.blur)
        dx = cv2.GaussianBlur(dx, ksize, 0)
        dy = cv2.GaussianBlur(dy, ksize, 0)

        # output
        return np.ones_like(mask), mask, dx, dy

class Compose_Maps(Map):
    '''
    Class to create random density maps

    Args:
        *maps (list[ tuple[float, Map] ]) : list of maps and their probabilities

    '''
    def __init__(self, *maps:list[tuple[float, Map]]):
        self.maps = maps

        # get initial displacement and probability
        self.d = self.maps[0][1].d
        prob = self.maps[0][0]

        # update displacement and check probability
        for p, d_map in self.maps[1:]:
            self.d = [min(self.d[0], d_map.d[0]), max(self.d[1], d_map.d[1])]
            prob += p

        # invalid probability
        if prob != 1:
            raise ValueError(f'Expected proabilities to sum to 1.0 but got {prob}')

    def create(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Create a random density map

        Args:
            None

        Returns:
            bg_mask (np.ndarray) : background object mask
            dp_mask (np.ndarray) : displaced object mask
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
    
class Background():
    '''
    Background template class

    Args:
        width (int) : background width
        height (int) : background height

    '''
    def __init__(self, width:int=2760, height:int=2760):
        self.w = width
        self.h = height

    def create(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Create background

        Args:
            None

        Returns:
            bg_mask (np.ndarray) : background
        '''
        background = np.zeros((self.h, self.w))

        return background
    
class Synth_Background():
    '''
    Synthetic background

    Args:
        width (int) : background width
        height (int) : background height
        size (int) : pixel size

    '''
    def __init__(self, width:int=2760, height:int=2760, size:int=10):
        self.w = width
        self.h = height

        self.size = size

    def create(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Create background

        Args:
            None

        Returns:
            bg_mask (np.ndarray) : background
        '''
        # create noise
        background = np.random.random((self.h // self.size, self.w // self.size))

        # mask
        mask = background > 0.5
        background[mask] = 1.0
        background[~mask] = -1.0

        # resize
        background = cv2.resize(background, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        return background
    
class Hybrid_Background():
    '''
    Background from real image

    Args:
        width (int) : background width
        height (int) : background height
        dirname (str) : directory with backgrounds

    '''
    def __init__(self, dirname:str='backgrounds', width:int=2760, height:int=2760):
        self.w = width
        self.h = height

        # get list of files
        self.dirname = os.path.abspath(dirname)
        self.images = os.listdir(self.dirname)

    def create(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Create background

        Args:
            None

        Returns:
            background (np.ndarray) : background
        '''
        # read random background image
        ind = int(np.random.rand() * len(self.images))
        background = cv2.imread(os.path.join(self.dirname, self.images[ind]))

        # augmentations
        background = transform_backgound(image=background)['image']

        # grayscale
        background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)

        # trim
        ar = self.w / self.h
        
        h, w = background.shape
        ar_img = w / h

        if ar_img > ar:
            background = background[:, :round(h * ar)]
        else:
            background = background[:round(w / ar), :]

        # resize
        background = cv2.resize(background, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

        # normalize
        background = background.astype(np.float32) / 127.5 - 1

        return background

class Compose_Backgrounds(Background):
    '''
    Class to create random density backgrounds

    Args:
        *backgrounds (list[ tuple[float, Map] ]) : list of backgrounds and their probabilities

    '''
    def __init__(self, *backgrounds:list[tuple[float, Map]]):
        self.backgrounds = backgrounds

        # get probability
        prob = self.backgrounds[0][0]

        # update probability
        for p, _ in self.backgrounds[1:]:
            prob += p

        # invalid probability
        if prob != 1:
            raise ValueError(f'Expected proabilities to sum to 1.0 but got {prob}')

    def create(self) -> np.ndarray:
        '''
        Create a random density map

        Args:
            None

        Returns:
            background (np.ndarray) : background
        '''
        x = np.random.rand()
        p = 0

        # get map
        for prob, background in self.backgrounds:
            p += prob

            # x is within probability
            if p > x:
                # output map
                return background.create()

class BOS_Dataset_Generator():
    '''
    Generate a BOS dataset

    Args:
        density_map (Map) : density map gerenator
        background (Background) : background generator
        length (int) : dataset length (defualt=1600)
        width (int) : image width (defualt=256)
        height (int) : image height (defualt=256)
        displacement (int) : maximum displacements (default=10)
        octaves (int) : noise octives (default=4)

    '''
    def __init__(self, density_map:Map, background:Background, length:int=1600, width:int=256, height:int=256):
        self.length = length

        self.w = width
        self.h = height

        self.d_map = density_map
        self.d = int(np.ceil(self.d_map.d[1]))

        self.background = background

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
            background_out (np.ndarray) : resized background
            displaced (np.ndarray) : displaced background
        '''
        # build displaced image
        displaced = np.zeros((self.h, self.w))
        background_out = np.zeros((self.h, self.w))

        # indexes
        col, row = np.meshgrid(np.arange(self.w), np.arange(self.h))

        # size change
        sh = background.shape[0] / (self.h + 2 * self.d)
        sw = background.shape[1] / (self.w + 2 * self.d)

        # pull displaced
        row_bg = np.around((self.d + row + dy) * sh).astype(np.uint16)
        col_bg = np.around((self.d + col + dx) * sw).astype(np.uint16)

        displaced[row, col] = background[row_bg, col_bg]

        # pull background
        row_bg = np.around((self.d + row) * sh).astype(np.uint16)
        col_bg = np.around((self.d + col) * sw).astype(np.uint16)

        background_out[row, col] = background[row_bg, col_bg]

        # output
        return background_out, displaced

    def _gen(self) -> None:
        '''
        Generate a single synthetic image pair

        Args:
            None

        Returns:
            None
        '''
        # generate background
        background = self.background.create()

        # create density map
        bg_mask, dp_mask, dx, dy = self.d_map.create()

        # build displaced image
        background, displaced = self._trace(background, dx, dy)

        # mask
        background[bg_mask == 0] = -1.0
        displaced[dp_mask == 0] = -1.0

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
                if i < workers:
                    time.sleep(0.5)

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

        self.clamped = clamped

        # image values
        self.input_images = []
        self.target_images = []

        # load images
        print('Loading Images')
        for file in tqdm(os.listdir(dirname)):
            # get image
            path = os.path.join(dirname, file)
            image = cv2.imread(path)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # cut images
            input_image, target_image = self._cut(image)

            # convert to displacements
            if not self.clamped:
                # get displacement size
                d = os.path.basename(path)
                d = os.path.splitext(d)[0]
                d = d.split('=')[1].strip()
                d = float(d)

                # convert
                target_image = target_image * d

            # augmentations
            transformed = transform(image=input_image, target=target_image)
            input_image = transformed['image']
            target_image = transformed['target']

            input_image = transform_input(image=input_image)['image']
            target_image = transform_target(image=target_image)['image']

            # add images
            self.input_images.append(input_image)
            self.target_images.append(target_image)

    def _cut(self, image:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''
        Cut dataset image into a pair

        Args:
            image (np.ndarray) : dataset image

        Returns:
            input_image (np.ndarray) : input image
            target_image (np.ndarray) : target image
        '''
        # slice image
        _, w, _ = image.shape

        cut = w // 2

        input_image = image[:, :cut, :].astype(np.float32)
        target_image = image[:, cut:, :].astype(np.float32)

        # normalize input image
        input_image = (input_image - 127.5) / 127.5

        # normailize target
        target_image = (target_image - 127.5) / 127.5

        # output
        return input_image, target_image

    def __len__(self) -> int:
        '''
        Length of dataset

        Args:
            None

        Returns:
            len (int) : length
        '''
        return len(self.input_images)

    def __getitem__(self, ind:Union[int, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Get dataset item at index

        Args:
            ind (list, torch.Tensor) : indexes

        Returns:
            input_image (torch.Tensor) : Input image
            target_image (torch.Tensor) : Target image
        '''
        # convert to index
        if torch.is_tensor(ind):
            ind = ind.tolist()

        # get images
        input_image = self.input_images[ind]
        target_image = self.target_images[ind]

        # output
        return input_image, target_image

if __name__ == '__main__':
    import utils
    from torch.utils.data import DataLoader

    # create density maps
    map0 = Candle(width=512, height=512)
    map1 = Candle(width=512, height=512, thickness=2)
    map2 = Candle(width=512, height=512, radius=50)

    map3 = Perlin(width=512, height=512, octaves=[3])
    map4 = Perlin(width=512, height=512, octaves=[4])
    map5 = Perlin(width=512, height=512, octaves=[6])
    map6 = Perlin(width=512, height=512, octaves=[2, 4])

    d_map = Compose_Maps((0.2, map0), (0.1, map1), (0.2, map2), (0.1, map3), (0.15, map4), (0.15, map5), (0.1, map6))

    # create backgrounds
    bg0 = Hybrid_Background(width=5320, height=5320)
    bg1 = Hybrid_Background(width=5320, height=5320)

    background = Compose_Backgrounds((0.5, bg0), (0.5, bg1))

    # create dataset
    data = BOS_Dataset_Generator(d_map, background, length=800, width=512, height=512)
    data.build('datasets\\hybrid\\val')

    # show part of the dataset
    dataset = BOS_Dataset('datasets\\hybrid\\val')
    loader = DataLoader(dataset, batch_size=25)

    # get last batch
    x, y = None, None
    for xi, yi in loader:
        x = xi
        y = yi

    # plot
    utils.plot_bos_images(x)
    utils.plot_images(x, 'Input Images', rows=5, cols=5, show=False)
    utils.plot_images(y, 'Target Images', rows=5, cols=5)