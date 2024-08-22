'''
SYNTH GEN

Synthetic image generation

Referances:

'''
import os
import cv2
import numpy as np

from perlin_noise import PerlinNoise

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

class BOS_Dataset_Generator():
    '''
    Generate a BOS dataset

    Args:
        length (int) : dataset length (defualt=1600)
        width (int) : image width (defualt=256)
        height (int) : image height (defualt=256)
        detail (int) : subpixel level (default=10)
        displacement (int) : maximum displacements (default=10)
        octaves (int) : noise octives (default=4)

    '''
    def __init__(self, length:int=1600, width:int=256, height:int=256, detail:int=10, displacement:int=10, octaves:int=4):
        self.length = length

        self.w = width
        self.h = height
        self.n = detail

        self.d = displacement
        self.octaves = octaves

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

        # resize
        background_upscaled = cv2.resize(background, (background.shape[1] * self.n, background.shape[0] * self.n), interpolation=cv2.INTER_NEAREST)

        # create noise maps
        noise = PerlinNoise(octaves=self.octaves)

        dx = np.array([[noise([0.5 * i / self.h, 0.5 * j / self.w]) for i in range(self.h)] for j in range(self.h)])
        dy = np.array([[noise([1 - 0.5 * i / self.h, 1 - 0.5 * j / self.w]) for i in range(self.h)] for j in range(self.h)])

        # convert noise to displacements
        dx = dx * self.d
        dy = dy * self.d

        # build displaced image
        displaced = np.zeros((self.h, self.w))

        # indexes
        row, col = np.meshgrid(np.arange(self.h), np.arange(self.w))
        row = row.flatten()
        col = col.flatten()

        # pull
        row_bg = np.around((self.d + row + dy.flatten()) * self.n).astype(np.uint16)
        col_bg = np.around((self.d + col + dx.flatten()) * self.n).astype(np.uint16)

        displaced[row, col] = background_upscaled[row_bg, col_bg]

        # crop background
        background = background[self.d:-self.d, self.d:-self.d]

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
        target_image = (target_image / self.d * 127 + 127).astype(np.uint8)

        # stack together
        image = np.hstack([input_image, target_image])

        # to bgr
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # output
        return image

    def save(self, dirname:str, name:str=''):
        '''
        Save current dataset images to directory

        Args:
            dirname (str) : path to save directory
            name (str) : naming prefix (default='')
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
            cv2.imwrite(os.path.join(dirname, f'{name}{i+imgs:04d}_d={self.d}.jpg'), image)

    def build(self, dirname:str, name:str=''):
        '''
        Generate and save image by image

        Args:
            dirname (str) : path to save directory
            name (str) : naming prefix (default='')
        '''
        # get full path and number of existing files
        dirname = os.path.abspath(dirname)
        imgs = len(os.listdir(dirname))

        # save function
        def make(i) -> tuple[int, np.ndarray]:
            # gernate image
            input_image, target_image = self._gen()

            # reformat
            image = self._build_i2i(input_image, target_image)

            # output
            return i, image
        
        # threading for speed
        with ThreadPoolExecutor() as executor:
            futures = []

            # itterate though sections
            print('Starting Dataset Build')
            for i in tqdm(range(self.length)):
                # start process
                futures.append(executor.submit(make, i))

            # get frames
            print('Building Dataset')
            for future in tqdm(futures):
                # save images
                i, image = future.result()
                
                # save
                cv2.imwrite(os.path.join(dirname, f'{name}{i+imgs:04d}_d={self.d}.jpg'), image)
            

if __name__ == '__main__':
    import utils
    import dataset
    from torch.utils.data import DataLoader

    # create dataset
    data = BOS_Dataset_Generator(length=4)
    data.build('datasets\\bos\\train')

    #utils.plot_bos_images(data.input_images)

    # show part of the dataset
    dataset = dataset.I2I_Dataset('datasets\\bos\\train')
    loader = DataLoader(dataset, batch_size=25)

    x, y = next(iter(loader))

    # plot
    utils.plot_bos_images(x)
    utils.plot_images(x, 'Input Images', show=False)
    utils.plot_images(y, 'Target Images')