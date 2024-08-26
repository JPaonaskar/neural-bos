'''
PREDICT

Predict output from video

Referances:

'''
import torch
from torch.utils.data import Dataset

import os
import cv2
import numpy as np

import imageio

from tqdm import tqdm
from typing import Union

import pix2pix

class List_Dataset(Dataset):
    '''
    List of frames as a dataset (no target image)

    Args:
        images (Union[list, np.ndarray]) : list of images
    '''
    def __init__(self, images:Union[list, np.ndarray]):
        self.images = images

    def __len__(self) -> int:
        '''
        Length of dataset

        Args:
            None

        Returns:
            len (int) : length
        '''
        return len(self.images)

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

        # make image pair
        image = self.images[ind]

        # covnert to torch tensor
        image = torch.Tensor(image).permute(2, 0, 1)

        # output
        return image, torch.zeros_like(image)
    
class Video_Compute():
    '''
    Compute a video

    Args:
        checkpoint (str) : path to latest checkpoint
        video (str) : path to video (default=None)
        device (torch.device) : model device (default=torch.device('cuda'))
    '''
    def __init__(self, checkpoint:str, video:str=None, device:torch.device=torch.device('cuda')):
        # create model
        self.model = pix2pix.Pix2PixModel(device)
        self.model.load_checkpoint(checkpoint)

        # store frames
        self.frames = []

        # store computed results
        self.computed = []

        # read video
        if video:
            self.read(video)

    def read(self, video:str) -> None:
        '''
        Read video

        Args:
            video (str) : path to video

        Returns:
            None
        '''
        # get full path
        video = os.path.abspath(video)

        # open video feed
        cap = cv2.VideoCapture(video)

        # read frames
        self.frames = []
        while cap.isOpened():
            # get frame
            ret, frame = cap.read()

            # stop and end / bad frame
            if not ret:
                print(f'Read {len(self.frames)} frames')
                break

            # save frame
            self.frames.append(frame)

        # close capture
        cap.release()

    def compute(self, filename:str, fps:int=30, overlap:float=0.5, start:int=0, stop:int=None, step:int=1, colormap:int=None) -> None:
        '''
        Compute displacements with cascade pairing

        Args:
            filename (str) : path to save to
            fps (int) : frames per second
            start (int) : starting index
            stop (int) : starting index
            step (int) : starting index
            colormap (int) : colormap to use (default=None)

        Returns:
            None
        '''
        # get referance image
        ref = cv2.cvtColor(self.frames[start], cv2.COLOR_BGR2GRAY)
        h, w = ref.shape

        # get stop value
        if not stop:
            stop = len(self.frames) - start

        # get frames
        frames = self.frames[start+1:stop:step]

        # create padded image pairs
        row_count = np.ceil(h / pix2pix.IMAGE_HEIGHT).astype(int)
        col_count = np.ceil(w / pix2pix.IMAGE_WIDTH).astype(int)

        images = np.zeros((
            len(frames),
            row_count * pix2pix.IMAGE_HEIGHT,
            col_count * pix2pix.IMAGE_WIDTH,
            3
        ), dtype=np.float32)

        print('Pairing and Converting Frames')
        for i, frame in enumerate(tqdm(frames)):
            # convert to grayscale
            disp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # stack
            image = np.dstack([ref, disp, np.zeros_like(ref)]).astype(np.float32)

            # normalize
            image = image / 127.5 - 1

            # add
            images[i][:h, :w, :] = image

        # free memory
        del ref
        del frames
        
        # subdivide into image set
        print(f'Predicting and Writing')

        # open video writer
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video_out = cv2.VideoWriter(filename, fourcc, fps, (w, h))

        # create lists
        c_step = (1.0 - overlap)
        rows = np.arange(0, row_count, c_step)
        cols = np.arange(0, col_count, c_step)

        # bound
        rows = rows[rows <= row_count - 1]
        cols = cols[cols <= col_count - 1]

        # write video
        for image in tqdm(images):
            # placeholders
            counts = np.zeros_like(image, dtype=np.float32)
            computed = np.zeros_like(image, dtype=np.float32)

            # build dataset
            dataset = []
            for row in rows:
                for col in cols:
                    y1 = int(row * pix2pix.IMAGE_HEIGHT)
                    x1 = int(col * pix2pix.IMAGE_WIDTH)
                    y2 = int(y1 + pix2pix.IMAGE_HEIGHT)
                    x2 = int(x1 + pix2pix.IMAGE_WIDTH)

                    # add
                    dataset.append(image[y1:y2, x1:x2, :])

            # create as dataset
            dataset = List_Dataset(dataset)

            # predict
            _, _, pred = self.model.predict(dataset, shuffle=False, single_batch=False)
            pred = pred.permute(0, 2, 3, 1).cpu().numpy()

            # reconstruct
            for i, row in enumerate(rows):
                for j, col in enumerate(cols):
                    y1 = int(row * pix2pix.IMAGE_HEIGHT)
                    x1 = int(col * pix2pix.IMAGE_WIDTH)
                    y2 = int(y1 + pix2pix.IMAGE_HEIGHT)
                    x2 = int(x1 + pix2pix.IMAGE_WIDTH)

                    # add image
                    image = pred[i * len(cols) + j]
                    computed[y1:y2, x1:x2, :] += image
                    counts[y1:y2, x1:x2, :] += np.ones_like(image)

            # average
            computed = computed / counts

            # crop
            computed = computed[:h, :w, :]

            # colormap
            if colormap:
                # average displacement length (sqrt(dx ^ 2 + dy ^ 2) & d_mag)
                computed = np.sqrt(np.square(computed[:, :, 0]) + np.square(computed[:, :, 1])) + computed[:, :, 2]

                # convert to uint8
                computed = (computed * 127.5).astype(np.uint8)

                # colormap
                computed = cv2.applyColorMap(computed, colormap)
            
            # raw
            else:
                # convert
                computed = (computed * 127.5 + 127.5).astype(np.uint8)

            # write to steam
            video_out.write(computed)
        
        # close stream
        video_out.release()

if __name__ == '__main__':
    vid = Video_Compute('checkpoints\\bos\\epoch_100.pt')
    vid.read('D:\\BOS\\Sample Data\\P8100004.MOV')
    vid.compute('computed.avi', overlap=0.5, start=35)#, colormap=cv2.COLORMAP_BONE)