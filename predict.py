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
        scale (float) : displacement scale (default=10.0)
        device (torch.device) : model device (default=torch.device('cuda'))
    '''
    def __init__(self, checkpoint:str, video:str=None, scale:float=10.0, device:torch.device=torch.device('cuda')):
        # create model
        self.model = pix2pix.Pix2PixModel(device)
        self.model.load_checkpoint(checkpoint)

        # store frames
        self.frames = []

        # scaling
        self.scale = scale

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

    def compute(self, start:int=0, stop:int=None, step:int=1) -> None:
        '''
        Compute displacements with cascade pairing

        Args:
            start (int) : starting index
            stop (int) : starting index
            step (int) : starting index

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
        rows = np.ceil(h / pix2pix.IMAGE_HEIGHT).astype(int)
        cols = np.ceil(w / pix2pix.IMAGE_WIDTH).astype(int)

        images = np.zeros((
            len(frames),
            rows * pix2pix.IMAGE_HEIGHT,
            cols * pix2pix.IMAGE_WIDTH,
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
        
        # subdivide into image set
        print(f'Predicting {rows} rows and {cols} cols')
        for row in tqdm(range(rows), desc='Col'):
            for col in tqdm(range(cols), desc='Row', leave=False):
                y1 = row * pix2pix.IMAGE_HEIGHT
                x1 = col * pix2pix.IMAGE_WIDTH
                y2 = y1 + pix2pix.IMAGE_HEIGHT
                x2 = x1 + pix2pix.IMAGE_WIDTH

                # create as dataset
                dataset = List_Dataset(images[:, y1:y2, x1:x2, :])

                # predict
                _, _, pred = self.model.predict(dataset, shuffle=False, single_batch=False)
                pred = pred.permute(0, 2, 3, 1).cpu().numpy()

                # store
                images[:, y1:y2, x1:x2, :] = pred * self.scale

        # crop and save
        self.computed = images[:, :h, :w, :].astype(np.float16)

    def save(self, filename:str, fps:int=30) -> None:
        '''
        Save computed data to numpy file

        Args:
            filename (str) : path to save to
            fps (int) : frames per second

        Returns:
            None
        '''
        filename = os.path.abspath(filename)

        # get image shape
        size = (self.computed.shape[2], self.computed.shape[1])
        
        # write video
        if os.path.splitext(filename)[1] == '.avi':
            # open video writer
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            video_out = cv2.VideoWriter(filename, fourcc, fps, size)

            # write video
            print("Writting Video")
            for frame in tqdm(self.computed):
                video_out.write((frame * 127.5 / self.scale + 127.5).astype(np.uint8))

            # release writer
            video_out.release()
        
        # write gif
        if os.path.splitext(filename)[1] == '.gif':
            print("Writting GIF")
            with imageio.get_writer(filename, mode='I') as writer:
                for frame in tqdm(self.computed):
                    writer.append_data((frame * 127.5 / self.scale + 127.5).astype(np.uint8))

if __name__ == '__main__':
    vid = Video_Compute('checkpoints\\bos\\epoch_300.pt', scale=10.0)
    vid.read('D:\\BOS\\Sample Data\\P8100004.MOV')
    vid.compute(start=35)
    vid.save('computed.gif')