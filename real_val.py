'''
REAL VAL

Test model on real data

Referances:

'''
import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

from tqdm import tqdm
from typing import Union

import pix2pix

class Cascade_Dataset(Dataset):
    '''
    Image bos dataset with cascade pairing (without target)

    Args:
        frames (np.ndarray) : list of frames
    '''
    def __init__(self, frames:np.ndarray):
        gray = np.zeros(frames.shape[:-1], dtype=np.float32)

        # convert to grayscale
        print('Loading Frames')
        for i, frame in enumerate(tqdm(frames)):
            gray[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # setup cascade pairing
        self.ref = gray[0]
        self.frames = gray[1:]

    def __len__(self) -> int:
        '''
        Length of dataset

        Args:
            None

        Returns:
            len (int) : length
        '''
        return len(self.frames)

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
        image = np.dstack([self.ref, self.frames[ind], np.zeros_like(self.ref)])

        # min-max
        image = image - image.min()
        image = image / image.max()

        # convert to [-1, 1]
        image = 2 * image - 1

        # covnert to torch tensor
        image = torch.Tensor(image).permute(2, 0, 1)

        # output
        return image, torch.zeros_like(image)


class VideoGUI():
    '''
    GUI interface to test with a video file

    Args:
        video (str) : path to video (default=None)
        device (torch.device) : model device (default=torch.device('cuda'))

    '''
    def __init__(self, video:str=None, device:torch.device=torch.device('cuda')):
        # create model
        self.model = pix2pix.Pix2PixModel(device)

        # store frames
        self.w = None
        self.h = None
        self.frames = []

        # store results
        self.results = []

        # store analysis region
        self.moving = False
        self.window = None

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

        # create centered window
        self.h, self.w, _ = self.frames[0].shape
        self._assign_window(self.w // 2, self.h // 2)

    def generate(self, start:int=0) -> None:
        '''
        Generate results using cascade pairing

        Args:
            start (int) : starting index

        Returns:
            None
        '''
        # create arrays
        frames = np.array(self.frames, dtype=np.uint8)
        self.results = frames.copy()

        # slice and crop
        frames = frames[start:, self.window[1]:self.window[3], self.window[0]:self.window[2], :]

        # create dataset
        dataset = Cascade_Dataset(frames)

        # predict the entire dataset
        _, _, pred = self.model.predict(dataset, batch_size=len(dataset), shuffle=False)

        # convert predictions to numpy
        pred = pred.cpu().detach().permute(0, 2, 3, 1).numpy()

        # convert predictions to image
        pred = (pred * 127.5 + 127.5).astype(np.uint8)

        # paste prediction
        self.results[start+1:, self.window[1]:self.window[3], self.window[0]:self.window[2], :] = pred

    def save(self, filename:str) -> None:
        '''
        Save processed frames

        Args:
            filename (str) : path to save file

        Returns:
            None
        '''

    def _assign_window(self, x:int, y:int) -> None:
        '''
        Assign window location from center coordiantes

        Args:
            x (int) : center x coordinate
            y (int) : center y coordinate

        Return:
            None
        '''
        # get initial point
        wx = x - pix2pix.IMAGE_WIDTH // 2
        wy = y - pix2pix.IMAGE_WIDTH // 2

        # bound
        wx = max(0, min(wx, self.w - pix2pix.IMAGE_WIDTH))
        wy = max(0, min(wy, self.h - pix2pix.IMAGE_HEIGHT))

        # assign window
        self.window = (
            wx,                        # x1
            wy,                        # y1
            wx + pix2pix.IMAGE_WIDTH,  # x2
            wy + pix2pix.IMAGE_HEIGHT, # y2
        )

    def _mouse_window(self, event:int, x:int, y:int, flags:int, param:int) -> None:
        '''
        Mouse events for the window

        Args:
            event (int) : OpenCV event
            x (int) : event x
            y (int) : event y
            flags (int) : event flags
            param (int) : event parameter

        Return:
            None
        '''
        if event == cv2.EVENT_LBUTTONDOWN:
            self.moving = True
            self._assign_window(x, y)
    
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.moving == True:
                self._assign_window(x, y)
    
        elif event == cv2.EVENT_LBUTTONUP:
            self.moving = False
            self._assign_window(x, y)

    def _get_annotated_frame(self, ind:int, font:int=cv2.FONT_HERSHEY_SIMPLEX, font_scale:float=0.5, font_color:tuple[int, int, int]=(255, 255, 255), font_thickness:int=1, font_pad:int=8) -> np.ndarray:
        '''
        Draw annotations on a frame

        Args:
            ind (int) : frame index
            font (int) : overlay font, None displays no text (default=cv2.FONT_HERSHEY_SIMPLEX)
            font_scale (float) : overlay font scale (default=0.5)
            font_color (tuple[int, int, int]) : overlay font color (default=(255, 255, 255))
            font_thickness (int) : overlay font thickness (default=1)
            font_pad (int) : overlay font padding from edges (default=8)

        Returns:
            frame (np.ndarray) : annotated frame
        '''
        # copy frame
        frame = self.frames[ind].copy()

        # draw frame number
        if font != None:
            # get text size
            text = f'{ind+1} / {len(self.frames)}'
            (w, h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

            # put text
            org = (frame.shape[1] - w - font_pad, frame.shape[0] - h - font_pad)
            frame = cv2.putText(frame, text, org, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        # draw window
        p1 = (self.window[0], self.window[1])
        p2 = (self.window[2], self.window[3])
        cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)

        # return frame
        return frame
    
    def _get_predicted_frame(self, ind:int) -> tuple[bool, np.ndarray]:
        '''
        Get predicted image on frame

        Args:
            ind (int) : frame index

        Returns:
            ret (bool) : success
            frame (np.ndarray) : predicted frame
        '''
        if len(self.results) == len(self.frames):
            return True, self.results[ind]
        return False, None
    
    def _key_step(self, k:int, ind:int) -> int:
        '''
        Apply frame stepping

        Args:
            k (int) : key id
            ind (int) : current index

        Returns:
            ind (int) : new index
        '''
        # large stepping
        if k == ord('a'):
            # full step
            if ind >= 10:
                ind -= 10
            # ceil
            else:
                ind = 0
        elif k == ord('d'):
            # full step
            if ind < len(self.frames) - 10:
                ind += 10
            # floor
            else:
                ind = len(self.frames) - 1

        # small stepping
        elif k == ord(','):
            if ind > 0:
                ind -= 1
        elif k == ord('.'):
            if ind < len(self.frames) - 1:
                ind += 1

        # return index
        return ind

    def show(self) -> None:
        '''
        Run GUI

        Args:
            None

        Returns:
            None
        '''
        # placeholders
        ind = 0

        # mouse events
        cv2.namedWindow('Video')
        cv2.setMouseCallback('Video', self._mouse_window)

        # loop
        while True:
            # get frames
            frame = self._get_annotated_frame(ind)
            ret, pred = self._get_predicted_frame(ind)

            # draw frame
            cv2.imshow('Video', frame)

            # draw prediction
            if ret:
                cv2.imshow('Prediction', pred)

            # keys
            k = cv2.waitKey(10)

            # quit
            if k == ord('q'):
                break

            # process
            elif k == ord('c'):
                print('Processing')
                self.generate(ind)

            else:
                # stepping
                ind = self._key_step(k, ind)

        # close window
        cv2.destroyAllWindows()

if __name__ == '__main__':
    gui = VideoGUI('C:\\Users\\jpthe\\Downloads\\1024x2.avi')
    gui.show()