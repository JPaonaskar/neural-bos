'''
UTILS

Utilities

Referances:

'''
import os

import torch
import numpy as np

import cv2
import matplotlib.pyplot as plt

def plot_images(data:torch.Tensor, title:str='data', rows:int=5, cols:int=5, show:bool=True) -> None:
    '''
    Plot a some images

    Args:
        data (torch.Tensor) : data to plot
        title (str) : figure title (default='data')
        rows (int) : number of rows (default=5)
        cols (int) : nuber of columns (default=5)
        show (bool) : show figure (default=True)

    Returns:
        None
    '''

    # setup figure
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(title, fontsize=16)

    # populate
    for i in range(rows * cols):
        # create plot
        plt.subplot(rows, cols, i+1)

        # remove grid
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        # get copy of image
        img = data[i].permute(1, 2, 0).detach().clone()

        # reformat image if needed
        if img.min() < 0:
            img = img.type(torch.float)
            img = img - img.min()
            img = img / img.max()

        # show image
        plt.imshow(img, cmap=plt.cm.binary)

    # show figure
    if show:
        plt.show()

def plot_bos_images(data:torch.Tensor) -> None:
    '''
    Plot a some images

    Args:
        data (torch.Tensor) : data to plot

    Returns:
        None
    '''
    # setup window
    cv2.namedWindow('BOS Images')

    # populate
    ind = 0
    channel = 0
    while True:
        # pull and convert
        if torch.is_tensor(data[ind]):
            img = data[ind][channel, :, :].numpy()
        else:
            img = data[ind][:, :, channel]
        img = (img * 0.5 + 0.5)

        # show
        cv2.imshow('BOS Images', img)
        k = cv2.waitKey(0)

        # quit
        if k == ord('q'):
            break
        
        # switch between images
        elif (k == ord('a')) and ind > 0:
            ind -= 1

        elif (k == ord('d')) and ind < len(data) - 1:
            ind += 1

        # toggle betweem background and displaced
        elif k == ord('w') and channel < 2:
            channel += 1

        elif k == ord('s') and channel > 0:
            channel -= 1


def plot_loss(losses:dict, show:bool=True) -> None:
    '''
    Plot losses

    Args:
        losses (dict) : dictionary of losses and their labels
        show (bool) : show figure (default=True)

    Returns:
        None
    '''
    plt.figure()

    # plot each loss
    for label, loss in losses.items():
        x = np.arange(len(loss))

        # plot
        plt.plot(x, loss, label=label)

    # style
    plt.xlabel('Itteration')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    # show figure
    if show:
        plt.show()

def save_sample(x:torch.Tensor, y:torch.Tensor, pred:torch.Tensor, dirname:str='samples', name:str='sample') -> None:
    '''
    Save a sample of data

    Args:
        x (torch.Tensor) : input image
        y (torch.Tensor) : target image
        pred (torch.Tensor) : predicted image
        dirname (str) : sample image save directory
        name (str) : name of sample image

    Returns:
        None
    '''
    # create path
    path = os.path.abspath(dirname)
    path = os.path.join(path, f'{name}.png')

    # convert to numpy
    x = x.permute(0, 2, 3, 1).detach().clone().numpy()
    y = y.permute(0, 2, 3, 1).detach().clone().numpy()
    pred = pred.permute(0, 2, 3, 1).detach().clone().numpy()

    # plot and save
    fig = plt.figure(figsize=(4, 6))

    for i in range(pred.shape[0]):
        if (i < 8):
            # stack
            x_i = x[i, :, :, :]
            y_i = y[i, :, :, :]
            pred_i = pred[i, :, :, :]

            image = np.hstack([x_i, y_i, pred_i])

            # plot
            plt.subplot(4, 2, i+1)
            plt.imshow(image * 0.5 + 0.5)
            plt.axis('off')

    plt.savefig(path)
    plt.close(fig)