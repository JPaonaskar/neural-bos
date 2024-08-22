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

def save_checkpoint(model, optimizer, filename:str='checkpoints\\checkpoint.pt') -> None:
    '''
    Save a checkpoint

    Args:
        model () : model to save
        optimizer () : optimizer to save
        filename (str) : save file

    Returns:
        None
    '''
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.abspath(filename))

def load_checkpoint(model, optimizer, learning_rate:float, device, filename:str='checkpoints\\checkpoint.pt'):
    '''
    Load a checkpoint

    Args:
        model () : model to load checkpoint to
        optimizer () : optimizer to load checkpoint to
        learning_rate (float) : current learning rate
        device () : device to send to
        filename (str) : file to load from

    Returns:
        None
    '''
    checkpoint = torch.load(os.path.abspath(filename), map_location=device)

    # assign state
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # set learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate