'''
TRAIN

Train Image-to-Image GAN on dataset

Referances:

'''
import torch

import utils
import models
import dataset

train = dataset.I2I_Dataset('datasets\\bos\\train') #'datasets\\maps\\train')
#val = dataset.I2I_Dataset() #'datasets\\maps\\val')

i2i = models.Image2Image(torch.device('cuda'))

i2i.train(train, epochs=20)
utils.plot_loss(i2i.history)

x, y, pred = i2i.predict(train, batch_size=25)
utils.plot_images(x, title='Input', show=False)
utils.plot_images(y, title='Target', show=False)
utils.plot_images(pred, title='Prediction')