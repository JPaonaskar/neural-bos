'''
TRAIN

Train Image-to-Image GAN on dataset

Referances:

'''
import torch

import utils
import models
import dataset
import synth_gen

train = synth_gen.BOS_Dataset('datasets\\bos\\train', clamped=True)
#train = dataset.I2I_Dataset('datasets\\maps\\train')

i2i = models.Image2Image(torch.device('cuda'))

i2i.learn(train, epochs=60, checkpoints=5)#, last_checkpoint='checkpoints\epoch_10.pt')
utils.plot_loss(i2i.history)

x, y, pred = i2i.predict(train, batch_size=25)
utils.plot_images(x, title='Input', show=False)
utils.plot_images(y, title='Target', show=False)
utils.plot_images(pred, title='Prediction')