'''
MAPS

Train Image-to-Image on Satillite -> Map dataset

Referances:
    https://medium.com/@Skpd/pix2pix-gan-for-generating-map-given-satellite-images-using-pytorch-6e50c318673a

'''

import pix2pix
import dataset

train = dataset.I2I_Dataset('datasets\\maps\\train')
val = dataset.I2I_Dataset('datasets\\maps\\val')

i2i = pix2pix.Image2Image()

i2i.train(train, epochs=130, load=20, checkpoint=10)
i2i.plot_history()
i2i.plot_predictions(val)