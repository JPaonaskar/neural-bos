'''
TRAIN

Train Image-to-Image GAN on dataset

Referances:

'''
import utils
import pix2pix
#import dataset
import synth_gen

train = synth_gen.BOS_Dataset('datasets\\bos\\train', clamped=True)
val = synth_gen.BOS_Dataset('datasets\\bos\\val', clamped=True)
#train = dataset.I2I_Dataset('datasets\\maps\\train')
#val = dataset.I2I_Dataset('datasets\\maps\\val')

i2i = pix2pix.Pix2PixModel()

i2i.learn(train, epochs=300, checkpoints=10, last_checkpoint='checkpoints\\bos\\epoch-200.pt')
utils.plot_loss(i2i.history)

x, y, pred = i2i.predict(val, batch_size=25)
utils.plot_images(x, title='Input', show=False)
utils.plot_images(y, title='Target', show=False)
utils.plot_images(pred, title='Prediction')