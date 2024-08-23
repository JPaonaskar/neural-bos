# neural-bos

## Introduction
An attempt to use an Image-to-Image network to create a same resolution schlieren density map

## Updates
Strong similarities between target and results with synthetic BOS data in only 20 epochs!

![BOS GAN Training](https://github.com/JPaonaskar/neural-bos/blob/main/figures/BOS_GAN_0-20.png)

|     | Input Images | Target Images | Predicted Image |
| --- | --- | --- | --- |
| BOS | ![Input Images](https://github.com/JPaonaskar/neural-bos/blob/main/figures/BOS_GAN_Input.png) | ![Target Images](https://github.com/JPaonaskar/neural-bos/blob/main/figures/BOS_GAN_Target.png) | ![Predicted Images](https://github.com/JPaonaskar/neural-bos/blob/main/figures/BOS_GAN_Pred.png) |
| MAP | ![Input Images](https://github.com/JPaonaskar/neural-bos/blob/main/figures/Map_GAN_Input.png) | ![Target Images](https://github.com/JPaonaskar/neural-bos/blob/main/figures/Map_GAN_Target.png) | ![Predicted Images](https://github.com/JPaonaskar/neural-bos/blob/main/figures/Map_GAN_Pred.png) |

## Resources
- https://phillipi.github.io/pix2pix/
- https://www.tensorflow.org/tutorials/generative/pix2pix
