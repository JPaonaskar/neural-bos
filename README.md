# neural-bos

## Introduction
An attempt to use an Image-to-Image network to create a same resolution schlieren density map

## Updates
Training with Map dataset, understanding of roads can be seen in predictions

| Input Images | Target Images | Predicted Image |
| --- | --- | --- |
| ![Input Images](https://github.com/JPaonaskar/neural-bos/blob/main/figures/Map_GAN_Input.png) | ![Target Images](https://github.com/JPaonaskar/neural-bos/blob/main/figures/Map_GAN_Target.png) | ![Predicted Images](https://github.com/JPaonaskar/neural-bos/blob/main/figures/Map_GAN_Pred.png) |

Training time is large (ish) at 5.56 hours for 800 epochs. Might be worth scaling down model and optimizing?

## Resources
- https://phillipi.github.io/pix2pix/
- https://www.tensorflow.org/tutorials/generative/pix2pix
