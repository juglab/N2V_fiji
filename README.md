[![](https://travis-ci.com/juglab/N2V_fiji.svg?branch=master)](https://travis-ci.com/juglab/N2V_fiji)

# N2V training in Fiji
Noise reduction (pixel wise independent) by training a CNN on single noisy images in Java.  

## How to use
- Add this update site to Fiji: https://sites.imagej.net/N2V
- For GPU support (Linux, Windows): install `CUDA 10.0` and `cuDNN`. In Fiji, open `Edit > Options > TensorFlow...` and install `TF 1.13.1 GPU`
- Try the plugins in `Plugins > CSBDeep > N2V`

## Plugin details 
- **`train`**: will take an image for training and an image for validation. If you choose the same image for both, 10% of the image will be used for validation, 90% for training. The plugin returns a window displaying the path to the zipped trained model from the last step and the zipped trained model with the lowest validation loss
- **`predict`**: takes a zipped trained model and an image; basically normalizes, calls CSBDeep, denormalizes
- **`train + predict`**: one-lick solution for content aware denoising. Takes a training and a prediction image, if they are the same, 10% of the image will not be used for training, but for validation. Otherwise the prediction image is used for validation.

## Credits
N2V for Fiji is written by Deborah Schmidt and Gabriella Turek.

The code is adapted from and based on N2V by Alexander Krull, Tim-Oliver Buchholz and Florian Jug:

https://arxiv.org/abs/1811.10980

Please cite this paper if you use N2V for Fiji for your research.
