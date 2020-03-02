[![](https://travis-ci.com/juglab/N2V_fiji.svg?branch=master)](https://travis-ci.com/juglab/N2V_fiji)

# N2V training in Fiji
Noise reduction (pixel wise independent) by training a CNN on single noisy images in Java.  

## How to use
- Add this update site to Fiji: https://sites.imagej.net/N2V
- For GPU support (Linux, Windows): install `CUDA 10.0` and `cuDNN`. In Fiji, open `Edit > Options > TensorFlow...` and install `TF 1.13.1 GPU`. Also see OS specific notes below. You can test if everything works by running `Edit > Options > TensorFlow...` again - in the bottom status line it should state that the GPU TF version is active.
- Try the plugins in `Plugins > CSBDeep > N2V`

### GPU support (Linux)
`CUDA` and `cuDNN` need to be added to your system variables in order for Fiji to be able to use them. You can do that by adding lines similar to this to your `.bashrc` or `.zshrc` file:
```
export PATH=/usr/local/cuda/bin${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
```
.. and then launch Fiji from command line. Another possibility is to edit the Fiji launcher to something like this:
```
export PATH=/usr/local/cuda/bin${PATH};export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH};/PATH_TO_APPS/Fiji.app/ImageJ-linux64
```

### GPU support (Windows)
To set the CUDA environment variables in Windows, please follow the steps described on [this page](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-windows) (section 4.4).


## Plugin details 
- **`train`**: will take an image for training and an image for validation. If you choose the same image for both, 10% of the image will be used for validation, 90% for training. The plugin returns a window displaying the path to the zipped trained model from the last step and the zipped trained model with the lowest validation loss
- **`predict`**: takes a zipped trained model and an image; basically normalizes, calls CSBDeep, denormalizes
- **`train + predict`**: one-click solution for content aware denoising. Takes a training and a prediction image, if they are the same, 10% of the image will not be used for training, but for validation. Otherwise the prediction image is used for validation.

## Credits
N2V for Fiji is written by Deborah Schmidt and Gabriella Turek.

The code is adapted from and based on N2V by Alexander Krull, Tim-Oliver Buchholz and Florian Jug:

https://arxiv.org/abs/1811.10980

Please cite this paper if you use N2V for Fiji for your research.
