# N2V training in Fiji

## How to use
- Add https://sites.imagej.net/N2V to Fiji
- For GPU support (Linux, Windows): install `CUDA 10.0` and `cuDNN`. In Fiji, open `Edit > Options > TensorFlow...` and install `TF 1.13.1 GPU`
- Try the plugins in `Plugins > CSBDeep > N2V`

## Plugin details 
- `train`: will take an image for training and an image for validation. If you choose the same image for both, 10% of the image will be used for validation, 90% for training. The plugin returns a window displaying the path to the zipped trained model and the normalization parameters (mean and stddev)
- `predict`: takes a zipped trained model, an image and the normalization parameters (mean and stddev), basically normalizes, calls CSBDeep, denormalizes
- `train + predict`: one-click solution: calls the plugins above in a row. Takes a training and a prediction image, if they are the same, 10% of the image will not be used for training, but for validation. Otherwise the prediction image is used for validation.
