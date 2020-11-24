#@IOService io
#@UIService ui
#@LogService log
#@DatasetService datasetService
#@ModelZooService modelZooService

from net.imagej.modelzoo.consumer import ModelZooPredictionOptions

# resource paths
modelPath = "/PATH/TO/MODEL.bioimage.io.model.zip"
inputPath = "/PATH/TO/INPUT.tif"
outputPath = "/PATH/TO/OUTPUT.tif"

# open model
model = modelZooService.io().open(modelPath)
# uncomment next line to display the model
# ui.show("model", model)

# open input image
img = io.open(inputPath)
# uncomment next line to display the input image
# ui.show("input", img)

# Define the axes of the image.
# In this case, the third dimension b will be handled as a batch dimension.
# This means each slice in this dimension can be processed individually.
axes = "xyb"

# create prediction options
options = ModelZooPredictionOptions.options()

# If the inpnut image has a batch dimension ("B" in the axes string), this option determines
# how many slices per batch should be processed at once.
# Choosing "1" is the most memory friendly option.
options.batchSize(1)

# Each batch (or, if no batch dimension exists, the whole image) can be split into tiles.
# this option determines the total number of tiles an image or batch should be split into.
# Increase this value if you get Out Of Memory errors.
# Increasing this value will increase computation time since an overlap between each tile has to be computed.
options.numberOfTiles(1)

# Uncomment next line if you want to use a specific directory for caching the prediction result.
# By default, a temporary directory is created on your system.
# If you adjust this, you have to handle cleaning up yourself,
# e.g. deleting all files in the cache folder after prediction is done.
# options.cacheDirectory("/disk/with/space/my_cache")

# Run prediction on input image.
output = modelZooService.predict(model, img, axes, options)

# Uncomment next line to display the output image.
# ui.show("output", output)

log.info("Saving result..")

# Saving result to disk
dataset = datasetService.create(output)
io.save(dataset, outputPath)

log.info("Saving done.")
