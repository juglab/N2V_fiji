/*-
 * #%L
 * N2V plugin
 * %%
 * Copyright (C) 2019 - 2020 Center for Systems Biology Dresden
 * %%
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */
package de.csbdresden.n2v.howto;

import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.ImageJ;
import net.imagej.modelzoo.ModelZooArchive;
import net.imagej.modelzoo.ModelZooService;
import net.imagej.modelzoo.consumer.ModelZooPredictionOptions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import org.scijava.io.IOService;
import org.scijava.log.LogService;
import org.scijava.ui.UIService;

/**
 * How to run prediction headless and with specific options
 */
public class RunPredictionWithOptions {

	public void run() throws Exception {

		// create new ImageJ instance
		ImageJ ij = new ImageJ();
		// launch ImageJ
		ij.launch();
//		ij.log().setLevel(LogLevel.DEBUG);

		// get required services
		UIService ui = ij.ui();
		IOService io = ij.io();
		LogService log = ij.log();
		DatasetService datasetService = ij.dataset();
		ModelZooService modelZooService = ij.get(ModelZooService.class);

		// resource paths
		String modelPath = "/home/random/Documents/2020-06 NEUBIAS/models/n2v-sem.bioimage.io.zip";
		String inputPath = "/media/random/cdc71243-1f7c-49d6-8b22-da102091386f/betaseg/Precise_annotation-1.tif";
		String outputPath = "/media/random/cdc71243-1f7c-49d6-8b22-da102091386f/betaseg/output.tif";

		// open model
		ModelZooArchive model = modelZooService.open(modelPath);
		// uncomment next line to display the model
//		ui.show("model", model);

		// open input image
		Img img = (Img) io.open(inputPath);
		// uncomment next line to display the input image
//		ui.show("input", img);

		// Define the axes of the image.
		// In this case, the third dimension B will be handled as a batch dimension.
		// This means each slice in this dimension can be processed individually.
		String axes = "XYB";

		// create prediction options
		ModelZooPredictionOptions options = ModelZooPredictionOptions.options();

		// If the inpnut image has a batch dimension ("B" in the axes string), this option determines
		// how how many slices per batch should be processed at once.
		// Choosing "1" is the most memory friendly option.
		options.batchSize(1);

		// Each batch (or, if no batch dimension exists, the whole image) can be split into tiles.
		// this option determines the total number of tiles an image or batch should be split into.
		// Increase this value if you get Out Of Memory errors.
		// Increasing this value will increase computation time since an overlap between each tile has to be computed.
		options.numberOfTiles(1);

		// Uncomment next line if you want to use a specific directory for caching the prediction result.
		// By default, a temporary directory is created on your system.
		// If you adjust this, you have to handle cleaning up yourself,
		// e.g. deleting all files in the cache folder after prediction is done.
//		options.cacheDirectory("/disk/with/space/my_cache");

		// Run prediction on input image.
		RandomAccessibleInterval output = modelZooService.predict(model, img, axes, options);

		// Uncomment next line to display the output image.
//		ui.show("output", output);

		log.info("Saving result..");

		Dataset dataset = datasetService.create(output);

		io.save(dataset, outputPath);

		log.info("Saving done.");

	}

	public static void main(String... args) throws Exception {
		new RunPredictionWithOptions().run();
	}
}
