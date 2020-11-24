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
package de.csbdresden.n2v.interactive;

import net.imagej.ImageJ;
import net.imagej.modelzoo.ModelZooArchive;
import net.imagej.modelzoo.ModelZooService;
import net.imagej.modelzoo.consumer.model.prediction.PredictionOutput;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import org.junit.After;
import org.junit.Test;
import org.scijava.log.LogLevel;

import java.util.Map;

public class InteractivePrediction {
	private ImageJ ij;

	@After
	public void tearDown() {
		ij.context().dispose();
	}

	@Test
	public void run() throws Exception {

		ij = new ImageJ();
		ij.launch();
//		ij.log().setLevel(LogLevel.DEBUG);

		// resource paths
		String modelPath = getClass().getResource("/format-0.2.0-csbdeep.bioimage.io.zip").getPath();
		String imgPath = getClass().getResource("/blobs.tif").getPath();

		ModelZooArchive model = ij.get(ModelZooService.class).io().open(modelPath);
		ij.ui().show(model);

		Img img = (Img) ij.io().open(imgPath);

		PredictionOutput outputs = ij.get(ModelZooService.class).predict(model, img, "XYB");
		outputs.asMap().forEach((name, output) -> {
			ij.ui().show(name, output);
		});

	}

	public static void main(String... args) throws Exception {
		new InteractivePrediction().run();
	}
}
