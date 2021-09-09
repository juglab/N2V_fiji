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

import de.csbdresden.n2v.train.N2VConfig;
import de.csbdresden.n2v.train.N2VTraining;
import net.imagej.ImageJ;
import net.imglib2.FinalDimensions;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.integer.IntType;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.ExecutionException;

public class TrainOnNoise {

	public static void main(String...args) throws IOException, ExecutionException {

		ImageJ ij = new ImageJ();
		ij.ui().setHeadless(true);
		Img<IntType> trainingBatches = ij.op().create().img(new FinalDimensions(180, 180, 10024), new IntType());
		Random random = new Random();
		trainingBatches.forEach(pix -> pix.set(random.nextInt(255)));
		Img<IntType> validationBatches = ij.op().create().img(new FinalDimensions(180, 180, 4), new IntType());
		validationBatches.forEach(pix -> pix.set(random.nextInt(255)));

		N2VTraining n2v = new N2VTraining(ij.context());
		n2v.init(new N2VConfig()
				.setTrainDimensions(2)
				.setNumEpochs(100)
				.setStepsPerEpoch(200)
				.setBatchSize(64)
				.setPatchShape(180)
				.setPatchShape(60));
		n2v.input().addTrainingData(ij.op().convert().float32(trainingBatches));
		n2v.input().addValidationData(ij.op().convert().float32(validationBatches));
		n2v.train();
		File latestModel = n2v.output().exportLatestTrainedModel();
		System.out.println("latest: " + latestModel);
		n2v.dispose();
	}

}
