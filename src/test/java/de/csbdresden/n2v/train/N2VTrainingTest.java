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
package de.csbdresden.n2v.train;

import net.imagej.ImageJ;
import net.imglib2.FinalDimensions;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;
import org.junit.Test;

import java.util.Random;
import java.util.concurrent.ExecutionException;

public class N2VTrainingTest {

	@Test
	public void testTrainingValidationBatches2D() throws ExecutionException {

		ImageJ ij = new ImageJ();
		ij.ui().setHeadless(true);
		Img<FloatType> trainingBatches = ij.op().create().img(new FinalDimensions(32, 32, 128), new FloatType());
		Random random = new Random();
		trainingBatches.forEach(pix -> pix.set(random.nextFloat()));
		Img<FloatType> validationBatches = ij.op().create().img(new FinalDimensions(32, 32, 4), new FloatType());
		validationBatches.forEach(pix -> pix.set(random.nextFloat()));

		long batchSize = trainingBatches.dimension(2);

//		for (int i = 0; i < 10; i++) {
			N2VTraining n2v = new N2VTraining(ij.context());
			n2v.init(new N2VConfig()
					.setTrainDimensions(2)
					.setNumEpochs(2)
					.setStepsPerEpoch(2)
					.setBatchSize((int)batchSize)
					.setPatchShape(32));
			n2v.input().addTrainingData(trainingBatches);
			n2v.input().addValidationData(validationBatches);
			n2v.train();
			n2v.dispose();
//		}

		ij.context().dispose();
	}

	@Test
	public void testTrainingValidationBatches3D() throws ExecutionException {

		ImageJ ij = new ImageJ();
		ij.ui().setHeadless(true);
		Img<FloatType> trainingBatches = ij.op().create().img(new FinalDimensions(32, 32, 32, 32), new FloatType());
		Random random = new Random();
		trainingBatches.forEach(pix -> pix.set(random.nextFloat()));
		Img<FloatType> validationBatches = ij.op().create().img(new FinalDimensions(32, 32, 32, 2), new FloatType());
		validationBatches.forEach(pix -> pix.set(random.nextFloat()));

		long batchSize = trainingBatches.dimension(3);

		N2VTraining n2v = new N2VTraining(ij.context());
		n2v.init(new N2VConfig()
				.setTrainDimensions(3)
				.setNumEpochs(1)
				.setStepsPerEpoch(2)
				.setBatchSize((int)batchSize)
				.setPatchShape(12));
		n2v.input().addTrainingData(trainingBatches);
		n2v.input().addValidationData(validationBatches);
		n2v.train();

		ij.context().dispose();
	}
}
