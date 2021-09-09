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
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

public class InteractiveTrainingRestart {
		public static void main(String...args) throws ExecutionException {
			ImageJ ij = new ImageJ();
			ij.launch();
			Img<FloatType> blackImg = ij.op().create().img(new FinalDimensions(32, 32), new FloatType());
			Img<FloatType> whiteImg = ij.op().create().img(new FinalDimensions(32, 32), new FloatType());
			RandomAccess<FloatType> ra = whiteImg.randomAccess();
			for (int i = 0; i < whiteImg.dimension(0)/2; i++) {
				for (int j = 0; j < whiteImg.dimension(1); j++) {
	//				for (int k = 0; k < whiteImg.dimension(2); k++) {
	//					ra.setPosition(new long[]{i, j, k});
						ra.setPosition(new long[]{i, j});
						ra.get().setOne();
	//				}
				}
			}
			List<RandomAccessibleInterval<FloatType>> batch = new ArrayList<>();
			for (int i = 0; i < 10; i++) {
	//			batch.add(blackImg);
				batch.add(whiteImg);
			}

			long batchSize = batch.size();
			RandomAccessibleInterval<FloatType> stack = Views.stack(batch);
			ij.ui().show("stack", stack);

			N2VTraining n2v = new N2VTraining(ij.context());
			n2v.init("/home/random/Development/imagej/project/CSBDeep/n2v-trained-model3853066279801947474.zip",
					new N2VConfig()
							.setTrainDimensions(2)
							.setNumEpochs(10)
							.setStepsPerEpoch(10)
							.setBatchSize((int)batchSize)
							.setPatchShape(32)
							.setPatchShape(16));
			n2v.input().addTrainingData(stack);
			n2v.input().addValidationData(stack);
			n2v.train();
		}
	}
