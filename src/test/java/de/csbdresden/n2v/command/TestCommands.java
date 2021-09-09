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
package de.csbdresden.n2v.command;

import net.imagej.ImageJ;
import net.imagej.modelzoo.ModelZooArchive;
import net.imglib2.Cursor;
import net.imglib2.FinalDimensions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;
import org.junit.Test;
import org.scijava.command.CommandModule;
import org.scijava.io.location.FileLocation;

import java.io.File;
import java.util.Random;
import java.util.concurrent.ExecutionException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class TestCommands {

	@Test
	public void testTrainPredict2D() throws ExecutionException, InterruptedException {
		ImageJ ij = new ImageJ();
		ij.ui().setHeadless(true);
		Img<FloatType> input = ij.op().create().img(new FinalDimensions(200, 200), new FloatType());
		Random random = new Random();
		input.forEach(pix -> pix.set(random.nextFloat()));

		CommandModule res = ij.command().run(N2VTrainPredictCommand.class, false,
				"training", input,
				"prediction", input,
				"numEpochs", 2,
				"numStepsPerEpoch", 3,
				"batchSize", 5,
				"patchShape", 32,
				"neighborhoodRadius", 2).get();

		ModelZooArchive latestExport = (ModelZooArchive) res.getOutput("latestTrainedModel");
		ModelZooArchive bestExport = (ModelZooArchive) res.getOutput("bestTrainedModel");
		RandomAccessibleInterval output = (RandomAccessibleInterval) res.getOutput("output");

		assertNotNull(latestExport);
		assertTrue(new File(latestExport.getLocation().getURI()).exists());
		assertNotNull(bestExport);
		assertTrue(new File(bestExport.getLocation().getURI()).exists());
		assertNotNull(output);
		assertEquals(2, output.numDimensions());
		assertEquals(input.dimension(0), output.dimension(0));
		assertEquals(input.dimension(1), output.dimension(1));

		ij.context().dispose();
	}

	@Test
	public void testTrainPredictSeparately2D() throws ExecutionException, InterruptedException {
		ImageJ ij = new ImageJ();
		ij.ui().setHeadless(true);
		Img<FloatType> input = ij.op().create().img(new FinalDimensions(200, 200), new FloatType());
		Random random = new Random();
		float[] values = new float[(int) input.size()];
		Cursor<FloatType> cursor = input.cursor();
		for (int i = 0; i < values.length; i++) {
			values[i] = random.nextFloat();
			cursor.next().set(values[i]);
		}

		CommandModule res = ij.command().run(N2VTrainCommand.class, false,
				"training", input,
				"validation", input,
				"numEpochs", 2,
				"numStepsPerEpoch", 3,
				"batchSize", 5,
				"patchShape", 32,
				"neighborhoodRadius", 2).get();

		ModelZooArchive latestExport = (ModelZooArchive) res.getOutput("latestTrainedModel");
		ModelZooArchive bestExport = (ModelZooArchive) res.getOutput("bestTrainedModel");

		assertNotNull(latestExport);
		File latestExportFile = new File(latestExport.getLocation().getURI());
		assertTrue(latestExportFile.exists());
		assertNotNull(bestExport);
		assertTrue(new File(bestExport.getLocation().getURI()).exists());

		res = ij.command().run(N2VPredictCommand.class, false,
				"input", input,
				"modelFile", latestExportFile).get();

		RandomAccessibleInterval output = (RandomAccessibleInterval) res.getOutput("output");
		assertNotNull(output);
		cursor = input.cursor();
		for (int i = 0; i < values.length; i++) {
			assertEquals(cursor.next().getRealFloat(), values[i], 0.0001);
		}
		assertEquals(2, output.numDimensions());
		assertEquals(input.dimension(0), output.dimension(0));
		assertEquals(input.dimension(1), output.dimension(1));

		latestExportFile.delete();
		((FileLocation)bestExport.getLocation()).getFile().delete();

		ij.context().dispose();
	}

}
