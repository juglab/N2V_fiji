package de.csbdresden.n2v.command;

import net.imagej.ImageJ;
import net.imglib2.FinalDimensions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;
import org.junit.Test;
import org.scijava.command.CommandModule;

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
		Img<FloatType> input = ij.op().create().img(new FinalDimensions(100, 100), new FloatType());
		Random random = new Random();
		input.forEach(pix -> pix.set(random.nextFloat()));

		CommandModule res = ij.command().run(N2VTrainPredictCommand.class, false,
				"training", input,
				"prediction", input,
				"numEpochs", 2,
				"numStepsPerEpoch", 3,
				"batchSize", 5,
				"batchDimLength", 40,
				"patchDimLength", 32,
				"neighborhoodRadius", 2).get();

		File latestExport = new File((String) res.getOutput("latestTrainedModelPath"));
		File bestExport = new File((String) res.getOutput("bestTrainedModelPath"));
		RandomAccessibleInterval output = (RandomAccessibleInterval) res.getOutput("output");

		assertNotNull(latestExport);
		assertTrue(latestExport.exists());
		assertNotNull(bestExport);
		assertTrue(bestExport.exists());
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
		Img<FloatType> input = ij.op().create().img(new FinalDimensions(100, 100), new FloatType());
		Random random = new Random();
		input.forEach(pix -> pix.set(random.nextFloat()));

		CommandModule res = ij.command().run(N2VTrainCommand.class, false,
				"training", input,
				"validation", input,
				"numEpochs", 2,
				"numStepsPerEpoch", 3,
				"batchSize", 5,
				"batchDimLength", 40,
				"patchDimLength", 32,
				"neighborhoodRadius", 2).get();

		File latestExport = new File((String) res.getOutput("latestTrainedModelPath"));
		File bestExport = new File((String) res.getOutput("bestTrainedModelPath"));

		assertNotNull(latestExport);
		assertTrue(latestExport.exists());
		assertNotNull(bestExport);
		assertTrue(bestExport.exists());

		res = ij.command().run(N2VPredictCommand.class, false,
				"input", input,
				"modelFile", latestExport).get();

		RandomAccessibleInterval output = (RandomAccessibleInterval) res.getOutput("output");
		assertNotNull(output);
		assertEquals(2, output.numDimensions());
		assertEquals(input.dimension(0), output.dimension(0));
		assertEquals(input.dimension(1), output.dimension(1));

		ij.context().dispose();
	}

}
