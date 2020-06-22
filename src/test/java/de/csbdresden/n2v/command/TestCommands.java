package de.csbdresden.n2v.command;

import net.imagej.ImageJ;
import net.imagej.modelzoo.ModelZooArchive;
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
		assertTrue(new File(latestExport.getSource().getURI()).exists());
		assertNotNull(bestExport);
		assertTrue(new File(bestExport.getSource().getURI()).exists());
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
		input.forEach(pix -> pix.set(random.nextFloat()));

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
		File latestExportFile = new File(latestExport.getSource().getURI());
		assertTrue(latestExportFile.exists());
		assertNotNull(bestExport);
		assertTrue(new File(bestExport.getSource().getURI()).exists());

		res = ij.command().run(N2VPredictCommand.class, false,
				"input", input,
				"modelFile", latestExportFile).get();

		RandomAccessibleInterval output = (RandomAccessibleInterval) res.getOutput("output");
		assertNotNull(output);
		assertEquals(2, output.numDimensions());
		assertEquals(input.dimension(0), output.dimension(0));
		assertEquals(input.dimension(1), output.dimension(1));

		ij.context().dispose();
	}

}
