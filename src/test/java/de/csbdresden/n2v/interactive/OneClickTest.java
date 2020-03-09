package de.csbdresden.n2v.interactive;

import de.csbdresden.n2v.command.N2VTrainPredictCommand;
import net.imagej.ImageJ;
import net.imglib2.FinalDimensions;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;

import java.util.Random;
import java.util.concurrent.ExecutionException;

public class OneClickTest {

	public void testTrainingAndPrediction() throws ExecutionException, InterruptedException {

		ImageJ ij = new ImageJ();
		ij.ui().setHeadless(true);
		Img<FloatType> trainingBatches = ij.op().create().img(new FinalDimensions(128, 128, 128), new FloatType());
		Random random = new Random();
		trainingBatches.forEach(pix -> pix.set(random.nextFloat()));
		Img<FloatType> predictionBatches = ij.op().create().img(new FinalDimensions(128, 128, 4), new FloatType());
		predictionBatches.forEach(pix -> pix.set(random.nextFloat()));

		for (int i = 0; i < 100; i++) {
			ij.command().run( N2VTrainPredictCommand.class, true,
					"training", trainingBatches,
					"prediction", predictionBatches,
					"numStepsPerEpoch", 20,
					"numEpochs", 20,
					"mode3D", false,
					"batchSize", 128,
					"batchDimLength", 128,
					"patchDimLength", 64).get();
		}
	}
}
