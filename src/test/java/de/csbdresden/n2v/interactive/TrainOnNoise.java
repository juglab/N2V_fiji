package de.csbdresden.n2v.interactive;

import de.csbdresden.n2v.N2VConfig;
import de.csbdresden.n2v.N2VTraining;
import net.imagej.ImageJ;
import net.imglib2.FinalDimensions;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.integer.IntType;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class TrainOnNoise {

	public static void main(String...args) throws IOException {

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
				.setPatchDimLength(180)
				.setPatchDimLength(60));
		n2v.input().addTrainingData(ij.op().convert().float32(trainingBatches));
		n2v.input().addValidationData(ij.op().convert().float32(validationBatches));
		n2v.train();
		File latestModel = n2v.output().exportLatestTrainedModel();
		System.out.println("latest: " + latestModel);
		n2v.dispose();
	}

}
