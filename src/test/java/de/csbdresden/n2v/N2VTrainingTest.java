package de.csbdresden.n2v;

import net.imagej.ImageJ;
import net.imglib2.FinalDimensions;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;
import org.junit.Test;

import java.util.Random;

public class N2VTrainingTest {

	@Test
	public void testTrainingValidationBatches2D() {

		ImageJ ij = new ImageJ();
		ij.ui().setHeadless(true);
		Img<FloatType> trainingBatches = ij.op().create().img(new FinalDimensions(32, 32, 128), new FloatType());
		Random random = new Random();
		trainingBatches.forEach(pix -> pix.set(random.nextFloat()));
		Img<FloatType> validationBatches = ij.op().create().img(new FinalDimensions(32, 32, 4), new FloatType());
		validationBatches.forEach(pix -> pix.set(random.nextFloat()));

		long batchSize = trainingBatches.dimension(2);

		N2VTraining n2v = new N2VTraining(ij.context());
		n2v.init();
		n2v.setTrainDimensions(2);
		n2v.setNumEpochs(2);
		n2v.setStepsPerEpoch(2);
		n2v.setBatchSize((int) batchSize);
		n2v.setBatchDimLength(32);
		n2v.setPatchDimLength(16);
		n2v.addTrainingData(trainingBatches);
		n2v.addValidationData(validationBatches);
		n2v.train();
	}

	@Test
	public void testTrainingValidationBatches3D() {

		ImageJ ij = new ImageJ();
		ij.ui().setHeadless(true);
		Img<FloatType> trainingBatches = ij.op().create().img(new FinalDimensions(32, 32, 32, 128), new FloatType());
		Random random = new Random();
		trainingBatches.forEach(pix -> pix.set(random.nextFloat()));
		Img<FloatType> validationBatches = ij.op().create().img(new FinalDimensions(32, 32, 32, 4), new FloatType());
		validationBatches.forEach(pix -> pix.set(random.nextFloat()));

		long batchSize = trainingBatches.dimension(3);

		N2VTraining n2v = new N2VTraining(ij.context());
		n2v.init();
		n2v.setTrainDimensions(3);
		n2v.setNumEpochs(2);
		n2v.setStepsPerEpoch(2);
		n2v.setBatchSize((int) batchSize);
		n2v.setBatchDimLength(32);
		n2v.setPatchDimLength(16);
		n2v.addTrainingData(trainingBatches);
		n2v.addValidationData(validationBatches);
		n2v.train();
	}
}
