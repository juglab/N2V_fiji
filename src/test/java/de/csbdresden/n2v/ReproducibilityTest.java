package de.csbdresden.n2v;

import net.imagej.ImageJ;
import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.scijava.log.LogLevel;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ReproducibilityTest<T extends RealType<T>> {

	private List<Pair<Img, Img>> testData;
	private ImageJ ij;
	private double minMaxDif;

	private void run(String[] args) throws IOException {
		ij = new ImageJ();
		ij.launch(args);
		System.out.println(ij.app().getApp().getBaseDirectory());
		Img trainImg = (Img) ij.io().open("/home/random/Development/python/n2v/examples/2D/denoising2D_BSD68/data/BSD68_reproducibility_data/train/DCNN400_train_gaussian25.tif");
		Img validateImg = (Img) ij.io().open("/home/random/Development/python/n2v/examples/2D/denoising2D_BSD68/data/BSD68_reproducibility_data/val/DCNN400_validation_gaussian25.tif");
		testData = new ArrayList<>();
		File dir = new File("/home/random/Development/python/n2v/examples/2D/denoising2D_BSD68/data/BSD68_reproducibility_data/test/data/");
		int i = 0;
		List<Img<T>> predictionData = new ArrayList<>();
		for (File file : dir.listFiles()) {
//			if(i == 3) break;
			if(file.getName().endsWith("_img.tif")) {
				Img img = (Img) ij.io().open(file.getAbsolutePath());
				Img gt = (Img) ij.io().open(file.getAbsolutePath().replace("_img.tif", "_gt.tif"));
				testData.add(new ImmutablePair<>(img, gt));
				predictionData.add(gt);
			}
			i++;
		}
		net.imglib2.util.Pair<T, T> minMax = ij.op().stats().minMax(Views.iterable(Views.stack(predictionData)));
		T dif = minMax.getB().copy();
		dif.sub(minMax.getA());
		minMaxDif = dif.getRealDouble();
		System.out.println("Min: " + minMax.getA().getRealDouble() + " max: " + minMax.getB().getRealDouble() + " difference: " + minMaxDif);
		predictionData.clear();

		N2VTraining training = new N2VTraining(ij.context());
		training.setTrainDimensions(2);
		training.setNumEpochs(200);
		training.setStepsPerEpoch(400);
		training.setBatchSize(96);
		training.setBatchDimLength(180);
		training.setPatchDimLength(64);
		training.setNeighborhoodRadius(2);
		training.init();
		training.addTrainingData(trainImg);
		training.addValidationData(validateImg);
		training.addCallbackOnEpochDone(this::calculatePSNR);

		training.train();

		ij.context().dispose();

	}

	private void calculatePSNR(N2VTraining training) {
		N2VPrediction prediction = new N2VPrediction(ij.context());
		prediction.setMean(training.getMean());
		prediction.setStdDev(training.getStdDev());
		try {
			prediction.setModelFile(training.exportLatestTrainedModel());
		} catch (IOException e) {
			e.printStackTrace();
		}
		float sumPSNR = 0;
		int currentLevel = ij.log().getLevel();
		ij.log().setLevel(LogLevel.NONE);
		for (Pair<Img, Img> pair : testData) {
			Img<T> networkInput = pair.getLeft();
			RandomAccessibleInterval<T> output = prediction.predict(Views.zeroMin(Views.interval(Views.extendZero(networkInput), Intervals.expand( networkInput, 32 ))));
//			System.out.println("mean gt   : " + ij.op().stats().mean(pair.getRight()).getRealDouble());
//			System.out.println("stdDev gt : " + ij.op().stats().stdDev(pair.getRight()));
//			System.out.println("mean out  : " + ij.op().stats().mean(Views.iterable(output)));
//			System.out.println("stdDev out: " + ij.op().stats().stdDev(Views.iterable(output)));
			output = Views.zeroMin(Views.interval(output, Intervals.expand(output, -32)));
			sumPSNR += calculatePSNR(pair.getRight(), output);
		}
		ij.log().setLevel(currentLevel);
		System.out.println("PSNR: " + sumPSNR/(double)testData.size());
	}

	private double calculatePSNR(RandomAccessibleInterval gt, RandomAccessibleInterval output) {
		RandomAccess<FloatType> outputRa = output.randomAccess();
		Cursor<FloatType> gtCursor = Views.iterable(gt).localizingCursor();
		long numPix = 0;
		float sumSquareDif = 0;
		while(gtCursor.hasNext()) {
			gtCursor.next();
			outputRa.setPosition(gtCursor);
			sumSquareDif += Math.pow(gtCursor.get().getRealFloat() - outputRa.get().getRealFloat(), 2);
			numPix++;
		}
		double mse = sumSquareDif / (double) numPix;
//		System.out.println("mse: " + mse);
		return 20.*Math.log10(minMaxDif) - 10*Math.log10(mse);
	}

	public static void main(String...args) throws IOException {
		new ReproducibilityTest().run(args);
	}
}
