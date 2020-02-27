package de.csbdresden.n2v.interactive;

import de.csbdresden.n2v.N2VTraining;
import net.imagej.ImageJ;
import net.imglib2.FinalDimensions;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import java.util.ArrayList;
import java.util.List;

public class InteractiveTraining {
		public static void main(String...args) {
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
			n2v.input().setTrainDimensions(2);
			n2v.input().setNumEpochs(10);
			n2v.input().setStepsPerEpoch(10);
			n2v.input().setBatchSize((int) batchSize);
			n2v.input().setBatchDimLength(32);
			n2v.input().setPatchDimLength(16);
			n2v.init();
			n2v.input().addTrainingData(stack);
			n2v.input().addValidationData(stack);
			n2v.train();
		}
	}
