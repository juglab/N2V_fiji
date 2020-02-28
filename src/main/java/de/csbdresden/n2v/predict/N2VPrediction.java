package de.csbdresden.n2v.predict;

import de.csbdresden.csbdeep.commands.GenericNetwork;
import de.csbdresden.n2v.train.ModelSpecification;
import de.csbdresden.n2v.util.N2VUtils;
import net.imagej.Dataset;
import net.imagej.DefaultDataset;
import net.imagej.ImgPlus;
import net.imagej.ops.OpService;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import org.scijava.Context;
import org.scijava.command.CommandModule;
import org.scijava.command.CommandService;
import org.scijava.plugin.Parameter;

import java.io.File;
import java.util.concurrent.ExecutionException;

public class N2VPrediction {

	private File zippedModel;
	private FloatType mean;
	private FloatType stdDev;
	private int trainDimensions;
	private boolean showDialog = false;

	@Parameter
	private OpService opService;

	@Parameter
	private CommandService commandService;

	@Parameter
	private Context context;

	public N2VPrediction(Context context) {
		context.inject(this);
	}

	public void setModelFile(File zippedModel) {
		this.zippedModel = zippedModel;
		ModelSpecification.readConfig(this, zippedModel);
	}

	public void setMean(FloatType mean) {
		this.mean = mean;
	}

	public void setStdDev(FloatType stdDev) {
		this.stdDev = stdDev;
	}

	public void setTrainDimensions(int numDimensions) {
		this.trainDimensions = numDimensions;
	}

	public void setShowDialog(boolean showDialog) {
		this.showDialog = showDialog;
	}

	public RandomAccessibleInterval predict(RandomAccessibleInterval input) {
		Img prediction = preprocess(input, mean, stdDev);
		Dataset inputDataset = new DefaultDataset(context, new ImgPlus(prediction));
		try {
			final CommandModule module = commandService.run(
					GenericNetwork.class, false,
					"input", inputDataset,
					"normalizeInput", false,
					"modelFile", zippedModel.getAbsolutePath(),
					"blockMultiple", 8,
					"nTiles", 8,
					"overlap", 32,
					"showProgressDialog", showDialog).get();
			if(module.isCanceled()) return null;
			RandomAccessibleInterval<FloatType> output = (RandomAccessibleInterval<FloatType>) module.getOutput("output");
			if(output == null) return null;
			postprocess(output, mean, stdDev);
			return output;
		} catch (InterruptedException | ExecutionException e) {
			e.printStackTrace();
		}
		return null;
	}

	private void postprocess(RandomAccessibleInterval<FloatType> output, FloatType mean, FloatType stdDev) {
		N2VUtils.denormalizeInplace(output, mean, stdDev, opService);
	}

	public Img preprocess(RandomAccessibleInterval input, FloatType mean, FloatType stdDev) {
		return (Img) N2VUtils.normalize(input, mean, stdDev, opService);
	}

	public RandomAccessibleInterval<FloatType> predictPadded(RandomAccessibleInterval<FloatType> input) {
		int padding = 32;
		FinalInterval bigger = new FinalInterval(input);
		for (int i = 0; i < trainDimensions; i++) {
			bigger = Intervals.expand(bigger, padding, i);
		}
		IntervalView<FloatType> paddedInput = Views.zeroMin(Views.interval(Views.extendZero(input), bigger));
		RandomAccessibleInterval<FloatType> output = predict(paddedInput);
		if(output == null) return null;
		FinalInterval smaller = new FinalInterval(output);
		for (int i = 0; i < trainDimensions; i++) {
			smaller = Intervals.expand(smaller, -padding, i);
		}
		return Views.zeroMin(Views.interval(output, smaller));
	}
}
