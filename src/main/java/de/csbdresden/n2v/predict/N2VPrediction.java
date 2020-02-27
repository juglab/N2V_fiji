package de.csbdresden.n2v.predict;

import de.csbdresden.csbdeep.commands.GenericNetwork;
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
import org.yaml.snakeyaml.Yaml;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.zip.ZipFile;

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
		readConfig();
	}

	private void readConfig() {
		double mean = 0.0f;
		double stdDev = 1.0f;
		int trainDimensions = 2;
		try {
			InputStream stream = extractFile(zippedModel, "config.yaml");
			Yaml yaml = new Yaml();
			Map<String, Object> obj = yaml.load(stream);
			System.out.println(obj);
			Object meanObj = obj.get("mean");
			if(meanObj != null) mean = (double) meanObj;
			Object stdDevObj = obj.get("stdDev");
			if(stdDevObj != null) stdDev = (double) stdDevObj;
			Object trainDimensionsObj = obj.get("trainDimensions");
			if(trainDimensionsObj != null) trainDimensions = (int) trainDimensionsObj;
		} catch (IOException e) {
			e.printStackTrace();
		}
		setMean(new FloatType((float) mean));
		setStdDev(new FloatType((float) stdDev));
		setTrainDimensions(trainDimensions);
	}

	public InputStream extractFile(File zipFile, String fileName) throws IOException {
		ZipFile zf = new ZipFile(zipFile);
		return zf.getInputStream(zf.getEntry(fileName));
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
		Img prediction = (Img) N2VUtils.normalize(input, mean, stdDev, opService);
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
			N2VUtils.denormalizeInplace(output, mean, stdDev, opService);
			return output;
		} catch (InterruptedException | ExecutionException e) {
			e.printStackTrace();
		}
		return null;
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
