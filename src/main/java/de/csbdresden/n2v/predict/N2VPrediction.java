package de.csbdresden.n2v.predict;

import de.csbdresden.n2v.train.N2VModelSpecification;
import de.csbdresden.n2v.train.TrainUtils;
import io.scif.MissingLibraryException;
import net.imagej.modelzoo.ModelZooArchive;
import net.imagej.modelzoo.consumer.DefaultSingleImagePrediction;
import net.imagej.modelzoo.consumer.ModelZooPrediction;
import net.imagej.modelzoo.consumer.model.InputImageNode;
import net.imagej.modelzoo.consumer.model.ModelZooAxis;
import net.imagej.modelzoo.consumer.model.ModelZooModel;
import net.imagej.ops.OpService;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import org.scijava.Context;
import org.scijava.command.CommandService;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.FileNotFoundException;

@Plugin(type = ModelZooPrediction.class, name = "n2v")
public class N2VPrediction extends DefaultSingleImagePrediction<FloatType, FloatType> {

	private FloatType mean;
	private FloatType stdDev;

	@Parameter
	private OpService opService;

	@Parameter
	private CommandService commandService;

	@Parameter
	private Context context;

	public N2VPrediction(Context context) {
		super(context);
	}

	@Override
	public void setTrainedModel(ModelZooArchive trainedModel) {
		super.setTrainedModel(trainedModel);
		N2VModelSpecification.setFromSpecification(this, trainedModel.getSpecification());
	}

	public void setMean(FloatType mean) {
		this.mean = mean;
	}

	public void setStdDev(FloatType stdDev) {
		this.stdDev = stdDev;
	}

	@Override
	public void setInput(String name, RandomAccessibleInterval<?> value, String axes) {
		preprocessInput(value, mean, stdDev);
		super.setInput(name, value, axes);
	}


	@Override
	public void run() throws OutOfMemoryError, FileNotFoundException, MissingLibraryException {
		//		super.run();
		ModelZooModel model = loadModel(getTrainedModel());
		if (model != null && model.isInitialized() && this.inputValidationAndMapping(model)) {
			increaseHalo(model);
			try {
				this.preprocessing(model);
				this.executePrediction(model);
				this.postprocessing(model);
			} finally {
				model.dispose();
			}

		} else {
			context.service(LogService.class).error("Model does not exist or cannot be loaded. Exiting.");
			if (model != null) {
				model.dispose();
			}

		}
		postprocessOutput(getOutput(), mean, stdDev);

	}

	private void increaseHalo(ModelZooModel model) {
		//TODO HACK to make tiling work. without increasing the halo the tiles become visible. something's calculated wrong at the border.
		InputImageNode<?> inputNode = model.getInputNodes().get(0);
		for (ModelZooAxis axis : inputNode.getAxes()) {
			axis.setHalo(axis.getHalo()+32);
		}
	}

	private void preprocessInput(RandomAccessibleInterval input, FloatType mean, FloatType stdDev) {
		TrainUtils.normalizeInplace(input, mean, stdDev);
	}

	private void postprocessOutput(RandomAccessibleInterval<FloatType> output, FloatType mean, FloatType stdDev) {
		TrainUtils.denormalizeInplace(output, mean, stdDev, opService);
	}

	private IntervalView<FloatType> getFirstChannel(RandomAccessibleInterval<FloatType> output) {
		long[] dims = new long[output.numDimensions()];
		output.dimensions(dims);
		dims[dims.length-1] = 1;
		return Views.interval(output, new FinalInterval(dims));
	}

	public RandomAccessibleInterval<FloatType> predictPadded(RandomAccessibleInterval<FloatType> input, String axes) throws FileNotFoundException, MissingLibraryException {
		setInput(input, axes);
		run();
		if(getOutput() == null) return null;
		return getOutput();
	}
}
