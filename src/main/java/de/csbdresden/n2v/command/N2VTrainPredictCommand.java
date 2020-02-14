package de.csbdresden.n2v.command;

import net.imagej.ImageJ;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.command.CommandModule;
import org.scijava.command.CommandService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.File;
import java.util.concurrent.ExecutionException;

@Plugin( type = Command.class, menuPath = "Plugins>CSBDeep>N2V>train + predict" )
public class N2VTrainPredictCommand implements Command {

	@Parameter
	private RandomAccessibleInterval< FloatType > training;

	@Parameter
	private RandomAccessibleInterval< FloatType > prediction;

	@Parameter( type = ItemIO.OUTPUT )
	private RandomAccessibleInterval< FloatType > output;

	@Parameter(type = ItemIO.OUTPUT)
	private String trainedModelPath;

	@Parameter
	int numEpochs = 300;

	@Parameter
	int numStepsPerEpoch = 200;

	@Parameter
	int batchSize = 180;

	@Parameter
	int batchDimlength = 180;

	@Parameter
	int patchDimlength = 60;

	@Parameter
	private CommandService commandService;

	private float mean;
	private float stdDev;

	@Override
	public void run() {

		if(runTraining()) runPrediction();

	}

	private boolean runTraining() {
		try {
			final CommandModule module = commandService.run(
					N2VTrainCommand.class, false,
					"training", training,
					"validation", prediction,
					"numStepsPerEpoch", numStepsPerEpoch,
					"numEpochs", numEpochs,
					"batchSize", batchSize,
					"batchDimLength", batchDimlength,
					"patchDimLength", patchDimlength).get();
			trainedModelPath = (String) module.getOutput("trainedModelPath");
			if(trainedModelPath == null) return false;
			mean = (float) module.getOutput("mean");
			stdDev = (float) module.getOutput("stdDev");
		} catch (InterruptedException | ExecutionException e) {
			e.printStackTrace();
			return false;
		}
		return true;
	}

	private void runPrediction() {
		try {
			final CommandModule module = commandService.run(
					N2VPredictCommand.class, false,
					"prediction", prediction,
					"mean", mean,
					"stdDev", stdDev,
					"modelFile", new File(trainedModelPath)).get();
			output = (RandomAccessibleInterval<FloatType>) module.getOutput("output");
		} catch (InterruptedException | ExecutionException e) {
			e.printStackTrace();
		}
	}

	public static void main( final String... args ) throws Exception {

		final ImageJ ij = new ImageJ();
		ij.launch( args );

		final File trainingImgFile = new File( "/home/random/Development/imagej/project/CSBDeep/train.tif" );

		if ( trainingImgFile.exists() ) {
			RandomAccessibleInterval _input = ( RandomAccessibleInterval ) ij.io().open( trainingImgFile.getAbsolutePath() );
			RandomAccessibleInterval _inputConverted = ij.op().convert().float32( Views.iterable( _input ) );
//			_inputConverted = Views.interval(_inputConverted, new FinalInterval(1024, 1024  ));

			RandomAccessibleInterval training = ij.op().copy().rai( _inputConverted );
			RandomAccessibleInterval prediction = training;

			ij.command().run( N2VTrainPredictCommand.class, true,"training", training, "prediction", prediction).get();
		} else
			System.out.println( "Cannot find training image " + trainingImgFile.getAbsolutePath() );

	}
}
