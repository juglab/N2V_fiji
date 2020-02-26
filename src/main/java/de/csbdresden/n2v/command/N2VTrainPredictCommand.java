package de.csbdresden.n2v.command;

import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import org.scijava.ItemIO;
import org.scijava.ItemVisibility;
import org.scijava.command.Command;
import org.scijava.command.CommandModule;
import org.scijava.command.CommandService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.File;
import java.util.concurrent.ExecutionException;

@Plugin( type = Command.class, menuPath = "Plugins>CSBDeep>N2V>train + predict" )
public class N2VTrainPredictCommand implements Command {

	@Parameter(label = "Image used for training")
	private RandomAccessibleInterval< FloatType > training;

	@Parameter(label = "Image to denoise after training")
	private RandomAccessibleInterval< FloatType > prediction;

	//TODO make these parameters work
//	@Parameter(label = "Training mode", choices = {"start new training", "continue training"})
//	private String trainBase;
//
//	@Parameter(required = false, visibility = ItemVisibility.MESSAGE)
//	private String newTrainingLabel = "<html><br/><span style='font-weight: normal'>Options for new training</span></html>";

	@Parameter(label = "Use 3D model instead of 2D")
	private boolean mode3D = false;

	//TODO make these parameters work
//	@Parameter(label = "Start from model trained on noise")
//	private boolean startFromNoise = false;
//
//	@Parameter(required = false, visibility = ItemVisibility.MESSAGE)
//	private String continueTrainingLabel = "<html><br/><span style='font-weight: normal'>Options for continuing training</span></html>";
//
//	@Parameter(required = false, label = "Pretrained model file (.zip)")
//	private File pretrainedNetwork;

	@Parameter(required = false, visibility = ItemVisibility.MESSAGE)
	private String advancedLabel = "<html><br/><span style='font-weight: normal'>Advanced options</span></html>";

	@Parameter(label = "Number of epochs")
	private int numEpochs = 300;

	@Parameter(label = "Number of steps per epoch")
	private int numStepsPerEpoch = 200;

	@Parameter(label = "Batch size per step")
	private int batchSize = 180;

	@Parameter(label = "Dimension length of batch")
	private int batchDimLength = 180;

	@Parameter(label = "Dimension length of patch")
	private int patchDimLength = 60;

	@Parameter( type = ItemIO.OUTPUT )
	private RandomAccessibleInterval< FloatType > output;

	@Parameter(type = ItemIO.OUTPUT)
	private String trainedModelPath;

	@Parameter
	private CommandService commandService;

	@Parameter
	private OpService opService;

	@Override
	public void run() {

		if(training.equals(prediction)) {
			prediction = opService.convert().float32( Views.iterable( prediction ) );
			training = prediction;
		} else {
			prediction = opService.convert().float32( Views.iterable( prediction ) );
			training = opService.convert().float32( Views.iterable( training ) );
		}

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
					"batchDimLength", batchDimLength,
					"patchDimLength", patchDimLength,
					"mode3D", mode3D).get();
			trainedModelPath = (String) module.getOutput("bestTrainedModelPath");
			if(trainedModelPath == null) return false;
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
			RandomAccessibleInterval training = ( RandomAccessibleInterval ) ij.io().open( trainingImgFile.getAbsolutePath() );

			ij.command().run( N2VTrainPredictCommand.class, true,"training", training, "prediction", training).get();
		} else
			System.out.println( "Cannot find training image " + trainingImgFile.getAbsolutePath() );

	}
}
