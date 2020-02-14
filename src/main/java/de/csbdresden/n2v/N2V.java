package de.csbdresden.n2v;

import net.imagej.ImageJ;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import org.scijava.Context;
import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.command.CommandModule;
import org.scijava.command.CommandService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutionException;

@Plugin( type = Command.class, menuPath = "Plugins>CSBDeep>N2V" )
public class N2V implements Command {

	@Parameter
	private RandomAccessibleInterval< FloatType > training;

	@Parameter
	private RandomAccessibleInterval< FloatType > prediction;

	@Parameter( type = ItemIO.OUTPUT )
	private RandomAccessibleInterval< FloatType > output;

	@Parameter
	int numEpochs = 20;

	@Parameter
	int numStepsPerEpoch = 20;

	@Parameter
	int batchSize = 128;

	@Parameter
	private CommandService commandService;

	@Parameter
	private Context context;

	private File trainedModelZip;

	private N2VDialog dialog;

	@Override
	public void run() {
		dialog = new N2VDialog();

		runTraining();
		runPrediction( prediction );
		
	}

	private void runTraining() {
		N2VTraining n2v = new N2VTraining(context);
		n2v.init(dialog);
		n2v.setNumEpochs(numEpochs);
		n2v.setStepsPerEpoch(numStepsPerEpoch);
		n2v.setBatchSize(batchSize);
		n2v.addTrainingData(training);
		n2v.train();
		try {
			trainedModelZip = n2v.exportTrainedModel();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void runPrediction( RandomAccessibleInterval< FloatType > inputRAI ) {

		try {
			final CommandModule module = commandService.run(
					N2VPredictionCommand.class, false,
					"prediction", inputRAI,
					"modelFile", trainedModelZip.getAbsolutePath()).get();
			output = (RandomAccessibleInterval<FloatType>) module.getOutput("output");
		} catch (InterruptedException | ExecutionException e) {
			e.printStackTrace();
		}

	}

	public static void main( final String... args ) throws Exception {

		final ImageJ ij = new ImageJ();
		ij.launch( args );

//		File graphDefFile = new File("/home/random/Development/imagej/project/CSBDeep/N2V/test-graph.pb");
		final File trainingImgFile = new File( "/Users/turek/Desktop/train.tif" );

		if ( trainingImgFile.exists() ) {
			RandomAccessibleInterval _input = ( RandomAccessibleInterval ) ij.io().open( trainingImgFile.getAbsolutePath() );
			RandomAccessibleInterval _inputConverted = ij.op().convert().float32( Views.iterable( _input ) );
//			_inputConverted = Views.interval(_inputConverted, new FinalInterval(1024, 1024  ));

			RandomAccessibleInterval training = ij.op().copy().rai( _inputConverted );
			RandomAccessibleInterval prediction = training;

			CommandModule plugin = ij.command().run( N2V.class, true,"training", training, "prediction", prediction).get();
			ij.ui().show( plugin.getOutput( "output" ) );
		} else
			System.out.println( "Cannot find training image " + trainingImgFile.getAbsolutePath() );

	}
}
