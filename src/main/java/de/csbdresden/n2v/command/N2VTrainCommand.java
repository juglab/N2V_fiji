package de.csbdresden.n2v.command;

import de.csbdresden.n2v.N2VTraining;
import net.imagej.ImageJ;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import org.scijava.Cancelable;
import org.scijava.Context;
import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.command.CommandService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.File;
import java.io.IOException;

@Plugin( type = Command.class, menuPath = "Plugins>CSBDeep>N2V>train" )
public class N2VTrainCommand implements Command, Cancelable {

	@Parameter
	private RandomAccessibleInterval< FloatType > training;

	@Parameter
	private RandomAccessibleInterval< FloatType > validation;

	@Parameter(type = ItemIO.OUTPUT)
	private String trainedModelPath;

	@Parameter(type = ItemIO.OUTPUT)
	private float mean;

	@Parameter(type = ItemIO.OUTPUT)
	private float stdDev;

	@Parameter
	private boolean mode3D = false;

	@Parameter
	private int numEpochs = 300;

	@Parameter
	private int numStepsPerEpoch = 200;

	@Parameter
	private int batchSize = 128;

	@Parameter
	private int batchDimLength = 180;

	@Parameter
	private int patchDimLength = 60;

	@Parameter
	private CommandService commandService;

	@Parameter
	private Context context;

	private N2VTraining n2v;
	private boolean canceled;

	@Override
	public void run() {
		n2v = new N2VTraining(context);
		n2v.init();
		if(mode3D) n2v.setTrainDimensions(3);
		else n2v.setTrainDimensions(2);
		n2v.setNumEpochs(numEpochs);
		n2v.setStepsPerEpoch(numStepsPerEpoch);
		n2v.setBatchSize(batchSize);
		n2v.setBatchDimLength(batchDimLength);
		n2v.setPatchDimLength(patchDimLength);
		try {
			if(training.equals(validation)) {
				System.out.println("Using 10% of training data for validation");
				n2v.addTrainingAndValidationData(training, 0.1);
			} else {
				n2v.addTrainingData(training);
				n2v.addValidationData(validation);
			}
			n2v.train();
		}
		catch(Exception e) {
			n2v.dispose();
			e.printStackTrace();
			return;
		}
		stdDev = n2v.getStdDev().getRealFloat();
		mean = n2v.getMean().getRealFloat();
		try {
			File savedModel = n2v.exportTrainedModel();
			if(savedModel == null) return;
			trainedModelPath = savedModel.getAbsolutePath();
		} catch (IOException e) {
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

			ij.command().run( N2VTrainCommand.class, true,"training", training, "validation", prediction).get();
		} else
			System.out.println( "Cannot find training image " + trainingImgFile.getAbsolutePath() );

	}

	@Override
	public boolean isCanceled() {
		return canceled;
	}

	@Override
	public void cancel(String reason) {
		canceled = true;
	}

	@Override
	public String getCancelReason() {
		return null;
	}
}
