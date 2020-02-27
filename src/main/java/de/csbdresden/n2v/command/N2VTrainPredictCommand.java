package de.csbdresden.n2v.command;

import de.csbdresden.n2v.N2VPrediction;
import de.csbdresden.n2v.N2VTraining;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import org.scijava.Cancelable;
import org.scijava.Context;
import org.scijava.ItemIO;
import org.scijava.ItemVisibility;
import org.scijava.command.Command;
import org.scijava.command.CommandService;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

@Plugin( type = Command.class, menuPath = "Plugins>CSBDeep>N2V>train + predict" )
public class N2VTrainPredictCommand implements Command, Cancelable {

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

	@Parameter(label = "Neighborhood radius")
	private int neighborhoodRadius = 5;

	@Parameter( type = ItemIO.OUTPUT )
	private RandomAccessibleInterval< FloatType > output;

	@Parameter(type = ItemIO.OUTPUT, label = "model from last training step")
	private String latestTrainedModelPath;

	@Parameter(type = ItemIO.OUTPUT, label = "model with lowest validation loss")
	private String bestTrainedModelPath;

	@Parameter
	private CommandService commandService;

	@Parameter
	private OpService opService;

	@Parameter
	private Context context;

	@Parameter
	private LogService logService;

	private boolean canceled = false;

	private ExecutorService pool;
	private Future<?> future;
	private N2VTraining n2v;

	@Override
	public void run() {

		pool = Executors.newSingleThreadExecutor();

		try {

			future = pool.submit(this::mainThread);
			future.get();

		} catch(CancellationException e) {
			logService.warn("N2V train + predict command canceled.");
		} catch (InterruptedException | ExecutionException e) {
			e.printStackTrace();
		}
	}

	private void mainThread() {

		if(training.equals(prediction)) {
			prediction = opService.convert().float32( Views.iterable( prediction ) );
			training = prediction;
		} else {
			prediction = opService.convert().float32( Views.iterable( prediction ) );
			training = opService.convert().float32( Views.iterable( training ) );
		}

		n2v = new N2VTraining(context);
		n2v.addCallbackOnCancel(this::cancel);
		n2v.setTrainDimensions(mode3D ? 3 : 2);
		n2v.setNumEpochs(numEpochs);
		n2v.setStepsPerEpoch(numStepsPerEpoch);
		n2v.setBatchSize(batchSize);
		n2v.setBatchDimLength(batchDimLength);
		n2v.setPatchDimLength(patchDimLength);
		n2v.setNeighborhoodRadius(neighborhoodRadius);
		n2v.init();
		try {
			if(training.equals(prediction)) {
				System.out.println("Using 10% of training data for validation");
				n2v.addTrainingAndValidationData(training, 0.1);
			} else {
				n2v.addTrainingData(training);
				n2v.addValidationData(prediction);
			}
			n2v.train();
			if(n2v.isCanceled()) cancel("");
		}
		catch(Exception e) {
			n2v.dispose();
			e.printStackTrace();
			return;
		}
		try {
			File savedModel = n2v.exportLatestTrainedModel();
			if(savedModel == null) return;
			latestTrainedModelPath = savedModel.getAbsolutePath();
			savedModel = n2v.exportBestTrainedModel();
			bestTrainedModelPath = savedModel.getAbsolutePath();
		} catch (IOException e) {
			e.printStackTrace();
		}

		n2v.getDialog().setTaskStart(2);

		if(latestTrainedModelPath == null) return;

		N2VPrediction prediction = new N2VPrediction(context);
		prediction.setModelFile(new File(latestTrainedModelPath));
		int padding = 32;
		FinalInterval expand = Intervals.expand(this.prediction, padding);
		RandomAccessibleInterval output = prediction.predict(Views.zeroMin(Views.interval(Views.extendZero(this.prediction), expand)));
		this.output = Views.zeroMin(Views.interval(output, Intervals.expand(output, -padding)));

		n2v.getDialog().setTaskDone(2);

	}

	private void cancel() {
		cancel("");
	}

	@Override
	public boolean isCanceled() {
		return canceled;
	}

	@Override
	public void cancel(String reason) {
		canceled = true;
		if(n2v != null) n2v.dispose();
		if(future != null) {
			future.cancel(true);
		}
		if(pool != null) {
			pool.shutdownNow();
		}
	}

	@Override
	public String getCancelReason() {
		return null;
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
