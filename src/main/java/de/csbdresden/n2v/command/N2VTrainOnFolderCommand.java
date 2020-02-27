package de.csbdresden.n2v.command;

import de.csbdresden.n2v.train.N2VConfig;
import de.csbdresden.n2v.train.N2VTraining;
import net.imagej.ImageJ;
import org.scijava.Cancelable;
import org.scijava.Context;
import org.scijava.ItemIO;
import org.scijava.ItemVisibility;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.File;
import java.io.IOException;

import static org.scijava.widget.FileWidget.DIRECTORY_STYLE;

@Plugin( type = Command.class, menuPath = "Plugins>CSBDeep>N2V>train on folder" )
public class N2VTrainOnFolderCommand implements Command, Cancelable {

	@Parameter(label = "Folder containing images used for training", style = DIRECTORY_STYLE)
	private File training;

	@Parameter(label = "Folder containing images used for validation (can be the same as the training folder)")
	private File validation;

	@Parameter(label = "Use 3D model instead of 2D")
	private boolean mode3D = false;

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

	@Parameter(type = ItemIO.OUTPUT, label = "model from last training step")
	private String latestTrainedModelPath;

	@Parameter(type = ItemIO.OUTPUT, label = "model with lowest validation loss")
	private String bestTrainedModelPath;

	@Parameter
	private Context context;

	private boolean canceled;

	@Override
	public void run() {
		N2VTraining n2v = new N2VTraining(context);
		n2v.init(new N2VConfig()
				.setTrainDimensions(mode3D ? 3 : 2)
				.setNumEpochs(numEpochs)
				.setStepsPerEpoch(numStepsPerEpoch)
				.setBatchSize(batchSize)
				.setPatchDimLength(batchDimLength)
				.setPatchDimLength(patchDimLength)
				.setNeighborhoodRadius(neighborhoodRadius));
		try {
			if(training.equals(validation)) {
				System.out.println("Using 10% of training data for validation");
				n2v.input().addTrainingAndValidationData(training, 0.1);
			} else {
				n2v.input().addTrainingData(training);
				n2v.input().addValidationData(validation);
			}
			n2v.train();
		}
		catch(Exception e) {
			n2v.dispose();
			e.printStackTrace();
			return;
		}
		try {
			File savedModel = n2v.output().exportLatestTrainedModel();
			if(savedModel == null) return;
			latestTrainedModelPath = savedModel.getAbsolutePath();
			savedModel = n2v.output().exportBestTrainedModel();
			bestTrainedModelPath = savedModel.getAbsolutePath();
		} catch (IOException e) {
			e.printStackTrace();
		}
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

	public static void main( final String... args ) throws Exception {

		final ImageJ ij = new ImageJ();
		ij.launch( args );

		final File trainingFolderFile = new File( "/home/random/Development/imagej/project/3DAnalysisFIBSegmentation/owncloud/screenshots/" );

		ij.command().run( N2VTrainOnFolderCommand.class, true,"training", trainingFolderFile, "validation", trainingFolderFile).get();

	}
}
