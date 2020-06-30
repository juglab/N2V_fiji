/*-
 * #%L
 * N2V plugin
 * %%
 * Copyright (C) 2019 - 2020 Center for Systems Biology Dresden
 * %%
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */
package de.csbdresden.n2v.command;

import de.csbdresden.n2v.predict.N2VPrediction;
import de.csbdresden.n2v.train.N2VConfig;
import de.csbdresden.n2v.train.N2VTraining;
import io.scif.MissingLibraryException;
import net.imagej.ImageJ;
import net.imagej.modelzoo.ModelZooArchive;
import net.imagej.modelzoo.ModelZooService;
import net.imagej.ops.OpService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
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
import org.scijava.widget.NumberWidget;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

@Plugin( type = Command.class, menuPath = "Plugins>CSBDeep>N2V>N2V train + predict" )
public class N2VTrainPredictCommand implements Command, Cancelable {

	@Parameter(label = "Image used for training")
	private RandomAccessibleInterval< FloatType > training;

	@Parameter(label = "Image to denoise after training")
	private RandomAccessibleInterval< FloatType > prediction;

	@Parameter(label = "Axes of prediction input (subset of XYZB, B = batch)")
	private String axes = "XY";

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
	private int batchSize = 64;

	@Parameter(label = "Patch shape", min = "16", max = "512", stepSize = "16", style= NumberWidget.SLIDER_STYLE)
	private int patchShape = 64;

	@Parameter(label = "Neighborhood radius")
	private int neighborhoodRadius = 5;

	@Parameter( type = ItemIO.OUTPUT )
	private RandomAccessibleInterval< FloatType > output;

	@Parameter(type = ItemIO.OUTPUT, label = "model from last training step")
	private ModelZooArchive latestTrainedModel;

	@Parameter(type = ItemIO.OUTPUT, label = "model with lowest validation loss")
	private ModelZooArchive bestTrainedModel;

	@Parameter
	private CommandService commandService;

	@Parameter
	private OpService opService;

	@Parameter
	private Context context;

	@Parameter
	private LogService logService;

	@Parameter
	private ModelZooService modelZooService;

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
		n2v.init(new N2VConfig()
				.setTrainDimensions(mode3D ? 3 : 2)
				.setNumEpochs(numEpochs)
				.setStepsPerEpoch(numStepsPerEpoch)
				.setBatchSize(batchSize)
				.setPatchShape(patchShape)
				.setNeighborhoodRadius(neighborhoodRadius));
		if(n2v.getDialog() != null) n2v.getDialog().addTask( "Prediction" );

		try {
			if(training.equals(prediction)) {
				System.out.println("Using 10% of training data for validation");
				n2v.input().addTrainingAndValidationData(training, 0.1);
			} else {
				n2v.input().addTrainingData(training);
				n2v.input().addValidationData(prediction);
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
			File savedModel = n2v.output().exportLatestTrainedModel();
			if(savedModel == null) return;
			openSavedModels(n2v, savedModel);
		} catch (IOException e) {
			e.printStackTrace();
		}

		if(isCanceled()) return;

		if(n2v.getDialog() != null) n2v.getDialog().setTaskStart(2);

		if(latestTrainedModel == null) return;

		N2VPrediction prediction = new N2VPrediction(context);
		prediction.setTrainedModel(latestTrainedModel);
		try {
			this.output = prediction.predict(this.prediction, axes);
		} catch (FileNotFoundException | MissingLibraryException e) {
			e.printStackTrace();
			n2v.getDialog().dispose();
		}

		if(n2v.getDialog() != null) n2v.getDialog().setTaskDone(2);

	}

	private void openSavedModels(N2VTraining training, File savedModel) throws IOException {
		latestTrainedModel = modelZooService.open(savedModel);
		savedModel = training.output().exportBestTrainedModel();
		bestTrainedModel = modelZooService.open(savedModel);
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
