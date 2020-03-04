package de.csbdresden.n2v.train;

import de.csbdresden.csbdeep.network.model.tensorflow.DatasetTensorFlowConverter;
import de.csbdresden.n2v.ui.N2VProgress;
import de.csbdresden.n2v.util.N2VUtils;
import io.scif.services.DatasetIOService;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import net.imagej.tensorflow.TensorFlowService;
import net.imglib2.Dimensions;
import net.imglib2.FinalDimensions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import org.apache.commons.math3.util.Pair;
import org.scijava.Context;
import org.scijava.app.StatusService;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.thread.DefaultThreadService;
import org.scijava.ui.DialogPrompt;
import org.scijava.ui.UIService;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class N2VTraining {

	private File graphDefFile;

	@Parameter
	private TensorFlowService tensorFlowService;

	@Parameter
	private OpService opService;

	@Parameter
	private UIService uiService;

	@Parameter
	private LogService logService;

	@Parameter
	private DatasetIOService datasetIOService;

	@Parameter
	private StatusService statusService;

	@Parameter
	Context context;

	private static final String tensorXOpName = "input";
	private static final String tensorYOpName = "activation_11_target";
	private static final String trainingTargetOpName = "training/group_deps";
	private static final String predictionTargetOpName = "activation_11/Identity";
	private static final String validationTargetOpName = "group_deps";
	private static final String sampleWeightsOpName = "activation_11_sample_weights";
	private static final String learningPhaseOpName = "batch_normalization_1/keras_learning_phase";
	private static final String lossOpName = "loss/mul";
	private static final String absOpName = "metrics/n2v_abs/Mean";
	private static final String mseOpName = "metrics/n2v_mse/Mean";
	private static final String lrOpName = "Adam/lr/read";
	private static final String lrAssignOpName = "Adam/lr";

	private N2VProgress dialog;
	private PreviewHandler previewHandler;
	private OutputHandler outputHandler;
	private InputHandler inputHandler;

	private boolean stopTraining = false;

	private List<TrainingCallback> onEpochDoneCallbacks = new ArrayList<>();
	private List<TrainingCanceledCallback> onTrainingCanceled = new ArrayList<>();

	//TODO make setters etc.
	private boolean continueTraining = false;
	private File zipFile;
	private boolean canceled = false;
	private ExecutorService pool;
	private Future<?> future;
	private Session session;
	private N2VConfig config;
	private int stepsFinished = 0;

	public interface TrainingCallback {

		void accept(N2VTraining training);

	}
	public interface TrainingCanceledCallback {

		void accept();
	}
	public N2VTraining(Context context) {
		context.inject(this);
	}

	public void init(String trainedModel, N2VConfig config) {
		if (Thread.interrupted()) return;
		continueTraining = true;
		zipFile = new File(trainedModel);
		init(config);
	}

	public void init(N2VConfig config) {

		this.config = config;

		inputHandler= new InputHandler(context, config);

		if (Thread.interrupted()) return;
		if(!headless()) {
			dialog = N2VProgress.create( this, config.getNumEpochs(), config.getStepsPerEpoch(), statusService, new DefaultThreadService() );
			inputHandler.setDialog(dialog);
			dialog.addTask( "Preparation" );
			dialog.addTask( "Training" );
			dialog.display();
			dialog.setTaskStart( 0 );
			dialog.setCurrentTaskMessage( "Loading TensorFlow" );

			//TODO warning if no GPU support
			//dialog.setWarning("WARNING: this will take for ever!");

		}

		if (Thread.interrupted()) return;
		logService.info( "Load TensorFlow.." );
		tensorFlowService.loadLibrary();
		logService.info( tensorFlowService.getStatus().getInfo() );

		addCallbackOnEpochDone(new ReduceLearningRateOnPlateau()::reduceLearningRateOnPlateau);


	}
	private boolean headless() {
		return uiService.isHeadless();
	}
	public void train() {

		pool = Executors.newSingleThreadExecutor();

		try {

			future = pool.submit(this::mainThread);
			future.get();

		} catch(InterruptedException | CancellationException e) {
			if(stopTraining) return;
			logService.warn("N2V training canceled.");
		} catch (ExecutionException e) {
			e.printStackTrace();
		}

	}

	private void mainThread() {
		outputHandler = new OutputHandler(config, this);
		addCallbackOnEpochDone(outputHandler::copyBestModel);

		logService.info( "Create session.." );
		if ( !headless() ) dialog.setCurrentTaskMessage( "Creating session" );
		if (Thread.interrupted() || isCanceled()) return;

		try (Graph graph = new Graph();
		     Session sess = new Session( graph )) {

			this.session = sess;

			try {
				if(!continueTraining) {
					logService.info( "Import graph.." );
					output().loadUntrainedGraph(graph);
					outputHandler.createSavedModelDirs();
				}
				else {
					logService.info( "Import trained graph.." );
					File trainedModel = output().loadTrainedGraph(graph, zipFile);
					outputHandler.createSavedModelDirsFromExisting(trainedModel);
				}
			} catch (IOException e) {
				e.printStackTrace();
			}

			if (Thread.interrupted() || isCanceled()) return;
			Operation opTrain = graph.operation( trainingTargetOpName );
			if ( opTrain == null ) throw new RuntimeException( "Training op not found" );

			output().initTensors(sess);

			if (Thread.interrupted() || isCanceled()) return;
			logService.info("Normalizing..");
			if(!headless()) dialog.setCurrentTaskMessage("Normalizing ...");

			normalize();

			if (Thread.interrupted() || isCanceled()) return;
			logService.info("Augment tiles..");
			if(!headless()) dialog.setCurrentTaskMessage("Augment tiles ...");

			N2VDataGenerator.augment(input().getX());
			N2VDataGenerator.augment(input().getValidationX());

			if (Thread.interrupted() || isCanceled()) return;
			logService.info("Prepare training batches...");
			if(!headless()) dialog.setCurrentTaskMessage("Prepare training batches...");

//			uiService.show("_X", opService.copy().rai(_X));
//			uiService.show("_validationX",opService.copy().rai(_validationX));

			double n2v_perc_pix = 1.6;

			if (!batchNumSufficient(input().getX().size())) return;

			N2VDataWrapper<FloatType> training_data = makeTrainingData(n2v_perc_pix);

			if (Thread.interrupted()) return;
			logService.info("Prepare validation batches..");
			if(!headless()) dialog.setCurrentTaskMessage("Prepare validation batches...");

			List<Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>>> validation_data = makeValidationData(n2v_perc_pix);

			int index = 0;
//			List<RandomAccessibleInterval<FloatType>> inputs = new ArrayList<>();
//			List<RandomAccessibleInterval<FloatType>> targets = new ArrayList<>();
			Tensor<Float> tensorWeights = makeWeightsTensor();

			if (Thread.interrupted() || isCanceled()) {
				tensorWeights.close();
				return;
			}
			logService.info("Start training..");
			if(!headless()) {
				dialog.setCurrentTaskMessage("Starting training ...");
				dialog.setTaskDone( 0 );
				dialog.setTaskStart( 1 );
			}

			RemainingTimeEstimator remainingTimeEstimator = new RemainingTimeEstimator();
			remainingTimeEstimator.setNumSteps(config().getNumEpochs());


			if(!headless()) {
				previewHandler = new PreviewHandler(context, config().getTrainDimensions());
				previewHandler.update(validation_data.get(0).getFirst(), validation_data.get(0).getFirst());
			}

			for (int i = 0; i < config().getNumEpochs(); i++) {
				remainingTimeEstimator.setCurrentStep(i);
				String remainingTimeString = remainingTimeEstimator.getRemainingTimeString();
				logService.info("Epoch " + (i + 1) + "/" + config().getNumEpochs() + " " + remainingTimeString);

				List<Double> losses = new ArrayList<>(config().getStepsPerEpoch());

				for (int j = 0; j < config().getStepsPerEpoch() && !stopTraining; j++) {

					if (Thread.interrupted() || isCanceled()) {
						tensorWeights.close();
						return;
					}

					if (index * config().getTrainBatchSize() + config().getTrainBatchSize() > input().getX().size() - 1) {
						index = 0;
						logService.info("starting with index 0 of training batches");
					}

					Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> item = training_data.getItem(index);
//					inputs.add( item.getFirst() );
//					targets.add( item.getSecond() );
//					uiService.show("input", opService.copy().rai(item.getFirst()));
//					uiService.show("target", opService.copy().rai(item.getSecond()));

					runTrainingOp(sess, tensorWeights, item);

					losses.add((double) outputHandler.getCurrentLoss());
					logStatusInConsole(j + 1, config().getStepsPerEpoch(), outputHandler);
					if(!headless()) dialog.updateTrainingProgress(i + 1, j + 1);

					stepsFinished = config().getStepsPerEpoch()*i+j+1;

					index++;

				}

				if (Thread.interrupted() || isCanceled()) {
					tensorWeights.close();
					return;
				}
				training_data.on_epoch_end();
				float validationLoss = validate(sess, validation_data, tensorWeights);
				if (Thread.interrupted() || isCanceled()) {
					tensorWeights.close();
					return;
				}
				outputHandler.saveCheckpoint(sess);
				outputHandler.setCurrentValidationLoss(validationLoss);
				if(!headless()) dialog.updateTrainingChart(i + 1, losses, validationLoss);
				onEpochDoneCallbacks.forEach(callback -> callback.accept(this));
			}

			tensorWeights.close();

//			sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/control_dependency").run();

			if ( !headless() ) dialog.setTaskDone( 1 );
			stopTraining = true;
			logService.info( "Training done." );

//			if (inputs.size() > 0) uiService.show("inputs", Views.stack(inputs));
//			if (targets.size() > 0) uiService.show("targets", Views.stack(targets));

		}
		catch(IllegalStateException e) {
			cancel();
			e.printStackTrace();
			if(e.getMessage().contains("OOM")) {
				logService.error("Not enough memory available. Try to reduce the training batch size.");
			} else {
				e.printStackTrace();
			}
		}
	}

	private N2VDataWrapper<FloatType> makeTrainingData(double n2v_perc_pix) {
		long[] patchShapeData = new long[config().getTrainDimensions()];
		Arrays.fill(patchShapeData, config().getTrainPatchDimLength());
		Dimensions patch_shape = new FinalDimensions(patchShapeData);

		return new N2VDataWrapper<>(input().getX(), config().getTrainBatchSize(), n2v_perc_pix, patch_shape, config().getNeighborhoodRadius(), N2VDataWrapper::uniform_withCP);
	}

	private List<Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>>> makeValidationData(double n2v_perc_pix) {
		int n_train = input().getX().size();
		int n_val = input().getValidationX().size();
		double frac_val = (1.0 * n_val) / (n_train + n_val);
		double frac_warn = 0.05;
		if (frac_val < frac_warn) {
			logService.info("small number of validation images (only " + (100 * frac_val) + "% of all images)");
		}
		long[] patchShapeData = new long[config().getTrainDimensions()];
		Arrays.fill(patchShapeData, config().getTrainPatchDimLength());
		Dimensions patch_shape = new FinalDimensions(patchShapeData);
		N2VDataWrapper<FloatType> valData = new N2VDataWrapper<>(input().getValidationX(),
				Math.min(config().getTrainBatchSize(), input().getValidationX().size()),
				n2v_perc_pix, patch_shape, config().getNeighborhoodRadius(),
				N2VDataWrapper::uniform_withCP);

		List<Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>>> validationDataList = new ArrayList<>();
		for (int i = 0; i < valData.numBatches(); i++) {
			validationDataList.add(valData.getItem(i));
		}
		return validationDataList;
	}

	private void normalize() {
		FloatType mean = outputHandler.getMean();
		FloatType stdDev = outputHandler.getStdDev();
		mean.set( opService.stats().mean( Views.iterable( Views.stack(input().getX()) ) ).getRealFloat() );
		stdDev.set( opService.stats().stdDev( Views.iterable( Views.stack(input().getX()) ) ).getRealFloat() );
		logService.info("mean: " + mean.get());
		logService.info("stdDev: " + stdDev.get());

		N2VUtils.normalize( input().getX(), mean, stdDev, opService );
		N2VUtils.normalize( input().getValidationX(), mean, stdDev, opService );
	}

	private void runTrainingOp(Session sess, Tensor<Float> tensorWeights, Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> item) {
		Tensor tensorX = DatasetTensorFlowConverter.datasetToTensor(item.getFirst(), getMapping());
		Tensor tensorY = DatasetTensorFlowConverter.datasetToTensor(item.getSecond(), getMapping());

		Session.Runner runner = sess.runner();

		Tensor<Float> learningRate = Tensors.create(outputHandler.getCurrentLearningRate());
		Tensor<Boolean> learningPhase = Tensors.create(true);
		runner.feed(tensorXOpName, tensorX).feed(tensorYOpName, tensorY)
				.feed(learningPhaseOpName, learningPhase)
				.feed(lrAssignOpName, learningRate)
				.feed(sampleWeightsOpName, tensorWeights).addTarget(trainingTargetOpName);
		runner.fetch(lossOpName);
		runner.fetch(absOpName);
		runner.fetch(mseOpName);
		runner.fetch(lrOpName);

		List<Tensor<?>> fetchedTensors = runner.run();
		outputHandler.setCurrentLoss(fetchedTensors.get(0).floatValue());
		outputHandler.setCurrentAbs(fetchedTensors.get(1).floatValue());
		outputHandler.setCurrentMse(fetchedTensors.get(2).floatValue());
		outputHandler.setCurrentLearningRate(fetchedTensors.get(3).floatValue());

		fetchedTensors.forEach(Tensor::close);
		tensorX.close();
		tensorY.close();
		learningPhase.close();
		learningRate.close();
	}

	public void addCallbackOnEpochDone(TrainingCallback callback) {
		onEpochDoneCallbacks.add(callback);
	}

	private int[] getMapping() {
		if(config().getTrainDimensions() == 2) return new int[]{ 1, 2, 0, 3 };
		if(config().getTrainDimensions() == 3) return new int[]{ 1, 2, 3, 0, 4 };
		return new int[0];
	}

	private boolean batchNumSufficient(int n_train) {
		if(config().getTrainBatchSize() > n_train) {
			String errorMsg = "Not enough training data (" + n_train + " batches). At least " + config().getTrainBatchSize() + " batches needed.";
			logService.error(errorMsg);
			stopTraining = true;
			dispose();
			uiService.showDialog(errorMsg, DialogPrompt.MessageType.ERROR_MESSAGE);
			return false;
		}
		return true;
	}

	private Tensor<Float> makeWeightsTensor() {
		float[] weightsdata = new float[config().getTrainBatchSize()];
		Arrays.fill(weightsdata, 1);
		return Tensors.create(weightsdata);
	}

	private float validate(Session sess, List<Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>>> validationData, Tensor tensorWeights) {

		float avgLoss = 0;
		float avgAbs = 0;
		float avgMse = 0;

		long validationBatches = validationData.size();
		for (int i = 0; i < validationBatches; i++) {

			if (Thread.interrupted() || isCanceled()) {
				break;
			}

			Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> item = validationData.get(i);

			Tensor tensorX = DatasetTensorFlowConverter.datasetToTensor(item.getFirst(), getMapping());
			Tensor tensorY = DatasetTensorFlowConverter.datasetToTensor(item.getSecond(), getMapping());
			Tensor<Boolean> tensorLearningPhase = Tensors.create(false);

			Session.Runner runner = sess.runner();

			runner.feed(tensorXOpName, tensorX)
					.feed(tensorYOpName, tensorY)
					.feed(learningPhaseOpName, tensorLearningPhase)
					.feed(sampleWeightsOpName, tensorWeights)
					.addTarget(validationTargetOpName);
			runner.fetch(lossOpName);
			runner.fetch(absOpName);
			runner.fetch(mseOpName);
			if(!headless() && i == 0) runner.fetch(predictionTargetOpName);

			List<Tensor<?>> fetchedTensors = runner.run();

			avgLoss += fetchedTensors.get(0).floatValue();
			avgAbs += fetchedTensors.get(1).floatValue();
			avgMse += fetchedTensors.get(2).floatValue();

			if(!headless() && i == 0) {
				Tensor outputTensor = fetchedTensors.get(3);
				RandomAccessibleInterval<FloatType> output = DatasetTensorFlowConverter.tensorToDataset(outputTensor, new FloatType(), getMapping(), false);
				previewHandler.update(item.getFirst(), output);
//			updateHistoryImage(output);
			}
			fetchedTensors.forEach(Tensor::close);
			tensorX.close();
			tensorY.close();
			tensorLearningPhase.close();
		}
		avgLoss /= (float)validationBatches;
		avgAbs /= (float)validationBatches;
		avgMse /= (float)validationBatches;

		logService.info("\nValidation loss: " + avgLoss + " abs: " + avgAbs + " mse: " + avgMse);
		return avgLoss;
	}

	public int getStepsFinished() {
		return stepsFinished;
	}

	public void setLearningRate(float newLR) {
		outputHandler.setCurrentLearningRate(newLR);
	}

	private static void logStatusInConsole(int step, int stepTotal, OutputHandler outputHandler) {
		int maxBareSize = 10; // 10unit for 100%
		int remainProcent = ( ( 100 * step ) / stepTotal ) / maxBareSize;
		char defaultChar = '-';
		String icon = "*";
		String bare = new String( new char[ maxBareSize ] ).replace( '\0', defaultChar ) + "]";
		StringBuilder bareDone = new StringBuilder();
		bareDone.append( "[" );
		for ( int i = 0; i < remainProcent; i++ ) {
			bareDone.append( icon );
		}
		String bareRemain = bare.substring( remainProcent );
		System.out.printf( "%d / %d %s%s - loss: %f mse: %f abs: %f lr: %f\n", step, stepTotal, bareDone, bareRemain,
				outputHandler.getCurrentLoss(),
				outputHandler.getCurrentMse(),
				outputHandler.getCurrentAbs(),
				outputHandler.getCurrentLearningRate() );
	}

	public N2VProgress getDialog() {
		return dialog;
	}

	public N2VConfig config() {
		return config;
	}

	public InputHandler input() {
		return inputHandler;
	}

	public OutputHandler output() {
		return outputHandler;
	}

	public void stopTraining() {
		if(stopTraining) {
			if(getDialog() != null) dialog.dispose();
			return;
		}
		stopTraining = true;
		getDialog().setTaskDone(1);
		if(session != null) outputHandler.saveCheckpoint(session);
		if(future != null) {
			future.cancel(true);
		}
		if(pool != null) {
			pool.shutdownNow();
		}
	}

	public void cancel() {
		canceled = true;
		onTrainingCanceled.forEach(TrainingCanceledCallback::accept);
		if(future != null) {
			future.cancel(true);
		}
		if(pool != null) {
			pool.shutdown();
		}
	}

	public boolean isCanceled() {
		return canceled;
	}

	public void addCallbackOnCancel(TrainingCanceledCallback callback) {
		onTrainingCanceled.add(callback);
	}

	public void dispose() {
		if(dialog != null) dialog.dispose();
		if(outputHandler != null) outputHandler.dispose();
	}

	public static void main( final String... args ) throws Exception {

		final ImageJ ij = new ImageJ();

		ij.launch( args );

//		ij.log().setLevel(LogLevel.TRACE);

//		File graphDefFile = new File("/home/random/Development/imagej/project/CSBDeep/N2V/test-graph.pb");

		final File trainingImgFile = new File( "/home/random/Development/imagej/project/CSBDeep/train.tif" );

		if ( trainingImgFile.exists() ) {
			RandomAccessibleInterval _input = ( RandomAccessibleInterval ) ij.io().open( trainingImgFile.getAbsolutePath() );
			RandomAccessibleInterval _inputConverted = ij.op().convert().float32( Views.iterable( _input ) );
//			_inputConverted = Views.interval(_inputConverted, new FinalInterval(1024, 1024  ));

			RandomAccessibleInterval training = ij.op().copy().rai( _inputConverted );

			N2VTraining n2v = new N2VTraining(ij.context());
			n2v.init(new N2VConfig()
					.setNumEpochs(5)
					.setStepsPerEpoch(5)
					.setBatchSize(128)
					.setPatchDimLength(64));
			n2v.input().addTrainingAndValidationData(training, 0.1);
			n2v.train();
		} else
			System.out.println( "Cannot find training image " + trainingImgFile.getAbsolutePath() );

	}

}
