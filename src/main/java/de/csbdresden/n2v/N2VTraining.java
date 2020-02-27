package de.csbdresden.n2v;

import de.csbdresden.csbdeep.network.model.tensorflow.DatasetTensorFlowConverter;
import de.csbdresden.n2v.ui.N2VProgress;
import io.scif.services.DatasetIOService;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import net.imagej.tensorflow.TensorFlowService;
import net.imglib2.Dimensions;
import net.imglib2.FinalDimensions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.converter.RealFloatConverter;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import org.apache.commons.compress.utils.IOUtils;
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
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
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

	private final List< RandomAccessibleInterval< FloatType > > X = new ArrayList<>();
	private final List< RandomAccessibleInterval< FloatType > > validationX = new ArrayList<>();

	private int numEpochs = 300;
	private int trainBatchSize = 180;
	private int trainBatchDimLength = 180;
	private int trainPatchDimLength = 60;
	private int stepsPerEpoch = 200;
	private int neighborhoodRadius = 5;

	private int trainDimensions = 2;

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

	public interface TrainingCallback {

		void accept(N2VTraining training);
	}

	public interface TrainingCanceledCallback {
		void accept();
	}

	public N2VTraining(Context context) {
		context.inject(this);
	}

	public void init(String trainedModel) {
		if (Thread.interrupted()) return;
		continueTraining = true;
		zipFile = new File(trainedModel);
		init();
	}
	public void init() {

		outputHandler = new OutputHandler(trainDimensions);

		if (Thread.interrupted()) return;
		if(!headless()) {
			dialog = N2VProgress.create( this, numEpochs, stepsPerEpoch, statusService, new DefaultThreadService() );
			dialog.addTask( "Preparation" );
			dialog.addTask( "Training" );
			dialog.addTask( "Prediction" );
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
		addCallbackOnEpochDone(outputHandler::copyBestModel);

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
		logService.info( "Create session.." );
		if ( !headless() ) dialog.setCurrentTaskMessage( "Creating session" );
		if (Thread.interrupted()) return;

		try (Graph graph = new Graph();
		     Session sess = new Session( graph )) {

			this.session = sess;

			try {
				if(!continueTraining) {
					loadUntrainedGraph(graph);
					outputHandler.createSavedModelDirs();
				}
				else {
					File trainedModel = loadTrainedGraph(graph, zipFile);
					outputHandler.createSavedModelDirsFromExisting(trainedModel);
				}
			} catch (IOException e) {
				e.printStackTrace();
			}

			if (Thread.interrupted()) return;
			Operation opTrain = graph.operation( trainingTargetOpName );
			if ( opTrain == null ) throw new RuntimeException( "Training op not found" );

			outputHandler.initTensors(sess);

			if (Thread.interrupted()) return;
			logService.info("Normalizing..");
			if(!headless()) dialog.setCurrentTaskMessage("Normalizing ...");

			normalize();
			outputHandler.writeModelConfigFile();

			if (Thread.interrupted()) return;
			logService.info("Augment tiles..");
			if(!headless()) dialog.setCurrentTaskMessage("Augment tiles ...");

			N2VDataGenerator.augment(X);
			N2VDataGenerator.augment(validationX);

			if (Thread.interrupted()) return;
			logService.info("Prepare training batches...");
			if(!headless()) dialog.setCurrentTaskMessage("Prepare training batches...");

//			uiService.show("_X", opService.copy().rai(_X));
//			uiService.show("_validationX",opService.copy().rai(_validationX));

			double n2v_perc_pix = 1.6;

			if (!batchNumSufficient(X.size())) return;

			N2VDataWrapper<FloatType> training_data = makeTrainingData(n2v_perc_pix);

			if (Thread.interrupted()) return;
			logService.info("Prepare validation batches..");
			if(!headless()) dialog.setCurrentTaskMessage("Prepare validation batches...");

			List<Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>>> validation_data = makeValidationData(n2v_perc_pix);

			int index = 0;
//			List<RandomAccessibleInterval<FloatType>> inputs = new ArrayList<>();
//			List<RandomAccessibleInterval<FloatType>> targets = new ArrayList<>();
			Tensor<Float> tensorWeights = makeWeightsTensor();

			if (Thread.interrupted()) return;
			logService.info("Start training..");
			if(!headless()) {
				dialog.setCurrentTaskMessage("Starting training ...");
				dialog.setTaskDone( 0 );
				dialog.setTaskStart( 1 );
			}

			RemainingTimeEstimator remainingTimeEstimator = new RemainingTimeEstimator();
			remainingTimeEstimator.setNumSteps(numEpochs);


			if(!headless()) {
				previewHandler = new PreviewHandler(context, trainDimensions);
				previewHandler.update(validation_data.get(0).getFirst(), validation_data.get(0).getFirst());
			}

			for (int i = 0; i < numEpochs; i++) {
				remainingTimeEstimator.setCurrentStep(i);
				String remainingTimeString = remainingTimeEstimator.getRemainingTimeString();
				logService.info("Epoch " + (i + 1) + "/" + numEpochs + " " + remainingTimeString);

				List<Double> losses = new ArrayList<>(stepsPerEpoch);

				for (int j = 0; j < stepsPerEpoch && !stopTraining; j++) {

					if (Thread.interrupted()) {
						tensorWeights.close();
						return;
					}

					if (index * trainBatchSize + trainBatchSize > X.size() - 1) {
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
					logStatusInConsole(j + 1, stepsPerEpoch, outputHandler);
					if(!headless()) dialog.updateTrainingProgress(i + 1, j + 1);

					index++;

				}

				if (Thread.interrupted()) return;
				training_data.on_epoch_end();
				outputHandler.saveCheckpoint(sess);
				float validationLoss = validate(sess, validation_data, tensorWeights);
				outputHandler.setCurrentValidationLoss(validationLoss);
				if(!headless()) dialog.updateTrainingChart(i + 1, losses, validationLoss);
				onEpochDoneCallbacks.forEach(callback -> callback.accept(this));
			}

			tensorWeights.close();

//			sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/control_dependency").run();

			if ( !headless() ) dialog.setTaskDone( 1 );
			logService.info( "Training done." );

//			if (inputs.size() > 0) uiService.show("inputs", Views.stack(inputs));
//			if (targets.size() > 0) uiService.show("targets", Views.stack(targets));

		}
	}

	private N2VDataWrapper<FloatType> makeTrainingData(double n2v_perc_pix) {
		long[] patchShapeData = new long[trainDimensions];
		Arrays.fill(patchShapeData, trainPatchDimLength);
		Dimensions patch_shape = new FinalDimensions(patchShapeData);

		return new N2VDataWrapper<>(X, trainBatchSize, n2v_perc_pix, patch_shape, neighborhoodRadius, N2VDataWrapper::uniform_withCP);
	}

	private List<Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>>> makeValidationData(double n2v_perc_pix) {
		int unet_n_depth = 2;
		int n_train = X.size();
		int n_val = validationX.size();

		double frac_val = (1.0 * n_val) / (n_train + n_val);
		double frac_warn = 0.05;
		if (frac_val < frac_warn) {
			logService.info("small number of validation images (only " + (100 * frac_val) + "% of all images)");
		}
		//        axes = axes_check_and_normalize('S'+self.config.axes,_X.ndim)
		//        ax = axes_dict(axes)
		int div_by = 2 * unet_n_depth;
		//        axes_relevant = ''.join(a for a in 'XYZT' if a in axes)
		long val_num_pix = 1;
		long train_num_pix = 1;
		//        val_patch_shape = ()

		long[] _val_patch_shape = new long[X.get(0).numDimensions() - 2];
		for (int i = 0; i < X.get(0).numDimensions() - 2; i++) {
			long n = X.get(0).dimension(i);
			val_num_pix *= validationX.get(0).dimension(i);
			train_num_pix *= X.get(0).dimension(i);
			_val_patch_shape[i] = validationX.get(0).dimension(i);
			if (n % div_by != 0) {
				System.err.println("training images must be evenly divisible by " + div_by
						+ "along axes XY (axis " + i + " has incompatible size " + n + ")");
			}
		}
		Dimensions val_patch_shape = FinalDimensions.wrap(_val_patch_shape);
		N2VDataWrapper<FloatType> valData = new N2VDataWrapper<>(validationX,
				(int) Math.min(trainBatchSize, validationX.size()),
				n2v_perc_pix, val_patch_shape, neighborhoodRadius,
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
		mean.set( opService.stats().mean( Views.iterable( Views.stack(X) ) ).getRealFloat() );
		stdDev.set( opService.stats().stdDev( Views.iterable( Views.stack(X) ) ).getRealFloat() );
		logService.info("mean: " + mean.get());
		logService.info("stdDev: " + stdDev.get());

		N2VUtils.normalize( X, mean, stdDev, opService );
		N2VUtils.normalize( validationX, mean, stdDev, opService );
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
		if(trainDimensions == 2) return new int[]{ 1, 2, 0, 3 };
		if(trainDimensions == 3) return new int[]{ 1, 2, 3, 0, 4 };
		return new int[0];
	}

	private boolean batchNumSufficient(int n_train) {
		if(trainBatchSize > n_train) {
			String errorMsg = "Not enough training data (" + n_train + " batches). At least " + trainBatchSize + " batches needed.";
			logService.error(errorMsg);
			stopTraining = true;
			dispose();
			uiService.showDialog(errorMsg, DialogPrompt.MessageType.ERROR_MESSAGE);
			return false;
		}
		return true;
	}

	private Tensor<Float> makeWeightsTensor() {
		float[] weightsdata = new float[trainBatchSize];
		Arrays.fill(weightsdata, 1);
		return Tensors.create(weightsdata);
	}

	private float validate(Session sess, List<Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>>> validationData, Tensor tensorWeights) {

		float avgLoss = 0;
		float avgAbs = 0;
		float avgMse = 0;

		long validationBatches = validationData.size();
		for (int i = 0; i < validationBatches; i++) {

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

	private void loadUntrainedGraph(Graph graph ) throws IOException {
		logService.info( "Import graph.." );
		String graphName = trainDimensions == 2 ? "graph_2d.pb" : "graph_3d.pb";
		byte[] graphDef = IOUtils.toByteArray( getClass().getResourceAsStream("/" + graphName) );
		graph.importGraphDef( graphDef );
//		graph.operations().forEachRemaining( op -> {
//			for ( int i = 0; i < op.numOutputs(); i++ ) {
//				Output< Object > opOutput = op.output( i );
//				String name = opOutput.op().name();
//				logService.info( name );
//			}
//		} );
	}

	private File loadTrainedGraph(Graph graph, File zipFile ) throws IOException {

		logService.info( "Import trained graph.." );

		File trainedModel = Files.createTempDirectory("n2v-imported-model").toFile();
		N2VUtils.unZipAll(zipFile, trainedModel);

		byte[] graphDef = new byte[ 0 ];
		try {
			graphDef = IOUtils.toByteArray( new FileInputStream(new File(trainedModel, "training_graph.pb")));
		} catch ( IOException e ) {
			e.printStackTrace();
		}
		graph.importGraphDef( graphDef );

//		graph.operations().forEachRemaining( op -> {
//			for ( int i = 0; i < op.numOutputs(); i++ ) {
//				Output< Object > opOutput = op.output( i );
//				String name = opOutput.op().name();
//				logService.info( name );
//			}
//		} );
		return trainedModel;
	}

	public void addTrainingAndValidationData(RandomAccessibleInterval<FloatType> training, double validationAmount) {
		if (Thread.interrupted()) return;

		logService.info( "Tile training and validation data.." );
		dialog.setCurrentTaskMessage("Tiling training and validation data" );

		List< RandomAccessibleInterval< FloatType > > tiles = N2VDataGenerator.createTiles( training, trainDimensions, trainBatchDimLength, logService );

		int trainEnd = (int) (tiles.size() * (1 - validationAmount));
		for (int i = 0; i < trainEnd; i++) {
			//TODO do I need to copy here?
			X.add( tiles.get( i ) );
		}
		int valEnd = tiles.size()-trainEnd % 2 == 1 ? tiles.size() - 1 : tiles.size();
		for (int i = trainEnd; i < valEnd; i++) {
			//TODO do I need to copy here?
			validationX.add( tiles.get( i ) );
		}
	}

	public void addTrainingAndValidationData(File trainingFolder, double validationAmount) {

		if(trainingFolder.isDirectory()) {
			File[] imgs = trainingFolder.listFiles();
			for (File file : imgs) {
				if (Thread.interrupted()) return;
				try {
					RandomAccessibleInterval img = datasetIOService.open(file.getAbsolutePath()).getImgPlus().getImg();
					addTrainingAndValidationData(convertToFloat(img), validationAmount);
				} catch (IOException e) {
					logService.warn("Could not load " + file.getAbsolutePath() + " as image");
				}
			}
		}
	}

	public static <T extends RealType<T>> RandomAccessibleInterval<FloatType> convertToFloat(RandomAccessibleInterval<T> img) {
		return Converters.convert(img, new RealFloatConverter<T>(), new FloatType());
	}

	public void addTrainingData(RandomAccessibleInterval<FloatType> training) {

		if (Thread.interrupted()) return;

		logService.info( "Tile training data.." );
		if(!headless()) dialog.setCurrentTaskMessage("Tiling training data" );

		logService.info("Training image dimensions: " + Arrays.toString(Intervals.dimensionsAsIntArray(training)));

		X.addAll(N2VDataGenerator.createTiles( training, trainDimensions, trainBatchDimLength, logService ));
	}

	public void addTrainingData(File trainingFolder) {

		if(trainingFolder.isDirectory()) {
			File[] imgs = trainingFolder.listFiles();
			for (File file : imgs) {
				if (Thread.interrupted()) return;
				try {
					RandomAccessibleInterval img = datasetIOService.open(file.getAbsolutePath()).getImgPlus().getImg();
					addTrainingData(convertToFloat(img));
				} catch (IOException e) {
					logService.warn("Could not load " + file.getAbsolutePath() + " as image");
				}
			}
		}
	}

	public void addValidationData(RandomAccessibleInterval<FloatType> validation) {

		if (Thread.interrupted()) return;

		logService.info( "Tile validation data.." );
		if(!headless()) dialog.setCurrentTaskMessage("Tiling validation data" );

		logService.info("Validation image dimensions: " + Arrays.toString(Intervals.dimensionsAsIntArray(validation)));

		validationX.addAll(N2VDataGenerator.createTiles( validation, trainDimensions, trainBatchDimLength, logService ));
	}

	public void addValidationData(File trainingFolder) {

		if(trainingFolder.isDirectory()) {
			File[] imgs = trainingFolder.listFiles();
			for (File file : imgs) {
				if (Thread.interrupted()) return;
				try {
					RandomAccessibleInterval img = datasetIOService.open(file.getAbsolutePath()).getImgPlus().getImg();
					addValidationData(convertToFloat(img));
				} catch (IOException e) {
					logService.warn("Could not load " + file.getAbsolutePath() + " as image");
				}
			}
		}
	}

	public void setStepsPerEpoch(final int steps) {
		stepsPerEpoch = steps;
	}

	public void setNumEpochs(final int numEpochs) {
		this.numEpochs = numEpochs;
	}

	public void setBatchSize(final int batchSize) {
		trainBatchSize = batchSize;
	}

	public void setPatchDimLength(final int patchDimLength) {
		trainPatchDimLength = patchDimLength;
	}

	public void setBatchDimLength(final int batchDimLength) {
		trainBatchDimLength = batchDimLength;
	}

	public N2VProgress getDialog() {
		return dialog;
	}

	public OutputHandler output() {
		return outputHandler;
	}

	public void setTrainDimensions(int trainDimensions) {
		this.trainDimensions = trainDimensions;
	}

	public void setNeighborhoodRadius(int radius) {
		this.neighborhoodRadius = radius;
	}

	public void stopTraining() {
		if(session != null) outputHandler.saveCheckpoint(session);
		stopTraining = true;
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
			pool.shutdownNow();
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
			n2v.setNumEpochs(5);
			n2v.setStepsPerEpoch(5);
			n2v.setBatchSize(128);
			n2v.setPatchDimLength(64);
			n2v.init();
			n2v.addTrainingAndValidationData(training, 0.1);
			n2v.train();
		} else
			System.out.println( "Cannot find training image " + trainingImgFile.getAbsolutePath() );

	}

}
