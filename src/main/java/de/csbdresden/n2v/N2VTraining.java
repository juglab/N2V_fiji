package de.csbdresden.n2v;

import de.csbdresden.csbdeep.network.model.tensorflow.DatasetTensorFlowConverter;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import net.imagej.tensorflow.TensorFlowService;
import net.imglib2.Cursor;
import net.imglib2.Dimensions;
import net.imglib2.FinalDimensions;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import org.apache.commons.compress.utils.IOUtils;
import org.apache.commons.math3.util.Pair;
import org.scijava.Context;
import org.scijava.display.Display;
import org.scijava.display.DisplayService;
import org.scijava.plugin.Parameter;
import org.scijava.ui.UIService;
import org.scijava.util.FileUtils;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class N2VTraining {

	private File graphDefFile;

	@Parameter
	private TensorFlowService tensorFlowService;

	@Parameter
	private OpService opService;

	@Parameter
	private UIService uiService;

	@Parameter
	private Context context;

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

	int[] mapping = { 1, 2, 0, 3 };
	private boolean checkpointExists;
	private Tensor< String > checkpointPrefix;
	private FloatType mean;
	private FloatType stdDev;

//	private boolean saveCheckpoints = true;
	private File modelDir;

	private N2VDialog dialog;

	private final List< RandomAccessibleInterval< FloatType > > X = new ArrayList<>();
	private final List< RandomAccessibleInterval< FloatType > > validationX = new ArrayList<>();
	private int numEpochs = 300;
	private int trainBatchSize = 180;
	private int trainBatchDimLength = 180;
	private int trainPatchDimLength = 60;
	private int stepsPerEpoch = 200;

	private boolean stopTraining = false;
	private RandomAccessibleInterval<FloatType> splitImage;
	private List<RandomAccessibleInterval<FloatType>> historyImages;
	private boolean noCheckpointSaved = true;

	public N2VTraining(Context context) {
		context.inject(this);
	}

	public void init() {

		//TODO GUI open window for status, indicate that preprocessing is starting
		dialog = new N2VDialog(this);
		dialog.updateProgressText("Loading TensorFlow" );

		System.out.println( "Load TensorFlow.." );
		tensorFlowService.loadLibrary();
		System.out.println( tensorFlowService.getStatus().getInfo() );

	}

	public void train() {

		if(stopTraining) return;

		System.out.println( "Create session.." );
		dialog.updateProgressText("Creating session" );
		try (Graph graph = new Graph();
		     Session sess = new Session( graph )) {

			loadGraph(graph);

			if (checkpointExists) {
				sess.runner()
						.feed("save/Const", checkpointPrefix)
						.addTarget("save/restore_all").run();
			} else {
				sess.runner().addTarget("init").run();
			}

			if(stopTraining) return;

			System.out.println("Prepare data for training..");
			dialog.updateProgressText("Preparing data for training");

			RandomAccessibleInterval<FloatType> _X = Views.concatenate(2, X);
			RandomAccessibleInterval<FloatType> _validationX = Views.concatenate(2, validationX);

			if(stopTraining) return;

			//		uiService.show("_X", _X);
			//		uiService.show("_validationX", _validationX);

			int unet_n_depth = 2;
			double n2v_perc_pix = 1.6;

			int n_train = X.size();
			int n_val = validationX.size();

			double frac_val = (1.0 * n_val) / (n_train + n_val);
			double frac_warn = 0.05;
			if (frac_val < frac_warn) {
				System.out.println("small number of validation images (only " + (100 * frac_val) + "% of all images)");
			}
			//        axes = axes_check_and_normalize('S'+self.config.axes,_X.ndim)
			//        ax = axes_dict(axes)
			int div_by = 2 * unet_n_depth;
			//        axes_relevant = ''.join(a for a in 'XYZT' if a in axes)
			long val_num_pix = 1;
			long train_num_pix = 1;
			//        val_patch_shape = ()

			long[] _val_patch_shape = new long[_X.numDimensions() - 2];
			for (int i = 0; i < _X.numDimensions() - 2; i++) {
				long n = _X.dimension(i);
				val_num_pix *= _validationX.dimension(i);
				train_num_pix *= _X.dimension(i);
				_val_patch_shape[i] = _validationX.dimension(i);
				if (n % div_by != 0) {
					System.err.println("training images must be evenly divisible by " + div_by
							+ "along axes XY (axis " + i + " has incompatible size " + n + ")");
				}
			}
			Dimensions val_patch_shape = FinalDimensions.wrap(_val_patch_shape);
			Dimensions patch_shape = new FinalDimensions(trainPatchDimLength, trainPatchDimLength);

//			int stepsPerEpoch = n_train / trainBatchSize;
			long[] targetDims = Intervals.dimensionsAsLongArray(_X);
			targetDims[targetDims.length - 1]++;

			//		uiService.show("validationY", validationY);

			Img<FloatType> target = makeTarget(_X, targetDims);

			N2VDataWrapper<FloatType> training_data = new N2VDataWrapper<>(context, _X, target, trainBatchSize, n2v_perc_pix, patch_shape, N2VDataWrapper::uniform_withCP);

			if(stopTraining) return;

			Img<FloatType> validationTarget = makeTarget(_validationX, targetDims);

			N2VDataWrapper<FloatType> validation_data = new N2VDataWrapper<>(context, _validationX,
					validationTarget, (int) Math.min(trainBatchSize, _validationX.dimension(2)),
					n2v_perc_pix, val_patch_shape,
					N2VDataWrapper::uniform_withCP);

			int index = 0;
//			List<RandomAccessibleInterval<FloatType>> inputs = new ArrayList<>();
//			List<RandomAccessibleInterval<FloatType>> targets = new ArrayList<>();
			float[] weightsdata = new float[trainBatchSize];
			for (int i1 = 0; i1 < weightsdata.length; i1++) {
				weightsdata[i1] = 1;
			}
			Tensor<Float> tensorWeights = Tensors.create(weightsdata);

			if(stopTraining) return;

			//TODO GUI - display time estimate until training is done - each step should take roughly the same time

			System.out.println("Start training..");
			dialog.updateProgressText("Starting training ...");

			// Create dialog
			dialog.initChart(numEpochs, stepsPerEpoch);
			List<Double> losses = null;

			for (int i = 0; i < numEpochs && !stopTraining; i++) {
				System.out.println("Epoch " + (i + 1) + "/" + numEpochs);

				float loss = 0;
				losses = new ArrayList<>(stepsPerEpoch);

				for (int j = 0; j < stepsPerEpoch && !stopTraining; j++) {

					if (index * trainBatchSize + trainBatchSize > n_train - 1) {
						index = 0;
						System.out.println("starting with index 0 of training batches");
					}

					Pair<RandomAccessibleInterval, RandomAccessibleInterval> item = training_data.getItem(index);
//					inputs.add( item.getFirst() );
//					targets.add( item.getSecond() );
//					uiService.show("input", item.getFirst());
//					uiService.show("target", item.getSecond());

					Tensor tensorX = DatasetTensorFlowConverter.datasetToTensor(item.getFirst(), mapping);
					Tensor tensorY = DatasetTensorFlowConverter.datasetToTensor(item.getSecond(), mapping);

					Session.Runner runner = sess.runner();

					runner.feed(tensorXOpName, tensorX).feed(tensorYOpName, tensorY)
							.feed(learningPhaseOpName, Tensors.create(true))
							.feed(sampleWeightsOpName, tensorWeights).addTarget(trainingTargetOpName);
					runner.fetch(lossOpName);
					runner.fetch(absOpName);
					runner.fetch(mseOpName);

					List<Tensor<?>> fetchedTensors = runner.run();
					loss = fetchedTensors.get(0).floatValue();
					losses.add((double) loss);
					float abs = fetchedTensors.get(1).floatValue();
					float mse = fetchedTensors.get(2).floatValue();

					fetchedTensors.forEach(tensor -> tensor.close());
					tensorX.close();
					tensorY.close();

					progressPercentage(j + 1, stepsPerEpoch, loss, abs, mse);
					dialog.updateProgress(i + 1, j + 1);

					//TODO GUI - update progress bar indicating the step of the current epoch
					index++;

				}
				training_data.on_epoch_end();

				if(stopTraining && noCheckpointSaved) return;
				sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/control_dependency").run();
				noCheckpointSaved = false;

				float validationLoss = validate(sess, validation_data, tensorWeights);

				dialog.updateChart(i + 1, losses, validationLoss);

			}

//			sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/control_dependency").run();

			dialog.updateProgressText("Training done.");
			System.out.println("Training done.");

//			if (inputs.size() > 0) uiService.show("inputs", Views.stack(inputs));
//			if (targets.size() > 0) uiService.show("targets", Views.stack(targets));

		}
	}

	public boolean cancelTraining() {
		stopTraining = true;
		if(noCheckpointSaved) dialog.dispose();
		return true;
	}

	private float validate(Session sess, N2VDataWrapper validationData, Tensor tensorWeights) {

		Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> item = validationData.getItem(0);

		Tensor tensorX = DatasetTensorFlowConverter.datasetToTensor(item.getFirst(), mapping);
		Tensor tensorY = DatasetTensorFlowConverter.datasetToTensor(item.getSecond(), mapping);

		Session.Runner runner = sess.runner();

		runner.feed(tensorXOpName, tensorX)
				.feed(tensorYOpName, tensorY)
				.feed(learningPhaseOpName, Tensors.create(true))
				.feed(sampleWeightsOpName, tensorWeights)
				.fetch(predictionTargetOpName)
				.addTarget(validationTargetOpName);
		runner.fetch(lossOpName);
		runner.fetch(absOpName);
		runner.fetch(mseOpName);

		List<Tensor<?>> fetchedTensors = runner.run();
		Tensor outputTensor = fetchedTensors.get(0);
		RandomAccessibleInterval<FloatType> output = DatasetTensorFlowConverter.tensorToDataset(outputTensor, new FloatType(), mapping, false);

		updateSplitImage(item.getFirst(), output);
		updateHistoryImage(output);

		float loss = fetchedTensors.get(1).floatValue();
		float abs = fetchedTensors.get(2).floatValue();
		float mse = fetchedTensors.get(3).floatValue();

		fetchedTensors.forEach(tensor -> tensor.close());
		tensorX.close();
		tensorY.close();
		System.out.println("\nValidation loss: " + loss + " abs: " + abs + " mse: " + mse);
		return loss;
	}

	private void updateSplitImage(RandomAccessibleInterval<FloatType> in, RandomAccessibleInterval<FloatType> out) {
		if(splitImage == null) splitImage = opService.copy().rai(out);
		else opService.copy().rai(splitImage, out);
		RandomAccess<FloatType> inRA = in.randomAccess();
		RandomAccess<FloatType> splitRA = splitImage.randomAccess();
		for (int i = 0; i < in.dimension(0); i++) {
			for (int j = 0; j < in.dimension(1); j++) {
				if(i < in.dimension(1)-j) {
					inRA.setPosition(i, 0);
					inRA.setPosition(j, 1);
					for (int k = 0; k < in.dimension(2); k++) {
						inRA.setPosition(k, 2);
						splitRA.setPosition(inRA);
						splitRA.get().set(inRA.get());
					}
				}
			}
		}
		Display<?> display = uiService.context().service(DisplayService.class).getDisplay("training preview");
		if(display == null) uiService.show("training preview", splitImage);
		else display.update();
	}

	private void updateHistoryImage(RandomAccessibleInterval<FloatType> out) {
		//TODO copying neccessary?
		for (int i = 2; i < out.numDimensions(); i++) {
			out = Views.hyperSlice(out, i, 0);
		}
		RandomAccessibleInterval<FloatType> outXY = opService.copy().rai(out);
		if(historyImages == null) historyImages = new ArrayList<>();
		historyImages.add(0, outXY);
		Display<?> display = uiService.context().service(DisplayService.class).getDisplay("training history");
		RandomAccessibleInterval<FloatType> stack = Views.stack(historyImages);
		System.out.println(Arrays.toString(Intervals.dimensionsAsIntArray(stack)));
		if(display == null) uiService.show("training history", stack);
		else {
			display.clear();
			display.display(stack);
			display.update();
		}
	}

	public static void progressPercentage( int step, int stepTotal, float loss, float abs, float mse ) {
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
		System.out.printf( "%d / %d %s%s - loss: %f mse: %f abs: %f\n", step, stepTotal, bareDone, bareRemain, loss, mse, abs );
	}

	private Img< FloatType > makeTarget( RandomAccessibleInterval< FloatType > X, long[] targetDims ) {
		Img< FloatType > target = opService.create().img( new FinalInterval( targetDims ), X.randomAccess().get().copy() );
		Cursor< FloatType > trainingCursor = Views.iterable( X ).localizingCursor();
		RandomAccess< FloatType > targetRA = target.randomAccess();
		while ( trainingCursor.hasNext() ) {
			trainingCursor.next();
			targetRA.setPosition( trainingCursor );
			targetRA.get().set( trainingCursor.get() );
		}
		return target;
	}

	private List< RandomAccessibleInterval< FloatType > > normalizeAndTile( RandomAccessibleInterval< FloatType > img ) {
		if(mean == null) {
			// calculate mean and stddev on first image, use these values for additional images
			mean = new FloatType();
			mean.set( opService.stats().mean( Views.iterable( img ) ).getRealFloat() );
			stdDev = new FloatType();
			stdDev.set( opService.stats().stdDev( Views.iterable( img ) ).getRealFloat() );
		}

		List< RandomAccessibleInterval< FloatType > > tiles = new ArrayList<>();
		tiles.addAll( createTiles( N2VUtils.normalize( img, mean, stdDev, opService ) ) );
		return tiles;
	}

	private void loadGraph(Graph graph ) {
		System.out.println( "Import graph.." );
		byte[] graphDef = new byte[ 0 ];
		try {
			graphDef = IOUtils.toByteArray( getClass().getResourceAsStream( "/graph.pb" ) );
		} catch ( IOException e ) {
			e.printStackTrace();
		}
		graph.importGraphDef( graphDef );

		graph.operations().forEachRemaining( op -> {
			for ( int i = 0; i < op.numOutputs(); i++ ) {
				Output< Object > opOutput = op.output( i );
				String name = opOutput.op().name();
//				System.out.println( name );
			}
		} );

		Operation opTrain = graph.operation( trainingTargetOpName );
		if ( opTrain == null ) throw new RuntimeException( "Training op not found" );

		String checkpointDir = "";
		try {
			checkpointDir = Files.createTempDirectory("n2v-checkpoints").toAbsolutePath().toString() + File.separator + "variables";
			byte[] predictionGraphDef = IOUtils.toByteArray( getClass().getResourceAsStream( "/prediction/saved_model.pb" ) );
			FileUtils.writeFile(new File(new File(checkpointDir).getParentFile(), "saved_model.pb"), predictionGraphDef);
		} catch (IOException e) {
			e.printStackTrace();
		}
		checkpointExists = false;
		checkpointPrefix =
				Tensors.create(Paths.get(checkpointDir, "variables").toString());
		modelDir = new File(checkpointDir).getParentFile();

	}

	private List< RandomAccessibleInterval< FloatType > > createTiles( RandomAccessibleInterval< FloatType > inputRAI ) {
		System.out.println( "Create tiles.." );
		List< RandomAccessibleInterval< FloatType > > data = new ArrayList<>();
		data.add( inputRAI );
		List< RandomAccessibleInterval< FloatType > > tiles = N2VDataGenerator.generateBatchesFromList(
				data,
				new FinalInterval( trainBatchDimLength, trainBatchDimLength ) );
		long[] tiledim = new long[ tiles.get( 0 ).numDimensions() ];
		tiles.get( 0 ).dimensions( tiledim );
		System.out.println( "Generated " + tiles.size() + " tiles of shape " + Arrays.toString( tiledim ) );

//		RandomAccessibleInterval<FloatType> tilesStack = Views.stack(tiles);
//		uiService.show("tiles", tilesStack);
		return tiles;
	}

	public void addTrainingAndValidationData(RandomAccessibleInterval<FloatType> training, double validationAmount) {

		if(stopTraining) return;

		System.out.println( "Normalize and tile training and validation data.." );
		dialog.updateProgressText("Normalizing and tiling training and validation data" );

		List< RandomAccessibleInterval< FloatType > > tiles = normalizeAndTile( training );

		int trainEnd = (int) (tiles.size() * (1 - validationAmount));
		for (int i = 0; i < trainEnd; i++) {
			//TODO do I need to copy here?
			X.add( opService.copy().rai( tiles.get( i ) ) );
		}
		int valEnd = tiles.size()-trainEnd % 2 == 1 ? tiles.size() - 1 : tiles.size();
		for (int i = trainEnd; i < valEnd; i++) {
			//TODO do I need to copy here?
			validationX.add( opService.copy().rai( tiles.get( i ) ) );
		}
	}

	public void addTrainingData(RandomAccessibleInterval<FloatType> training) {

		if(stopTraining) return;

		System.out.println( "Normalize and tile training data.." );
		dialog.updateProgressText("Normalizing and tiling training data" );

		System.out.println("Training image dimensions: " + Arrays.toString(Intervals.dimensionsAsIntArray(training)));

		List< RandomAccessibleInterval< FloatType > > tiles = normalizeAndTile( training );

		//TODO do I need to copy here?
		tiles.forEach(tile -> X.add( opService.copy().rai( tile ) ));
	}

	public void addValidationData(RandomAccessibleInterval<FloatType> validation) {

		if(stopTraining) return;

		System.out.println( "Normalize and tile validation data.." );
		dialog.updateProgressText("Normalizing and tiling validation data" );

		System.out.println("Validation image dimensions: " + Arrays.toString(Intervals.dimensionsAsIntArray(validation)));

		List< RandomAccessibleInterval< FloatType > > tiles = normalizeAndTile( validation );

		//TODO do I need to copy here?
		tiles.forEach(tile -> validationX.add( opService.copy().rai( tile ) ));
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
			n2v.init();
			n2v.setNumEpochs(5);
			n2v.setStepsPerEpoch(5);
			n2v.setBatchSize(128);
			n2v.setPatchDimLength(64);
			n2v.addTrainingAndValidationData(training, 0.1);
			n2v.train();
		} else
			System.out.println( "Cannot find training image " + trainingImgFile.getAbsolutePath() );

	}

	public File exportTrainedModel() throws IOException {
		if(noCheckpointSaved) return null;
		return N2VUtils.saveTrainedModel(modelDir);
	}

	public FloatType getMean() {
		return mean;
	}

	public FloatType getStdDev() {
		return stdDev;
	}

	public boolean isCanceled() {
		return stopTraining;
	}
}
