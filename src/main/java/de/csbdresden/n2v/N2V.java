package de.csbdresden.n2v;

import de.csbdresden.csbdeep.network.model.tensorflow.DatasetTensorFlowConverter;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import net.imagej.tensorflow.TensorFlowService;
import net.imglib2.Cursor;
import net.imglib2.Dimensions;
import net.imglib2.FinalDimensions;
import net.imglib2.FinalInterval;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import org.apache.commons.math3.util.Pair;
import org.scijava.Context;
import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.command.CommandModule;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Plugin(type = Command.class, menuPath = "Plugins>CSBDeep>N2V")
public class N2V implements Command {

	@Parameter
	private List<RandomAccessibleInterval<FloatType>> training;

	@Parameter
	private RandomAccessibleInterval<FloatType> prediction;

	@Parameter(type = ItemIO.OUTPUT)
	private RandomAccessibleInterval<FloatType> output;

	@Parameter(required = false)
	private File graphDefFile;

	@Parameter
	TensorFlowService tensorFlowService;

	@Parameter
	OpService opService;

	@Parameter
	UIService uiService;

	@Parameter
	Context context;

	private final String tensorXOpName = "input";
	private final String tensorYOpName = "activation_11_target";
	private final String kerasLearningOpName = "keras_learning_phase/input";
	private final String targetOpName = "activation_11/Identity";
	private final String sampleWeightsOpName = "activation_11_sample_weights";


	@Override
	public void run() {


		System.out.println("Load TensorFlow");
		tensorFlowService.loadLibrary();
		System.out.println(tensorFlowService.getStatus().getInfo());

//		final String checkpointDir = checkpoints.getAbsolutePath();
//		final boolean checkpointExists = Files.exists(Paths.get(checkpointDir));

		System.out.println("Create session");
		try (Graph graph = new Graph();
		     Session sess = new Session(graph)) {

			locateGraphDefFile();
			List<String> opNames = loadGraph(graph);

			printVariables(sess, opNames);

			List<RandomAccessibleInterval<FloatType>> tiles = createTiles(training);

			List<RandomAccessibleInterval<FloatType>> X = new ArrayList<>();
			List<RandomAccessibleInterval<FloatType>> validationX = new ArrayList<>();
			for (int i = 0; i < tiles.size() / 2; i++) {
				X.add(tiles.get(i));
			}
			int valEnd = tiles.size() / 2 % 2 == 1 ? tiles.size() - 1 : tiles.size();
			for (int i = tiles.size() / 2; i < valEnd; i++) {
				validationX.add(tiles.get(i));
			}

			train(sess, X, validationX);

//			runPrediction(prediction, sess, mapping);

			// Checkpoint.
			// The feed and target name are from the program that created the graph.
			// https://github.com/tensorflow/models/blob/master/samples/languages/java/training/model/create_graph.py.
//			sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/control_dependency").run();

//			// Example of "inference" in the same graph:
//			try (Tensor<Float> input = Tensors.create(1.0f);
//			     Tensor<Float> output =
//					     sess.runner().feed("input", input).fetch("output").run().get(0).expect(Float.class)) {
//				System.out.printf(
//						"For input %f, produced %f (ideally would produce 3*%f + 2)\n",
//						input.floatValue(), output.floatValue(), input.floatValue());
//			}
		}
	}

	private void locateGraphDefFile() {
		if (graphDefFile == null || !graphDefFile.exists()) {
			System.out.println("Loading graph def file from resources");
			try {
				graphDefFile = new File(getClass().getResource("/tf_model.pb").toURI());
			} catch (URISyntaxException e) {
				e.printStackTrace();
			}
		}
	}

	private void train(Session sess, List<RandomAccessibleInterval<FloatType>> _X, List<RandomAccessibleInterval<FloatType>> _validationX) {
		RandomAccessibleInterval<FloatType> X = Views.concatenate(2, _X);
		RandomAccessibleInterval<FloatType> validationX = Views.concatenate(2, _validationX);

		FloatType mean = new FloatType();
		mean.set(opService.stats().mean(Views.iterable(X)).getRealFloat());
		FloatType stdDev = new FloatType();
		stdDev.set(opService.stats().stdDev(Views.iterable(validationX)).getRealFloat());

		normalize(X, mean, stdDev);
		normalize(validationX, mean, stdDev);

		uiService.show("X", X);
		uiService.show("validationX", validationX);

		int[] mapping = {1, 2, 0, 3};
		Tensor XTensor = DatasetTensorFlowConverter.datasetToTensor(X, mapping);
		Tensor validationXTensor = DatasetTensorFlowConverter.datasetToTensor(validationX, mapping);

//			sig = MetaGraphDef.parseFrom(model.model().metaGraphDef()).getSignatureDefOrThrow(
//					DEFAULT_SERVING_SIGNATURE_DEF_KEY);


		int unet_n_depth = 2;
		int train_epochs = 2;
		int train_steps_per_epoch = 5;
		int train_batch_size = 16;
		double n2v_perc_pix = 1.6;

		int n_train = _X.size();
		int n_val = _validationX.size();

		double frac_val = (1.0 * n_val) / (n_train + n_val);
		double frac_warn = 0.05;
		if (frac_val < frac_warn) {
			System.out.println("small number of validation images (only " + (100 * frac_val) + "% of all images)");
		}
//        axes = axes_check_and_normalize('S'+self.config.axes,X.ndim)
//        ax = axes_dict(axes)
		int div_by = 2 * unet_n_depth;
//        axes_relevant = ''.join(a for a in 'XYZT' if a in axes)
		long val_num_pix = 1;
		long train_num_pix = 1;
//        val_patch_shape = ()
		long[] _val_patch_shape = new long[XTensor.numDimensions()-2];
		for (int i = 1; i < XTensor.shape().length - 1; i++) {
			long n = XTensor.shape()[i];
			val_num_pix *= validationXTensor.shape()[i];
			train_num_pix *= XTensor.shape()[i];
			_val_patch_shape[i-1] = validationXTensor.shape()[i];
			if (n % div_by != 0)
				System.err.println("training images must be evenly divisible by " + div_by
						+ "along axes XY (axis " + i + " has incompatible size " + n + ")");
		}
		Dimensions val_patch_shape = FinalDimensions.wrap(_val_patch_shape);

		int epochs = train_epochs;
		int steps_per_epoch = train_steps_per_epoch;
		long[] targetDims = Intervals.dimensionsAsLongArray(X);
		targetDims[targetDims.length - 1]++;

		RandomAccessibleInterval<FloatType> validationY = opService.create().img(new FinalInterval(targetDims), X.randomAccess().get().copy());
		N2VUtils.manipulate_val_data(validationX, validationY, n2v_perc_pix, val_patch_shape);

		uiService.show("validationY", validationY);

//        if not self._model_prepared:
//            self.prepare_for_training()

//        manipulator = eval('pm_{0}({1})'.format(self.config.n2v_manipulator, str(self.config.n2v_neighborhood_radius)))
//
//        # Here we prepare the Noise2Void data. Our input is the noisy data X and as target we take X concatenated with
//        # a masking channel. The N2V_DataWrapper will take care of the pixel masking and manipulating.

		Img<FloatType> target = opService.create().img(new FinalInterval(targetDims), X.randomAccess().get().copy());
		Cursor<FloatType> trainingCursor = Views.iterable(X).localizingCursor();
		RandomAccess<FloatType> targetRA = target.randomAccess();
		while (trainingCursor.hasNext()) {
			trainingCursor.next();
			targetRA.setPosition(trainingCursor);
			targetRA.get().set(trainingCursor.get());
		}

		N2V_DataWrapper<FloatType> training_data = new N2V_DataWrapper<>(context, X,
				target, train_batch_size,
				n2v_perc_pix, val_patch_shape,
				N2V_DataWrapper::uniform_withCP);
//        training_data = N2V_DataWrapper(X, np.concatenate((X, np.zeros(X.shape, dtype=X.dtype)), axis=axes.index('C')),
//                                                    self.config.train_batch_size, int(train_num_pix/100 * self.config.n2v_perc_pix),
//                                                    self.config.n2v_patch_shape, manipulator)
//
//        # validation_Y is also validation_X plus a concatenated masking channel.
//        # To speed things up, we precompute the masking vo the validation data.
//        validation_Y = np.concatenate((validation_X, np.zeros(validation_X.shape, dtype=validation_X.dtype)), axis=axes.index('C'))
//        n2v_utils.manipulate_val_data(validation_X, validation_Y,
//                                                        num_pix=int(val_num_pix/100 * self.config.n2v_perc_pix),
//                                                        shape=val_patch_shape,
//                                                        value_manipulation=manipulator)


		int index = 0;
		List<RandomAccessibleInterval<FloatType>> inputs = new ArrayList<>();
		List<RandomAccessibleInterval<FloatType>> targets = new ArrayList<>();
		for (int i = 0; i < epochs; i++) {
			for (int j = 0; j < steps_per_epoch; j++) {

				Pair<RandomAccessibleInterval, RandomAccessibleInterval> item = training_data.getItem(index);
				inputs.add(item.getFirst());
				targets.add(item.getSecond());

				Tensor tensorX = DatasetTensorFlowConverter.datasetToTensor(item.getFirst(), mapping);
				Tensor tensorY = DatasetTensorFlowConverter.datasetToTensor(item.getSecond(), mapping);

//				System.out.println("input tensorX: " + Arrays.toString(tensorX.shape()));
//				long[] min = new long[tile.numDimensions()];
//				long[] max = new long[tile.numDimensions()];
//				tile.min(min);
//				tile.max(max);
//				System.out.println("tile: " + Arrays.toString(min) + " -> " + Arrays.toString(max));

				Session.Runner runner = sess.runner();

				runner.feed(tensorXOpName, tensorX)
						.feed(tensorYOpName, tensorY)
						.feed(kerasLearningOpName, Tensors.create(true))
						.addTarget(targetOpName);
				runner.fetch("loss/activation_11_loss/truediv");
//					opNames.forEach(runner::fetch);
				List<Tensor<?>> loss = runner.run();
				System.out.println(loss.get(0).floatValue());

//					printVariables(sess, opNames);
				index++;
			}
			training_data.on_epoch_end();
		}

		uiService.show("inputs", Views.stack(inputs));
		uiService.show("targets", Views.stack(targets));
	}

	private List<RandomAccessibleInterval<FloatType>> createTiles(List<RandomAccessibleInterval<FloatType>> training) {
		List<RandomAccessibleInterval<FloatType>> tiles = new ArrayList<>();
		for (RandomAccessibleInterval<FloatType> trainingImg : training) {
//				tiles.addAll(createTiles(Views.interval(trainingImg, new long[]{0,0}, new long[]{640, 640})));
			tiles.addAll(createTiles(trainingImg));
		}
		return tiles;
	}

	private void runPrediction(RandomAccessibleInterval<FloatType> inputRAI, Session sess, int[] mapping) {
		RandomAccessibleInterval predictionInput = Views.interval(inputRAI, new long[]{0, 0}, new long[]{1687, 2495});
		predictionInput = Views.addDimension(predictionInput, 0, 0);
		predictionInput = Views.addDimension(predictionInput, 0, 0);

		Tensor inputTensor = DatasetTensorFlowConverter.datasetToTensor(predictionInput, mapping);

		Tensor outputTensor = sess.runner()
				.feed(tensorXOpName, inputTensor)
				.feed(kerasLearningOpName, Tensors.create(false))
				.fetch(targetOpName)
				.run().get(0);
		output = DatasetTensorFlowConverter.tensorToDataset(outputTensor, new FloatType(), mapping, false);
	}

	private List<String> loadGraph(Graph graph) {
		System.out.println("Import graph..");
		byte[] graphDef = new byte[0];
		try {
			graphDef = Files.readAllBytes(graphDefFile.toPath());
		} catch (IOException e) {
			e.printStackTrace();
		}
		graph.importGraphDef(graphDef);

		List<String> opNames = new ArrayList<>();
		graph.operations().forEachRemaining(op -> {
			for (int i = 0; i < op.numOutputs(); i++) {
				Output<Object> opOutput = op.output(i);
				String name = opOutput.op().name();
				if (name.equals(tensorYOpName)) continue;
				if (name.equals(kerasLearningOpName)) continue;
				if (name.equals(tensorXOpName)) continue;
				if (name.equals("activation_11_sample_weights")) continue;
				opNames.add(name);
				System.out.println(name);
			}
//			if(op.numOutputs()!= 0 && !op.name().equals("input")) opNames.add(op.name());
		});
//			opNames.add("loss");
//			System.out.println(opNames);

		// Initialize or restore.
		// The names of the tensors in the graph are printed out by the program
		// that created the graph:
		// https://github.com/tensorflow/models/blob/master/samples/languages/java/training/model/create_graph.py
//			if (checkpointExists) {
//				sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/restore_all").run();
//			} else {
//				sess.runner().addTarget("init").run();
//			}
//			System.out.print("Starting from       : ");
		return opNames;
	}

	private List<RandomAccessibleInterval<FloatType>> createTiles(RandomAccessibleInterval<FloatType> inputRAI) {
		System.out.println("Create tiles..");
		List<RandomAccessibleInterval<FloatType>> data = new ArrayList<>();
		data.add(inputRAI);
		List<RandomAccessibleInterval<FloatType>> tiles = N2VDataGenerator.generatePatchesFromList(
				data,
				new FinalInterval(64, 64));
		long[] tiledim = new long[tiles.get(0).numDimensions()];
		tiles.get(0).dimensions(tiledim);
		System.out.println("Generated " + tiles.size() + " tiles of shape " + Arrays.toString(tiledim));

		RandomAccessibleInterval<FloatType> tilesStack = Views.stack(tiles);
		uiService.show("tiles", tilesStack);
		return tiles;
	}

	private void normalize(RandomAccessibleInterval<FloatType> input, FloatType mean, FloatType stdDev) {
		IterableInterval<FloatType> rai = opService.math().subtract(Views.iterable(input), mean);
		opService.math().divide(rai, stdDev);
		opService.copy().iterableInterval(Views.iterable(input), rai);
	}

	private static void printVariables(Session sess, List<String> opNames) {
//		Session.Runner runner = sess.runner();
//		runner.fetch("loss/activation_11_loss/mul");
//		List<Tensor<?>> values = runner.run();
//		System.out.println(values.get(0).intValue());
//		for (int i = 0, opNamesSize = opNames.size(); i < opNamesSize; i++) {
//			String name = opNames.get(i);
//			System.out.println(name + ": " + values.get(i).floatValue());
//		}
//		for (Tensor<?> t : values) {
//			t.close();
//		}
	}

	public static void main(final String... args) throws Exception {

		final ImageJ ij = new ImageJ();

		ij.launch(args);

//		ij.log().setLevel(LogLevel.TRACE);

		// ask the user for a file to open
//		final File file = ij.ui().chooseFile(null, "open");
//		final File file = new File("/home/random/Development/imagej/project/CSBDeep/data/N2V/pred_train.tif");
		final File file = new File(N2V.class.getResource("/train.tif").toURI());

		if (file.exists()) {
			// load the dataset
//			final Dataset dataset = ij.scifio().datasetIO().open(file
//					.getAbsolutePath());
			RandomAccessibleInterval _input = (RandomAccessibleInterval) ij.io().open(file.getAbsolutePath());
			RandomAccessibleInterval _inputConverted = ij.op().convert().float32(Views.iterable(_input));

			List<RandomAccessibleInterval<FloatType>> inputs = new ArrayList<>();
			inputs.add(ij.op().copy().rai(_inputConverted));
			RandomAccessibleInterval prediction = ij.op().copy().rai(_inputConverted);
			// show the image
//			ij.ui().show(dataset);

			// invoke the plugin
			CommandModule plugin = ij.command().run(N2V.class, false,
					"training", inputs, "prediction", prediction).get();
			ij.ui().show(plugin.getOutput("output"));
//			ij.context().dispose();
		}

	}
}
