package de.csbdresden.n2v;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.compress.utils.IOUtils;
import org.apache.commons.math3.util.Pair;
import org.scijava.Context;
import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.command.CommandModule;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

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

@Plugin( type = Command.class, menuPath = "Plugins>CSBDeep>N2V" )
public class N2V implements Command {

	@Parameter
	//TODO make this a list in the future to support multiple training images
//	private List<RandomAccessibleInterval<FloatType>> training;
	private RandomAccessibleInterval< FloatType > training;

	@Parameter
	private RandomAccessibleInterval< FloatType > prediction;

	@Parameter( type = ItemIO.OUTPUT )
	private RandomAccessibleInterval< FloatType > output;

//	@Parameter
	private boolean useDefaultGraph = true;

//	@Parameter(required = false)
	private File graphDefFile;

	@Parameter
	private int train_epochs = 10;

	private int train_batch_size = 128;

//	@Parameter
//	int train_steps_per_epoch = 20;

	@Parameter
	private TensorFlowService tensorFlowService;

	@Parameter
	private OpService opService;

	@Parameter
	private UIService uiService;

	@Parameter
	private Context context;

	private final String tensorXOpName = "input";
	private final String tensorYOpName = "activation_11_target";
	private final String kerasLearningOpName = "keras_learning_phase/input";
	private final String trainingTargetOpName = "train";
	private final String predictionTargetOpName = "activation_11/Identity";
	private final String sampleWeightsOpName = "activation_11_sample_weights";

	private ArrayList< String > opNames;

	int[] mapping = { 1, 2, 0, 3 };
	private boolean checkpointExists;
	private Tensor< String > checkpointPrefix;
	private FloatType mean;
	private FloatType stdDev;

	//TODO make work, almost there
	private boolean saveCheckpoints = true;

	@Override
	public void run() {

		//TODO GUI open window for status, indicate that preprocessing is starting

		System.out.println( "Load TensorFlow.." );
		tensorFlowService.loadLibrary();
		System.out.println( tensorFlowService.getStatus().getInfo() );

		System.out.println( "Create session.." );
		try (Graph graph = new Graph();
				Session sess = new Session( graph )) {

//			locateGraphDefFile();
			loadGraph( graph );

//			if(saveCheckpoints) {
			if ( checkpointExists ) {
				sess.runner().feed( "save/Const", checkpointPrefix ).addTarget( "save/restore_all" ).run();
			} else {
				sess.runner().addTarget( "init" ).run();
			}
//			}

			System.out.println( "Normalize and tile training data.." );

			List< RandomAccessibleInterval< FloatType > > tiles = normalizeAndTile( training );

			List< RandomAccessibleInterval< FloatType > > X = new ArrayList<>();
			List< RandomAccessibleInterval< FloatType > > validationX = new ArrayList<>();
			for ( int i = 0; i < tiles.size() / 2; i++ ) {
				//TODO do I need to copy here?
				X.add( opService.copy().rai( tiles.get( i ) ) );
			}
			int valEnd = tiles.size() / 2 % 2 == 1 ? tiles.size() - 1 : tiles.size();
			for ( int i = tiles.size() / 2; i < valEnd; i++ ) {
				//TODO do I need to copy here?
				validationX.add( opService.copy().rai( tiles.get( i ) ) );
			}

			train( sess, X, validationX );

			//TODO GUI - indicate training is done, indicate prediction calculation is starting

			runPrediction( prediction, sess, mapping );

		}
	}

	private void locateGraphDefFile() {
		if ( useDefaultGraph || graphDefFile == null || !graphDefFile.exists() ) {
			System.out.println( "Loading graph def file from resources" );
			try {
				graphDefFile = new File( getClass().getResource( "/graph.pb" ).toURI() );
			} catch ( URISyntaxException e ) {
				e.printStackTrace();
			}
		}
	}

	private void train( Session sess, List< RandomAccessibleInterval< FloatType > > _X, List< RandomAccessibleInterval< FloatType > > _validationX ) {

		System.out.println( "Prepare data for training.." );

		RandomAccessibleInterval< FloatType > X = Views.concatenate( 2, _X );
		RandomAccessibleInterval< FloatType > validationX = Views.concatenate( 2, _validationX );

		uiService.show( "X", X );
//		uiService.show("validationX", validationX);

		Tensor XTensor = DatasetTensorFlowConverter.datasetToTensor( X, mapping );
		Tensor validationXTensor = DatasetTensorFlowConverter.datasetToTensor( validationX, mapping );

		int unet_n_depth = 2;
		double n2v_perc_pix = 1.6;

		int n_train = _X.size();
		int n_val = _validationX.size();

		double frac_val = ( 1.0 * n_val ) / ( n_train + n_val );
		double frac_warn = 0.05;
		if ( frac_val < frac_warn ) {
			System.out.println( "small number of validation images (only " + ( 100 * frac_val ) + "% of all images)" );
		}
//        axes = axes_check_and_normalize('S'+self.config.axes,X.ndim)
//        ax = axes_dict(axes)
		int div_by = 2 * unet_n_depth;
//        axes_relevant = ''.join(a for a in 'XYZT' if a in axes)
		long val_num_pix = 1;
		long train_num_pix = 1;
//        val_patch_shape = ()
		long[] _val_patch_shape = new long[ XTensor.numDimensions() - 2 ];
		for ( int i = 1; i < XTensor.shape().length - 1; i++ ) {
			long n = XTensor.shape()[ i ];
			val_num_pix *= validationXTensor.shape()[ i ];
			train_num_pix *= XTensor.shape()[ i ];
			_val_patch_shape[ i - 1 ] = validationXTensor.shape()[ i ];
			if ( n % div_by != 0 )
				System.err.println( "training images must be evenly divisible by " + div_by + "along axes XY (axis " + i + " has incompatible size " + n + ")" );
		}
		Dimensions val_patch_shape = FinalDimensions.wrap( _val_patch_shape );

		int epochs = train_epochs;
//		int steps_per_epoch = train_steps_per_epoch;
		int steps_per_epoch = n_train / train_batch_size;
		long[] targetDims = Intervals.dimensionsAsLongArray( X );
		targetDims[ targetDims.length - 1 ]++;

		RandomAccessibleInterval< FloatType > validationY = opService.create().img( new FinalInterval( targetDims ), X.randomAccess().get().copy() );
		N2VUtils.manipulate_val_data( validationX, validationY, n2v_perc_pix, val_patch_shape );

//		uiService.show("validationY", validationY);

		Img< FloatType > target = makeTarget( X, targetDims );

		N2V_DataWrapper< FloatType > training_data = new N2V_DataWrapper<>( context, X, target, train_batch_size, n2v_perc_pix, val_patch_shape, N2V_DataWrapper::uniform_withCP );

		int index = 0;
		List< RandomAccessibleInterval< FloatType > > inputs = new ArrayList<>();
		List< RandomAccessibleInterval< FloatType > > targets = new ArrayList<>();
		float[] weightsdata = new float[ train_batch_size ];
		for ( int i1 = 0; i1 < weightsdata.length; i1++ ) {
			weightsdata[ i1 ] = 1;
		}
		Tensor< Float > tensorWeights = Tensors.create( weightsdata );

		//TODO GUI - indicate that training is starting
		//TODO GUI - show progress bar indicating the current epoch (i, epochs)
		//TODO GUI - show progress bar indicating the step of the current epoch (j, steps_per_epoch)
		//TODO GUI - show plot to display loss (later we might add validation, but this is not yet calculated at all)
		//TODO GUI - display time estimate until training is done - each step should take roughly the same time
		//TODO GUI - add footer with buttons to cancel command and to stop training

		System.out.println( "Start training.." );

		// Create dialog
		N2VDialog dialog = new N2VDialog(epochs, steps_per_epoch);
		List<Double> losses = null;

		for ( int i = 0; i < epochs; i++ ) {
			System.out.println( "\nEpoch " + ( i + 1 ) + "/" + epochs + "\n" );

			float loss = 0;
			losses = new ArrayList<Double>(steps_per_epoch);

			for ( int j = 0; j < steps_per_epoch; j++ ) {

				if ( index * train_batch_size + train_batch_size > n_train - 1 ) {
					index = 0;
					System.out.println( "starting with index 0 of training batches" );
				}

				Pair< RandomAccessibleInterval, RandomAccessibleInterval > item = training_data.getItem( index );
				inputs.add( item.getFirst() );
				targets.add( item.getSecond() );

				Tensor tensorX = DatasetTensorFlowConverter.datasetToTensor( item.getFirst(), mapping );
				Tensor tensorY = DatasetTensorFlowConverter.datasetToTensor( item.getSecond(), mapping );

				Session.Runner runner = sess.runner();

				runner.feed( tensorXOpName, tensorX ).feed( tensorYOpName, tensorY )
//						.feed(kerasLearningOpName, Tensors.create(true))
						.feed( sampleWeightsOpName, tensorWeights ).addTarget( trainingTargetOpName );
				runner.fetch( "loss/mul" );
				runner.fetch( "metrics/n2v_abs/Mean" );
				runner.fetch( "metrics/n2v_mse/Mean" );

				List< Tensor< ? > > fetchedTensors = runner.run();
				loss = fetchedTensors.get( 0 ).floatValue();
				losses.add( (double) loss );
				float abs = fetchedTensors.get( 1 ).floatValue();
				float mse = fetchedTensors.get( 2 ).floatValue();

				fetchedTensors.forEach(tensor -> tensor.close());
				tensorX.close();
				tensorY.close();
				
				progressPercentage( j + 1, steps_per_epoch, loss, abs, mse );

				//TODO GUI - update progress bar indicating the step of the current epoch
				index++;
			}
			training_data.on_epoch_end();
			if ( saveCheckpoints ) {
				sess.runner().feed( "save/Const", checkpointPrefix ).addTarget( "save/control_dependency" ).run();
			}
			
		}

		System.out.println( "Training done." );

		if ( inputs.size() > 0 ) uiService.show( "inputs", Views.stack( inputs ) );
		if ( targets.size() > 0 ) uiService.show( "targets", Views.stack( targets ) );
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
		System.out.print( step + "/" + stepTotal + " \t" + bareDone + bareRemain + " - loss: " + loss + " \tmse: " + mse + " \tabs: " + abs );
		System.out.print( "\n" );

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

	private List< RandomAccessibleInterval< FloatType > > normalizeAndTile( List< RandomAccessibleInterval< FloatType > > training ) {
		mean = new FloatType();
		mean.set( opService.stats().mean( Views.iterable( training.get( 0 ) ) ).getRealFloat() );
		stdDev = new FloatType();
		stdDev.set( opService.stats().stdDev( Views.iterable( training.get( 0 ) ) ).getRealFloat() );

		List< RandomAccessibleInterval< FloatType > > tiles = new ArrayList<>();
		for ( RandomAccessibleInterval< FloatType > trainingImg : training ) {
			tiles.addAll( createTiles( normalized( trainingImg, mean, stdDev ) ) );
		}
		return tiles;
	}

	private List< RandomAccessibleInterval< FloatType > > normalizeAndTile( RandomAccessibleInterval< FloatType > training ) {
		mean = new FloatType();
		mean.set( opService.stats().mean( Views.iterable( training ) ).getRealFloat() );
		stdDev = new FloatType();
		stdDev.set( opService.stats().stdDev( Views.iterable( training ) ).getRealFloat() );

		List< RandomAccessibleInterval< FloatType > > tiles = new ArrayList<>();
		tiles.addAll( createTiles( normalized( training, mean, stdDev ) ) );
		return tiles;
	}

	private void runPrediction( RandomAccessibleInterval< FloatType > inputRAI, Session sess, int[] mapping ) {
		long dimX = inputRAI.dimension( 0 );
		long dimY = inputRAI.dimension( 1 );
		long x_new = dimX - ( dimX % 8 ) + 8;
		long y_new = dimY - ( dimY % 8 ) + 8;
		RandomAccessibleInterval predictionInput = Views.interval( Views.extendZero( inputRAI ), new long[] { 0, 0 }, new long[] { x_new - 1, y_new - 1 } );
		predictionInput = Views.addDimension( predictionInput, 0, 0 );
		predictionInput = Views.addDimension( predictionInput, 0, 0 );

		predictionInput = normalized( predictionInput, mean, stdDev );

		Tensor inputTensor = DatasetTensorFlowConverter.datasetToTensor( predictionInput, mapping );

		Tensor outputTensor = sess.runner().feed( tensorXOpName, inputTensor )
//				.feed(kerasLearningOpName, Tensors.create(false))
				.fetch( predictionTargetOpName ).run().get( 0 );
		output = DatasetTensorFlowConverter.tensorToDataset( outputTensor, new FloatType(), mapping, false );
		output = Views.interval( inputRAI, new long[] { 0, 0 }, new long[] { dimX - 1, dimY - 1 } );
	}

	private List< String > loadGraph( Graph graph ) {
		System.out.println( "Import graph.." );
		byte[] graphDef = new byte[ 0 ];
		try {
			graphDef = IOUtils.toByteArray( getClass().getResourceAsStream( "/graph.pb" ) );
//			graphDef = Files.readAllBytes(graphDefFile.toPath());
		} catch ( IOException e ) {
			e.printStackTrace();
		}
		graph.importGraphDef( graphDef );
		Operation opTrain = graph.operation( trainingTargetOpName );
		if ( opTrain == null ) throw new RuntimeException( "Training op not found" );

//		if(saveCheckpoints) {
		final String checkpointDir = "n2v-checkpoint";
		checkpointExists = Files.exists( Paths.get( checkpointDir ) );
		checkpointPrefix =
				Tensors.create( Paths.get( checkpointDir, "ckpt" ).toString() );
//		}

		opNames = new ArrayList<>();
		graph.operations().forEachRemaining( op -> {
			for ( int i = 0; i < op.numOutputs(); i++ ) {
				Output< Object > opOutput = op.output( i );
				String name = opOutput.op().name();
				opNames.add( name );
				System.out.println( name );
			}
		} );
		return opNames;
	}

	private List< RandomAccessibleInterval< FloatType > > createTiles( RandomAccessibleInterval< FloatType > inputRAI ) {
		System.out.println( "Create tiles.." );
		List< RandomAccessibleInterval< FloatType > > data = new ArrayList<>();
		data.add( inputRAI );
		List< RandomAccessibleInterval< FloatType > > tiles = N2VDataGenerator.generatePatchesFromList(
				data,
				new FinalInterval( 64, 64 ) );
		long[] tiledim = new long[ tiles.get( 0 ).numDimensions() ];
		tiles.get( 0 ).dimensions( tiledim );
		System.out.println( "Generated " + tiles.size() + " tiles of shape " + Arrays.toString( tiledim ) );

		RandomAccessibleInterval< FloatType > tilesStack = Views.stack( tiles );
		uiService.show( "tiles", tilesStack );
		return tiles;
	}

	private RandomAccessibleInterval< FloatType > normalized( RandomAccessibleInterval< FloatType > input, FloatType mean, FloatType stdDev ) {
//		Img<FloatType> rai = opService.create().img(input);
//		LoopBuilder.setImages( rai, input ).forEachPixel( ( res, in ) -> {
//			res.set(in);
//			res.sub(mean);
//			res.div(stdDev);
//
//		} );
		IterableInterval< FloatType > rai = opService.math().subtract( Views.iterable( input ), mean );
		return ( RandomAccessibleInterval< FloatType > ) opService.math().divide( rai, stdDev );
	}

	public static void main( final String... args ) throws Exception {

		final ImageJ ij = new ImageJ();

		ij.launch( args );

//		ij.log().setLevel(LogLevel.TRACE);

//		File graphDefFile = new File("/home/random/Development/imagej/project/CSBDeep/N2V/test-graph.pb");

		final File trainingImgFile = new File( "/Users/turek/Desktop/train.tif" );

		if ( trainingImgFile.exists() ) {
			RandomAccessibleInterval _input = ( RandomAccessibleInterval ) ij.io().open( trainingImgFile.getAbsolutePath() );
			RandomAccessibleInterval _inputConverted = ij.op().convert().float32( Views.iterable( _input ) );
//			_inputConverted = Views.interval(_inputConverted, new FinalInterval(1024, 1024  ));

			RandomAccessibleInterval training = ij.op().copy().rai( _inputConverted );
			RandomAccessibleInterval prediction = training;

			CommandModule plugin = ij.command().run(
					N2V.class,
					false,
					"training",
					training,
					"prediction",
					prediction/*
							   * , "useDefaultGraph", true, "graphDefFile",
							   * graphDefFile
							   */ ).get();
			ij.ui().show( plugin.getOutput( "output" ) );
		} else
			System.out.println( "Cannot find training image " + trainingImgFile.getAbsolutePath() );

	}
}
