package de.csbdresden.n2v.command;

import de.csbdresden.csbdeep.converter.FloatRealConverter;
import de.csbdresden.n2v.predict.N2VPrediction;
import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.ImageJ;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.converter.RealFloatConverter;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import org.scijava.Context;
import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.command.CommandModule;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.File;

@Plugin( type = Command.class, menuPath = "Plugins>CSBDeep>N2V>predict" )
public class N2VPredictCommand <T extends RealType<T>> implements Command {

	@Parameter
	private RandomAccessibleInterval< T > input;

	@Parameter( type = ItemIO.OUTPUT )
	private Dataset output;

	@Parameter
	private File modelFile;

	@Parameter
	private Context context;

//	@Parameter
//	private DisplayService displayService;

	@Parameter
	private DatasetService datasetService;

//	@Parameter
//	private ImageDisplayService imageDisplayService;

	@Override
	public void run() {
		//TODO make transferring LUTs work..
		//TODO the following code works for IJ2, but not for LUTs set via IJ1
//		List<Display<?>> displays = displayService.getDisplays(prediction);
//		List<ColorTable> colorTables = new ArrayList<>();
//		if(displays.size() > 0) {
//			ImageDisplay display = (ImageDisplay) displays.get(0);
//			display.update();
//			DatasetView view = imageDisplayService.getActiveDatasetView(display);
//			colorTables = view.getColorTables();
//		}
		N2VPrediction prediction = new N2VPrediction(context);
		prediction.setModelFile(modelFile);
		prediction.setShowDialog(true);
		RandomAccessibleInterval<FloatType> converted = Converters.convert(input, new RealFloatConverter<>(), new FloatType());
		output = datasetService.create(prediction.predictPadded(converted));
//		output = Converters.convert(_output, new FloatRealConverter<>(), input.randomAccess().get());
//		output = datasetService.create(_output);
//		output.initializeColorTables(colorTables.size());
//		for (int i = 0; i < colorTables.size(); i++) {
//			output.setColorTable(colorTables.get(i), i);
//		}
	}

	public static void main( final String... args ) throws Exception {

		final ImageJ ij = new ImageJ();

		ij.launch( args );

//		ij.log().setLevel(LogLevel.TRACE);

		File modelFile = new File("/home/random/Development/imagej/project/CSBDeep/CSBDeep-N2V/src/main/resources/trained-model.zip");

		final File predictionInput = new File( "/home/random/Development/python/n2v/examples/2D/denoising2D_BSD68/data/BSD68_reproducibility_data/val/DCNN400_validation_gaussian25.tif" );

		if ( predictionInput.exists() ) {
			RandomAccessibleInterval _input = ( RandomAccessibleInterval ) ij.io().open( predictionInput.getAbsolutePath() );
			RandomAccessibleInterval _inputConverted = ij.op().convert().float32( Views.iterable( _input ) );
//			_inputConverted = Views.interval(_inputConverted, new FinalInterval(1024, 1024  ));

			RandomAccessibleInterval prediction = ij.op().copy().rai( _inputConverted );

			CommandModule plugin = ij.command().run( N2VPredictCommand.class, false,
					"input", prediction, "modelFile", modelFile ).get();
			ij.ui().show( plugin.getOutput( "output" ) );
		} else
			System.out.println( "Cannot find training image " + predictionInput.getAbsolutePath() );

	}
}
