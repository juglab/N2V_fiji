package de.csbdresden.n2v.command;

import de.csbdresden.n2v.predict.N2VPrediction;
import net.imagej.ImageJ;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import org.scijava.Context;
import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.command.CommandModule;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.File;

@Plugin( type = Command.class, menuPath = "Plugins>CSBDeep>N2V>predict" )
public class N2VPredictCommand implements Command {

	@Parameter
	private RandomAccessibleInterval< FloatType > prediction;

	@Parameter( type = ItemIO.OUTPUT )
	private RandomAccessibleInterval< FloatType > output;

	@Parameter
	private File modelFile;

	@Parameter
	private Context context;

	@Override
	public void run() {
		N2VPrediction prediction = new N2VPrediction(context);
		prediction.setModelFile(modelFile);
		int padding = 32;
		FinalInterval expand = Intervals.expand(this.prediction, padding);
		RandomAccessibleInterval output = prediction.predict(Views.zeroMin(Views.interval(Views.extendZero(this.prediction), expand)));
//			System.out.println("mean gt   : " + ij.op().stats().mean(pair.getRight()).getRealDouble());
//			System.out.println("stdDev gt : " + ij.op().stats().stdDev(pair.getRight()));
//			System.out.println("mean out  : " + ij.op().stats().mean(Views.iterable(output)));
//			System.out.println("stdDev out: " + ij.op().stats().stdDev(Views.iterable(output)));
		this.output = Views.zeroMin(Views.interval(output, Intervals.expand(output, -padding)));
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
					"prediction", prediction, "modelFile", modelFile ).get();
			ij.ui().show( plugin.getOutput( "output" ) );
		} else
			System.out.println( "Cannot find training image " + predictionInput.getAbsolutePath() );

	}
}
