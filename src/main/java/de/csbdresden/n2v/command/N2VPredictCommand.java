package de.csbdresden.n2v.command;

import de.csbdresden.csbdeep.commands.GenericNetwork;
import de.csbdresden.n2v.N2VUtils;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.command.CommandModule;
import org.scijava.command.CommandService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.File;
import java.util.concurrent.ExecutionException;

@Plugin( type = Command.class, menuPath = "Plugins>CSBDeep>N2V>predict" )
public class N2VPredictCommand implements Command {

	@Parameter
	private RandomAccessibleInterval< FloatType > prediction;

	@Parameter( type = ItemIO.OUTPUT )
	private RandomAccessibleInterval< FloatType > output;

	@Parameter
	private File modelFile;

	@Parameter
	float mean;

	@Parameter
	float stdDev;

	@Parameter
	private CommandService commandService;

	@Parameter
	private OpService opService;

	@Override
	public void run() {

		prediction = N2VUtils.normalize( prediction, new FloatType(mean), new FloatType(stdDev), opService );

		File zip = modelFile;

		System.out.println("Loading model from " + modelFile);

		try {
			final CommandModule module = commandService.run(
					GenericNetwork.class, false,
					"input", prediction,
					"normalizeInput", false,
					"modelFile", zip.getAbsolutePath(),
					"blockMultiple", 8,
					"nTiles", 8,
					"overlap", 64,
					"showProgressDialog", true).get();
			output = (RandomAccessibleInterval<FloatType>) module.getOutput("output");
			N2VUtils.denormalizeInplace(output, new FloatType(mean), new FloatType(stdDev), opService);
		} catch (InterruptedException | ExecutionException e) {
			e.printStackTrace();
		}
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
