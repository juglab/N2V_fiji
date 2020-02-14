package de.csbdresden.n2v;

import net.imagej.ops.OpService;
import net.imglib2.Dimensions;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import org.apache.commons.lang3.NotImplementedException;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.ZipOutputStream;

public class N2VUtils {

	private static int n2v_neighborhood_radius = 5;

	static RandomAccessibleInterval< FloatType > normalize(RandomAccessibleInterval< FloatType > input, FloatType mean, FloatType stdDev, OpService opService ) {
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

	static void manipulate_val_data(RandomAccessibleInterval<FloatType> X_val, RandomAccessibleInterval<FloatType> Y_val, double perc_pix, Dimensions shape) {
		int dims = shape.numDimensions();
		long box_size;
		if(dims == 2) {
			box_size = Math.round(Math.sqrt(100./perc_pix));
		} else {
			throw new NotImplementedException("manipulate_val_data not implemented for dim>2");
		}
		long n_chan = X_val.dimension(X_val.numDimensions()-1);
		Views.iterable(Y_val).forEach(val -> val.set(0));
		for (int j = 0; j < X_val.dimension(dims); j++) {
			N2V_DataWrapper.manipulateY(j, box_size, shape, X_val, Y_val, dims, n_chan, N2V_DataWrapper::uniform_withCP);
		}
	}

	static float[] rand_float_coords2D(long boxsize) {
		return new float[]{(float) (Math.random() * boxsize), (float) (Math.random() * boxsize)};
	}

	static File saveTrainedModel(File checkpointDir) throws IOException {
		Path out = Files.createTempFile("n2v-trained-model", ".zip");
		FileOutputStream fos = new FileOutputStream(out.toFile());
		ZipOutputStream zipOut = new ZipOutputStream(fos);
		ZipDirectory.zipFile(checkpointDir, null, zipOut);
		zipOut.close();
		fos.close();
		return out.toFile();
	}


}
