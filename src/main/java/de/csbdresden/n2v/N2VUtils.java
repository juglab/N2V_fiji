package de.csbdresden.n2v;

import net.imagej.ops.OpService;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.ZipOutputStream;

public class N2VUtils {

	private static int n2v_neighborhood_radius = 5;

	public static RandomAccessibleInterval< FloatType > normalize(RandomAccessibleInterval<FloatType> input, FloatType mean, FloatType stdDev, OpService opService) {
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

	public static void denormalizeInplace(RandomAccessibleInterval<FloatType> input, FloatType mean, FloatType stdDev, OpService opService) {
		opService.math().multiply(input, Views.iterable( input ), stdDev );
		opService.math().add( input, Views.iterable(input), mean );
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
