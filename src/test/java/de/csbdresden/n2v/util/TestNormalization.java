package de.csbdresden.n2v.util;

import net.imagej.ImageJ;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.view.Views;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class TestNormalization {

	@Test
	public void testNormalization() {

		// create image with 100 pixels, values going from 0.5 to 99.5
		ImageJ ij = new ImageJ();
		Img<DoubleType> img = ij.op().create().img(new long[]{10, 10});
		RandomAccess<DoubleType> ra = img.randomAccess();
		for (int i = 0; i < img.dimension(0); i++) {
			for (int j = 0; j < img.dimension(1); j++) {
				ra.setPosition(i, 0);
				ra.setPosition(j, 1);
				ra.get().set(img.dimension(0)*i+j+0.5);
			}
		}
		DoubleType mean = ij.op().stats().mean(img);
		DoubleType stdDev = ij.op().stats().stdDev(img);

		// normalize image
		RandomAccessibleInterval<DoubleType> imgNormalized = N2VUtils.normalize(img, mean, stdDev, ij.op());
		DoubleType meanNormalized = ij.op().stats().mean(Views.iterable(imgNormalized));
		DoubleType stdDevNormalized = ij.op().stats().stdDev(Views.iterable(imgNormalized));

		//denormalize image
		N2VUtils.denormalizeInplace(imgNormalized, mean, stdDev, ij.op());
		DoubleType meanDenormalized = ij.op().stats().mean(Views.iterable(imgNormalized));
		DoubleType stdDevDenormalized = ij.op().stats().stdDev(Views.iterable(imgNormalized));

		// denormalized image and original image should have same statistics
		assertEquals(mean.get(), meanDenormalized.get(), 0.00001);
		assertEquals(stdDev.get(), stdDevDenormalized.get(), 0.00001);

		// normalized image should be centered around 0, stddev of 1
		assertEquals(0.0, meanNormalized.get(), 0.00001);
		assertEquals(1.0, stdDevNormalized.get(), 0.00001);

	}

}
