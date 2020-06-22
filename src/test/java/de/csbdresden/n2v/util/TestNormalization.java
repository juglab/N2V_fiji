/*-
 * #%L
 * N2V plugin
 * %%
 * Copyright (C) 2019 - 2020 Center for Systems Biology Dresden
 * %%
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */
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

		ij.context().dispose();

	}

}
