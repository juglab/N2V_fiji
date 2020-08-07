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
package de.csbdresden.n2v.train;

import net.imagej.ops.OpService;
import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converter;
import net.imglib2.converter.Converters;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

public class TrainUtils {

	public static <T extends RealType<T>> RandomAccessibleInterval<FloatType> normalizeConverter(RandomAccessibleInterval<T> data, FloatType mean, FloatType stdDev) {
		Converter<? super T, ? super FloatType> converter = (Converter<T, FloatType>) (input, output)
				-> output.set((input.getRealFloat() - mean.get())/stdDev.get());
		return Converters.convert(data, converter, new FloatType());
	}

	public static <T extends RealType<T>> void denormalizeInplace(RandomAccessibleInterval<T> input, T mean, T stdDev, OpService opService) {
		opService.math().multiply(input, Views.iterable( input ), stdDev );
		opService.math().add( input, Views.iterable(input), mean );
	}

	public static <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> copy(RandomAccessibleInterval<T> img) {
		Img<T> res = new ArrayImgFactory<>(img.randomAccess().get()).create(img);
		Cursor<T> inCursor = Views.iterable(img).localizingCursor();
		RandomAccess<T> outRA = res.randomAccess();
		while(inCursor.hasNext()) {
			inCursor.next();
			outRA.setPosition(inCursor);
			outRA.get().set(inCursor.get());
		}
		return res;
	}
}
