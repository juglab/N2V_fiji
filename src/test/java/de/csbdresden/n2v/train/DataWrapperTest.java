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

import net.imagej.ImageJ;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.Point;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import org.apache.commons.math3.util.Pair;
import org.junit.AfterClass;
import org.junit.Ignore;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

// TODO this is crap
public class DataWrapperTest {

	private static ImageJ ij = new ImageJ();

	@AfterClass
	public static void dispose() {
		ij.context().dispose();
	}

	@Test
	@Ignore //TODO write this test
	public void test_sample2D() {
		sample2D(new FinalInterval(20, 64, 64, 2), new FinalInterval(20, 32, 32, 2), 1);
		sample2D(new FinalInterval(10, 25, 25, 1), new FinalInterval(10, 12, 12, 1), 2);
		sample2D(new FinalInterval(10, 25, 25, 3), new FinalInterval(10, 13, 13, 3), 3);

		sample2D(new FinalInterval(20, 64, 64, 64, 2), new FinalInterval(20, 32, 32, 32, 2), 1);
		sample2D(new FinalInterval(10, 25, 25, 1), new FinalInterval(10, 12, 12, 1), 2);
		sample2D(new FinalInterval(10, 25, 25, 3), new FinalInterval(10, 13, 13, 3), 3);
	}

	private void sample2D(FinalInterval in_shape, FinalInterval out_shape, int seed) {
	}
//	    def _sample2D(in_shape, out_shape, seed):
//        X, Y, X_Batches, Y_Batches, indices = create_data(in_shape, out_shape)
//        np.random.seed(seed)
//        N2V_DataWrapper.__subpatch_sampling2D__(X, X_Batches, indices,
//                                                      range=in_shape[1:3]-out_shape[1:3], shape=out_shape[1:3])
//
//        assert ([*X_Batches.shape] == out_shape).all()
//        np.random.seed(seed)
//        range_y = in_shape[1] - out_shape[1]
//        range_x = in_shape[2] - out_shape[2]
//        for j in indices:
//            assert np.sum(X_Batches[j]) != 0
//            y_start = np.random.randint(0, range_y + 1)
//            x_start = np.random.randint(0, range_x + 1)
//            assert np.sum(X_Batches[j] - X[j, y_start:y_start+out_shape[1], x_start:x_start+out_shape[2]]) == 0
//
//        for j in range(in_shape[0]):
//            if j not in indices:

//                assert np.sum(X_Batches[j]) == 0

	@Test
	public void test_n2vWrapper_getitem() {
		_getitem2D(new FinalInterval(32, 32, 4, 2));
		_getitem2D(new FinalInterval(64, 64, 4, 2));
		_getitem2D(new FinalInterval(44, 55, 4, 2));
		//TODO make multiple channels work
//		_getitem2D(ij, new FinalInterval(45, 41, 4, 4));
	}

	private Img<DoubleType> createData(Interval interval) {
		Img<DoubleType> res = ij.op().create().img(interval);
		Random random = new Random();
		res.forEach(pixel -> pixel.set(random.nextDouble()));
		return res;
	}

	public static <T extends RealType<T> & NativeType<T>> double random_neighbor_withCP_uniform(IntervalView<T> patch, Point coord) {
		Random random = new Random();
		return random.nextDouble();
	}

	private void _getitem2D(Interval y_shape) {
		RandomAccessibleInterval<DoubleType> Y = createData(y_shape);
		int n_chan = (int) (y_shape.dimension(y_shape.numDimensions()-1)/2);
		List<RandomAccessibleInterval<DoubleType>> X;
//		if(n_chan == 1) {
//			X = Views.hyperSlice(Y, Y.numDimensions()-1, 0);
//			X = (RandomAccessibleInterval) Views.addDimension(X);
//		} else {
		X = getFirstHalfChannels(Y);
//		}
		N2VDataWrapper<DoubleType> dw = new N2VDataWrapper<>(X, 4, 0.198, new FinalInterval(32, 32), 5, DataWrapperTest::random_neighbor_withCP_uniform);
		Pair<RandomAccessibleInterval<DoubleType>, RandomAccessibleInterval<DoubleType>> res = dw.getItem(0);

		RandomAccessibleInterval<DoubleType> x_batch = res.getFirst();
		RandomAccessibleInterval<DoubleType> y_batch = res.getSecond();
		assertEquals(32, x_batch.dimension(0));
		assertEquals(32, x_batch.dimension(1));
		assertEquals(4, x_batch.dimension(2));
		assertEquals(n_chan, x_batch.dimension(3));

		assertEquals(32, y_batch.dimension(0));
		assertEquals(32, y_batch.dimension(1));
		assertEquals(4, y_batch.dimension(2));
		assertEquals(n_chan*2, y_batch.dimension(3));

		double sum_y = 0;
		RandomAccessibleInterval<DoubleType> secondHalfChannels = getSecondHalfChannels(y_batch);
		for (DoubleType pixel : Views.iterable(secondHalfChannels)) {
			sum_y += pixel.get();
		}
		System.out.println("sum y second half of channels: " + sum_y);
		// At least one pixel has to be a blind-spot per batch sample
		assertTrue(sum_y >= 4*n_chan);
		// At most four pixels can be affected per batch sample
		assertTrue(sum_y <= 4*4*n_chan);
	}

	private List<RandomAccessibleInterval<DoubleType>> getFirstHalfChannels(RandomAccessibleInterval<DoubleType> y) {
		List<RandomAccessibleInterval<DoubleType>> X = new ArrayList<>();
		RandomAccessibleInterval<DoubleType> _X;
		long[] start = new long[y.numDimensions()];
		long[] end = new long[y.numDimensions()];
		y.max(end);
		end[y.numDimensions()-1] = end[y.numDimensions()-1]/2;
		_X = Views.interval(y, new FinalInterval(start, end));
		for (int i = 0; i < y.dimension(y.numDimensions()-2); i++) {
			IntervalView<DoubleType> slice = Views.hyperSlice(_X, y.numDimensions() - 2, i);
			slice = Views.addDimension(slice, 0, 0);
			Views.permute(slice, slice.numDimensions()-1, slice.numDimensions()-2);
			X.add(slice);
		}
		return X;
	}

	private RandomAccessibleInterval<DoubleType> getSecondHalfChannels(RandomAccessibleInterval<DoubleType> y) {
		RandomAccessibleInterval<DoubleType> X;
		long[] start = new long[y.numDimensions()];
		long[] end = new long[y.numDimensions()];
		y.max(end);
		start[y.numDimensions()-1] = (end[y.numDimensions()-1]+1)/2;
		X = Views.interval(y, new FinalInterval(start, end));
		return X;
	}

}
