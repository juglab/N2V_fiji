package de.csbdresden.n2v;

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
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class DataWrapperTest {

	@Test
	public void test_n2vWrapper_getitem() {
		ImageJ ij = new ImageJ();
		_getitem2D(ij, new FinalInterval(32, 32, 4, 2));
		_getitem2D(ij, new FinalInterval(64, 64, 4, 2));
		//TODO make work
//		_getitem2D(ij, new FinalInterval(44, 55, 4, 2));
		//TODO make multiple channels work
//		_getitem2D(ij, new FinalInterval(45, 41, 4, 4));
	}

	private Img<DoubleType> createData(ImageJ ij, Interval interval) {
		Img<DoubleType> res = ij.op().create().img(interval);
		Random random = new Random();
		res.forEach(pixel -> pixel.set(random.nextDouble()));
		return res;
	}

	public static <T extends RealType<T> & NativeType<T>> double random_neighbor_withCP_uniform(IntervalView<T> patch, Point coord, int dims) {
		Random random = new Random();
		return random.nextDouble();
	}

	private void _getitem2D(ImageJ ij, Interval y_shape) {
		RandomAccessibleInterval<DoubleType> Y = createData(ij, y_shape);
		int n_chan = (int) (y_shape.dimension(y_shape.numDimensions()-1)/2);
		RandomAccessibleInterval<DoubleType> X;
//		if(n_chan == 1) {
//			X = Views.hyperSlice(Y, Y.numDimensions()-1, 0);
//			X = (RandomAccessibleInterval) Views.addDimension(X);
//		} else {
		X = getFirstHalfChannels(Y);
//		}
		N2V_DataWrapper dw = new N2V_DataWrapper<>(ij.context(), X, Y, 4, 0.198, new FinalInterval(32, 32), DataWrapperTest::random_neighbor_withCP_uniform);
		Pair<RandomAccessibleInterval, RandomAccessibleInterval> res = dw.getItem(0);

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

	private RandomAccessibleInterval<DoubleType> getFirstHalfChannels(RandomAccessibleInterval<DoubleType> y) {
		RandomAccessibleInterval<DoubleType> X;
		long[] start = new long[y.numDimensions()];
		long[] end = new long[y.numDimensions()];
		y.max(end);
		end[y.numDimensions()-1] = end[y.numDimensions()-1]/2;
		X = Views.interval(y, new FinalInterval(start, end));
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
