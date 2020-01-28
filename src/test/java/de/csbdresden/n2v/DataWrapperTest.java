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

public class DataWrapperTest {

	@Test
	public void test_n2vWrapper_getitem() {
		ImageJ ij = new ImageJ();
		_getitem2D(ij, new FinalInterval(32, 32, 4, 2));
		_getitem2D(ij, new FinalInterval(64, 64, 4, 2));
		_getitem2D(ij, new FinalInterval(44, 55, 4, 2));
		_getitem2D(ij, new FinalInterval(45, 41, 4, 4));
	}

	private RandomAccessibleInterval createData(ImageJ ij, Interval interval) {
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
		RandomAccessibleInterval Y = createData(ij, y_shape);
		int n_chan = (int) (y_shape.dimension(y_shape.numDimensions()-1)/2);
		RandomAccessibleInterval X;
//		if(n_chan == 1) {
//			X = Views.hyperSlice(Y, Y.numDimensions()-1, 0);
//			X = (RandomAccessibleInterval) Views.addDimension(X);
//		} else {
			long[] start = new long[Y.numDimensions()];
			long[] end = new long[Y.numDimensions()];
			Y.max(end);
			end[Y.numDimensions()-1] = end[Y.numDimensions()-1]/2;
			X = Views.interval(Y, new FinalInterval(start, end));
//		}
		N2V_DataWrapper dw = new N2V_DataWrapper<>(ij.context(), X, Y, 4, 0.198, new FinalInterval(32, 32), DataWrapperTest::random_neighbor_withCP_uniform);
		Pair<RandomAccessibleInterval, RandomAccessibleInterval> res = dw.getItem(0);

		RandomAccessibleInterval x_batch = res.getFirst();
		RandomAccessibleInterval y_batch = res.getSecond();
		assertEquals(32, x_batch.dimension(0));
		assertEquals(32, x_batch.dimension(1));
		assertEquals(4, x_batch.dimension(2));
		assertEquals(n_chan, x_batch.dimension(3));

		assertEquals(32, y_batch.dimension(0));
		assertEquals(32, y_batch.dimension(1));
		assertEquals(4, y_batch.dimension(2));
		assertEquals(n_chan*2, y_batch.dimension(3));
	}

//    def _getitem2D(y_shape):

//        val_manipulator = random_neighbor_withCP_uniform
//        dw = N2V_DataWrapper(X, Y, 4, perc_pix=0.198, shape=(32, 32), value_manipulation=val_manipulator)
//
//        x_batch, y_batch = dw.__getitem__(0)
//        assert x_batch.shape == (4, 32, 32, int(n_chan))
//        assert y_batch.shape == (4, 32, 32, int(2*n_chan))
//        # At least one pixel has to be a blind-spot per batch sample
//        assert np.sum(y_batch[..., n_chan:]) >= 4 * n_chan
//        # At most four pixels can be affected per batch sample
//        assert np.sum(y_batch[..., n_chan:]) <= 4*4 * n_chan
//
//
//    def _getitem3D(y_shape):
//        Y = create_data(y_shape)
//        n_chan = y_shape[-1]//2
//        X = Y[:,:,:,:,0][:,:,:,:,np.newaxis]
//        val_manipulator = random_neighbor_withCP_uniform
//        dw = N2V_DataWrapper(X, Y, 4, perc_pix=0.198, shape=(32, 32, 32), value_manipulation=val_manipulator)
//
//        x_batch, y_batch = dw.__getitem__(0)
//        assert x_batch.shape == (4, 32, 32, 32, 1)
//        assert y_batch.shape == (4, 32, 32, 32, 2)
//        # At least one pixel has to be a blind-spot per batch sample
//        assert np.sum(y_batch[..., n_chan:]) >= 1*4 * n_chan
//        # At most 8 pixels can be affected per batch sample
//        assert np.sum(y_batch[..., n_chan:]) <= 8*4 * n_chan
//
//

}
