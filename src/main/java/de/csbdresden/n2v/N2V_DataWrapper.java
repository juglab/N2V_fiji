package de.csbdresden.n2v;

import net.imagej.ops.OpService;
import net.imglib2.Dimensions;
import net.imglib2.FinalDimensions;
import net.imglib2.FinalInterval;
import net.imglib2.Point;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import org.apache.commons.math3.util.Pair;
import org.scijava.Context;
import org.scijava.plugin.Parameter;
import org.scijava.ui.UIService;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class N2V_DataWrapper <T extends RealType<T> & NativeType<T>> {

	private final Img<T> X_Batches;
	private final Img<T> Y_Batches;
	@Parameter
	OpService opService;

	private final RandomAccessibleInterval<T> X;
	private final RandomAccessibleInterval<T> Y;
	private final int batch_size;
	private ArrayList<Integer> perm;
	private final Dimensions shape;
	private final Dimensions range;
	private final int dims;
	private final long n_chan;
	private final long box_size;
	private static int local_sub_patch_radius = 5;
	private final ValueManipulatorConsumer<T> manipulator;

	interface ValueManipulatorConsumer<U> {
		double accept(IntervalView<U> u, Point v, int w);
	}

	private static <T> double value_manipulate(
			ValueManipulatorConsumer<T> c, IntervalView<T> arg2, Point arg3, int arg4) {
		return c.accept(arg2, arg3, arg4);
	}

	public N2V_DataWrapper(Context context, RandomAccessibleInterval<T> X, RandomAccessibleInterval<T> Y, int batch_size, double perc_pix, Dimensions shape, ValueManipulatorConsumer<T> manipulator) {

		context.inject(this);

		this.X = X;
		this.Y = Y;
		this.batch_size = batch_size;
		this.perm = generateRandom((int) X.dimension(2));
		this.shape = shape;
		this.range = new FinalDimensions(X.dimension(0) - shape.dimension(0), X.dimension(1) - shape.dimension(1));
		this.dims = shape.numDimensions();
		this.n_chan = X.dimension(3);

		long multiplyShape = 1;
		for (int i = 0; i < shape.numDimensions(); i++) {
			multiplyShape *= shape.dimension(i);
		}
		int num_pix = (int) (multiplyShape / 100 * perc_pix);

//            self.patch_sampler = self.__subpatch_sampling2D__
		this.box_size = Math.round(Math.sqrt(shape.dimension(0) * shape.dimension(1) / (float)num_pix));
//            self.get_stratified_coords = self.__get_stratified_coords2D__
		this.X_Batches = opService.create().img(new FinalDimensions(shape.dimension(0), shape.dimension(1), batch_size, X.dimension(3)), X.randomAccess().get().copy());
		this.Y_Batches = opService.create().img(new FinalDimensions(shape.dimension(0), shape.dimension(1), batch_size, Y.dimension(3)), X.randomAccess().get().copy());

		this.manipulator = manipulator;
	}

	public int len() {
		return (int) Math.ceil(X.dimension(2) / (float) batch_size);
	}

	public void on_epoch_end() {
		perm = generateRandom((int) X.dimension(2));
		T zero = X_Batches.randomAccess().get().copy();
		zero.setZero();
		opService.math().multiply((RandomAccessibleInterval<T>) X_Batches, X_Batches, zero);
		opService.math().multiply((RandomAccessibleInterval<T>) Y_Batches, Y_Batches, zero);
	}

	public Pair<RandomAccessibleInterval, RandomAccessibleInterval> getItem(int i) {
		System.out.println("------------- item " + i);
        int[] idx = new int[batch_size];
		for (int j = 0; j < idx.length; j++) {
			idx[j] = perm.get(i * batch_size + j);
		}
        subpatch_sampling2D(idx);

//		opService.context().getService(UIService.class).show("x batches", X_Batches);

		for (int j = 0; j < idx.length; j++) {
//            for c in range(self.n_chan):

			manipulateY(j, box_size, shape, X_Batches, Y_Batches, dims, n_chan, manipulator);
		}
		//TODO the current return value seems wrong
		//self.X_Batches[idx], self.Y_Batches[idx]
//		return new Pair<>(X_Batches, Y_Batches);
		//TODO this should probably not be copied
		return new Pair<>(opService.copy().rai(X_Batches), opService.copy().rai(Y_Batches));
	}

	static <T extends RealType<T> & NativeType<T>> void manipulateY(int j, long box_size, Dimensions shape, RandomAccessibleInterval<T> X_Batches, RandomAccessibleInterval<T> Y_Batches, int dims, long n_chan, ValueManipulatorConsumer<T> manipulator) {
		int c = 0;
		List<Point> coords = get_stratified_coords(box_size, shape);
//                                                    shape=np.array(self.X_Batches.shape)[1:-1])

		double[] x_val = new double[coords.size()];
		double[] y_val = new double[coords.size()];
		RandomAccess<T> batchXRA = X_Batches.randomAccess();
		RandomAccess<T> batchYRA = Y_Batches.randomAccess();
		for (int k = 0; k < coords.size(); k++) {
			batchXRA.setPosition(new long[]{coords.get(k).getLongPosition(0), coords.get(k).getLongPosition(1), j, c});
			y_val[k] = batchXRA.get().getRealDouble();
			IntervalView<T> XInterval = Views.hyperSlice(X_Batches, 2, j);
			XInterval = Views.hyperSlice(XInterval, 2, c);
			XInterval = Views.addDimension(XInterval, 0, 0);
			x_val[k] = value_manipulate(manipulator, XInterval, coords.get(k), dims);
			batchYRA.setPosition(new long[]{coords.get(k).getLongPosition(0), coords.get(k).getLongPosition(1), j, c});
//                    y_val.append(np.copy(self.Y_Batches[(j, *coords[k], ..., c)]))
//                    x_val.append(self.value_manipulation(self.X_Batches[j, ..., c][...,np.newaxis], coords[k], self.dims))
		}

		//self.Y_Batches[indexing] = y_val
		for (int k = 0; k < y_val.length; k++) {
			batchYRA.setPosition(new long[]{coords.get(k).getLongPosition(0), coords.get(k).getLongPosition(1), j, c});
			batchYRA.get().setReal(y_val[k]);
		}

//			self.Y_Batches[indexing_mask] = 1
		for (int k = 0; k < y_val.length; k++) {
			batchYRA.setPosition(new long[]{coords.get(k).getLongPosition(0), coords.get(k).getLongPosition(1), j, c+n_chan});
			batchYRA.get().setOne();
		}

//			self.X_Batches[indexing] = x_val
		for (int k = 0; k < x_val.length; k++) {
			batchXRA.setPosition(new long[]{coords.get(k).getLongPosition(0), coords.get(k).getLongPosition(1), j, c});
			batchXRA.get().setReal(x_val[k]);
		}
	}


	private static List<Point> get_stratified_coords(long box_size, Dimensions shape) {
		List<Point> coords = new ArrayList<>();
		int box_count_x = (int) Math.ceil(shape.dimension(0) / box_size);
		int box_count_y = (int) Math.ceil(shape.dimension(1) / box_size);
		for (int i = 0; i < box_count_x; i++) {
			for (int j = 0; j < box_count_y; j++) {
				Point p = new Point((long)Math.random() * box_size, (long)Math.random() * box_size, 0);
//                y, x = next(coord_gen)
				p.setPosition(i * box_size + p.getIntPosition(0), 0);
				p.setPosition(j * box_size + p.getIntPosition(1), 1);
				if (p.getIntPosition(0) < shape.dimension(0) && p.getIntPosition(1) < shape.dimension(1)) {
					coords.add(p);
				}
			}
		}
		return coords;
	}

	public static <T extends RealType<T> & NativeType<T>> double uniform_withCP(IntervalView<T> patch, Point coord, int dims) {
		IntervalView<T> sub_patch = get_subpatch(patch, coord, local_sub_patch_radius);
		Point rand_coord = new Point(coord.numDimensions());
		for (int i = 0; i < dims; i++) {
			rand_coord.setPosition((int)Math.floor(Math.random() * sub_patch.dimension(i)), i);
		}
		RandomAccess<T> ra = sub_patch.randomAccess();
		ra.setPosition(rand_coord);
		return ra.get().getRealDouble();
	}

	private static <T extends RealType<T> & NativeType<T>> IntervalView<T> get_subpatch(IntervalView<T> patch, Point coord, int local_sub_patch_radius) {

		Point start = new Point(coord.numDimensions());
		Point end = new Point(coord.numDimensions());
		Point shift = new Point(coord.numDimensions());

		for (int i = 0; i < coord.numDimensions(); i++) {
			start.setPosition(Math.max(0, coord.getIntPosition(i) - local_sub_patch_radius), i);
			end.setPosition(start.getIntPosition(i) + local_sub_patch_radius*2+1, i);
			shift.setPosition(Math.min(0, patch.dimension(i)-end.getIntPosition(i)), i);
			start.setPosition(start.getIntPosition(i) + shift.getIntPosition(i), i);
			end.setPosition(end.getIntPosition(i) + shift.getIntPosition(i), i);
		}

		long[] startPos = new long[coord.numDimensions()];
		long[] endPos = new long[coord.numDimensions()];
		start.localize(startPos);
		end.localize(endPos);
		return Views.interval(patch, startPos, endPos);
//		slices = [ slice(s, e) for s, e in zip(start, end)]
//		return patch[tuple(slices)]

	}

	private void subpatch_sampling2D(int[] idx) {

		Random r = new Random();
	    for (int i = 0; i < idx.length; i++) {
		    int j = idx[i];
		    int x_start = r.nextInt((int) (range.dimension(0) + 1));
		    int y_start = r.nextInt((int) (range.dimension(1) + 1));
		    setBatch(X_Batches, i, X,
				    new FinalInterval(
						    new long[]{x_start, y_start, j, 0},
						    new long[]{x_start + shape.dimension(0)-1, y_start + shape.dimension(1)-1, j, 0}));
		    setBatch(Y_Batches, i, Y,
				    new FinalInterval(
						    new long[]{x_start, y_start, j, 0},
						    new long[]{x_start + shape.dimension(0)-1, y_start + shape.dimension(1)-1, j, 0}));
//	    Y_Batches[j] = Y[j, y_start:y_start + shape[0], x_start:x_start + shape[1]]
	    }
    }

	private void setBatch(Img<T> batches, int numBatch, RandomAccessibleInterval<T> source, FinalInterval interval) {
		RandomAccess<T> ra = batches.randomAccess();
		RandomAccess<T> raSource = source.randomAccess();
//		System.out.println("batches: " + Arrays.toString(Intervals.dimensionsAsIntArray(batches)));
//		System.out.println("source: " + Arrays.toString(Intervals.dimensionsAsIntArray(source)));
//		System.out.println("interval: " + Arrays.toString(Intervals.dimensionsAsIntArray(interval)));
		for (long i = interval.min(0); i < interval.max(0); i++) {
			for (long j = interval.min(0); j < interval.max(1); j++) {
				long[] positionBatch = {i-interval.min(0), j-interval.min(1), numBatch, 0};
				long[] positionSource = {i, j, numBatch, 0};
				ra.setPosition(positionBatch);
				raSource.setPosition(positionSource);
				ra.get().set(raSource.get());
			}
		}
	}

	// Function to return the next random number
	static int getNum(ArrayList<Integer> v)
	{
		// Size of the vector
		int n = v.size();

		// Make sure the number is within
		// the index range
		int index = (int)(Math.random() * n);

		// Get random number from the vector
		int num = v.get(index);

		// Remove the number from the vector
		v.set(index, v.get(n - 1));
		v.remove(n - 1);

		// Return the removed number
		return num;
	}

	// Function to generate n
	// non-repeating random numbers
	static ArrayList<Integer> generateRandom(int n)
	{
		ArrayList<Integer> v = new ArrayList<>(n);

		for (int i = 0; i < n; i++)
			v.add(i);

		Collections.shuffle(v);

		return v;
	}
}
