package de.csbdresden.n2v;

import net.imglib2.Cursor;
import net.imglib2.Dimensions;
import net.imglib2.FinalDimensions;
import net.imglib2.FinalInterval;
import net.imglib2.Point;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import org.apache.commons.math3.util.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class N2VDataWrapper<T extends RealType<T> & NativeType<T>> {

	private final List<RandomAccessibleInterval<T>> X;
	private final int batchSize;
	private final int batchDim;
	private final Dimensions shape;
	private final Dimensions range;
	private final long numChannels;
	private final long box_size;
	private static int local_sub_patch_radius = 5;
	private final ValueManipulatorConsumer<T> manipulator;

	public long size() {
		return X.size();
	}

	public int numBatches() {
		if(size() % batchSize > 0) return (int)(size() / batchSize) + 1;
		else return (int)(size() / batchSize);
	}

	interface ValueManipulatorConsumer<U> {
		double accept(IntervalView<U> patch, Point coord);
	}

	private static <T> double value_manipulate(
			ValueManipulatorConsumer<T> c, IntervalView<T> patch, Point coord) {
		return c.accept(patch, coord);
	}

	public N2VDataWrapper(List<RandomAccessibleInterval<T>> X, int batchSize, double perc_pix, Dimensions shape, int neighborhoodRadius, ValueManipulatorConsumer<T> manipulator) {

		this.X = X;
		this.local_sub_patch_radius = neighborhoodRadius;
		this.batchSize = batchSize;
		this.batchDim = shape.numDimensions();
		this.shape = shape;
		this.range = computeRange(X, shape);
		this.numChannels = 1;

		long multiplyShape = getMultiplyShape(shape);
		int num_pix = (int) ((float)multiplyShape / 100. * perc_pix);

//		num_pix = int(np.product(shape)/100.0 * perc_pix)
		System.out.println(num_pix + " blind-spots will be generated per training patch of size " + Arrays.toString(Intervals.dimensionsAsIntArray(shape)) + ".");

//            self.patch_sampler = self.__subpatch_sampling2D__
		this.box_size = Math.round(Math.sqrt(multiplyShape / (float)num_pix));
//            self.get_stratified_coords = self.__get_stratified_coords2D__

		this.manipulator = manipulator;
	}

	private static <T extends RealType<T> & NativeType<T>> FinalDimensions computeRange(List<RandomAccessibleInterval<T>> X, Dimensions shape) {
		long[] rangeDims = new long[shape.numDimensions()];
		for (int i = 0; i < rangeDims.length; i++) {
			rangeDims[i] = X.get(0).dimension(i) - shape.dimension(i);
		}
		return new FinalDimensions(rangeDims);
	}

	private long getMultiplyShape(Dimensions shape) {
		long res = 1;
		for (int i = 0; i < shape.numDimensions(); i++) {
			res *= shape.dimension(i);
		}
		return res;
	}

	public void on_epoch_end() {
		Collections.shuffle(X);
	}

	public Pair<RandomAccessibleInterval<T>, RandomAccessibleInterval<T>> getItem(int i) {
		int[] idx = new int[(int) Math.min(batchSize, size() - i*batchSize)];
		for (int j = 0; j < idx.length; j++) {
			idx[j] = i * batchSize + j;
		}

		Pair<RandomAccessibleInterval<T>, RandomAccessibleInterval<T>> patches = subpatch_sampling(idx);

		RandomAccessibleInterval<T> patchX = patches.getFirst();
		RandomAccessibleInterval<T> patchY = patches.getSecond();
		for (int j = 0; j < patchX.dimension(batchDim); j++) {
//            for c in range(self.n_chan):
			IntervalView<T> patchXSlice = Views.hyperSlice(patchX, batchDim, j);
			IntervalView<T> patchYSlice = Views.hyperSlice(patchY, batchDim, j);
			manipulateX(box_size, shape, patchXSlice, patchYSlice, numChannels, manipulator);
		}
		return patches;
	}

	static <T extends RealType<T> & NativeType<T>> void manipulateX(long boxSize, Dimensions shape, RandomAccessibleInterval<T> patchX, RandomAccessibleInterval<T> patchY, long n_chan, ValueManipulatorConsumer<T> manipulator) {
		int c = 0;
		List<Point> coords = null;
		if(shape.numDimensions() == 2) coords = get_stratified_coords2D(boxSize, shape);
		if(shape.numDimensions() == 3) coords = get_stratified_coords3D(boxSize, shape);

		double[] x_val = new double[coords.size()];
		double[] originalValue = new double[coords.size()];
		RandomAccess<T> batchXRA = patchX.randomAccess();
		RandomAccess<T> batchYRA = patchY.randomAccess();
		for (int k = 0; k < coords.size(); k++) {

			Point point = coords.get(k);
			long[] oldPos = new long[shape.numDimensions()+1];
			for (int i = 0; i < shape.numDimensions(); i++) {
				oldPos[i] = point.getLongPosition(i);
			}
			oldPos[shape.numDimensions()] = c;

			batchXRA.setPosition(oldPos);
			originalValue[k] = batchXRA.get().getRealDouble();

			IntervalView<T> XInterval = Views.hyperSlice(patchX, shape.numDimensions(), c);
			XInterval = Views.addDimension(XInterval, 0, 0);
			x_val[k] = value_manipulate(manipulator, XInterval, point);
		}

		for (int k = 0; k < originalValue.length; k++) {

			Point point = coords.get(k);
			long[] channel1Pos = new long[patchX.numDimensions()];
			long[] channel2Pos = new long[patchX.numDimensions()];
			for (int i = 0; i < point.numDimensions(); i++) {
				channel1Pos[i] = point.getLongPosition(i);
				channel2Pos[i] = point.getLongPosition(i);
			}
			channel1Pos[channel1Pos.length-1] = c;
			channel2Pos[channel1Pos.length-1] = c+n_chan;

			batchYRA.setPosition(channel1Pos);
			batchYRA.get().setReal(originalValue[k]);

			batchYRA.setPosition(channel2Pos);
			batchYRA.get().setOne();

			batchXRA.setPosition(channel1Pos);
			batchXRA.get().setReal(x_val[k]);
		}
	}

	private static List<Point> get_stratified_coords3D(long box_size, Dimensions shape) {
		List<Point> coords = new ArrayList<>();
		int box_count_x = (int) Math.ceil(shape.dimension(0) / box_size);
		int box_count_y = (int) Math.ceil(shape.dimension(1) / box_size);
		int box_count_z = (int) Math.ceil(shape.dimension(2) / box_size);
		for (int i = 0; i < box_count_x; i++) {
			for (int j = 0; j < box_count_y; j++) {
				for (int k = 0; k < box_count_z; k++) {
					Point p = new Point((long)(Math.random() * box_size), (long)(Math.random() * box_size), (long)(Math.random() * box_size));
	//                y, x = next(coord_gen)
					p.setPosition(i * box_size + p.getIntPosition(0), 0);
					p.setPosition(j * box_size + p.getIntPosition(1), 1);
					p.setPosition(k * box_size + p.getIntPosition(2), 2);
					if (p.getIntPosition(0) < shape.dimension(0)
							&& p.getIntPosition(1) < shape.dimension(1)
							&& p.getIntPosition(2) < shape.dimension(2)) {
						coords.add(p);
					}
				}
			}
		}
		return coords;
	}

	private static List<Point> get_stratified_coords2D(long box_size, Dimensions shape) {
		List<Point> coords = new ArrayList<>();
		int box_count_x = (int) Math.ceil(shape.dimension(0) / box_size);
		int box_count_y = (int) Math.ceil(shape.dimension(1) / box_size);
		for (int i = 0; i < box_count_x; i++) {
			for (int j = 0; j < box_count_y; j++) {
				Point p = new Point((long)(Math.random() * box_size), (long)(Math.random() * box_size), 0);
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

	public static <T extends RealType<T> & NativeType<T>> double uniform_withCP(IntervalView<T> patch, Point coord) {
//		System.out.println("original coord: " + coord);
		IntervalView<T> sub_patch = Views.zeroMin(get_subpatch(patch, coord, local_sub_patch_radius));
		Point rand_coord = new Point(coord.numDimensions()+1);
		Random random = new Random();
		for (int i = 0; i < patch.numDimensions()-1; i++) {
			rand_coord.setPosition((int)Math.floor(random.nextInt((int) (sub_patch.dimension(i)-1))), i);
		}
		RandomAccess<T> ra = sub_patch.randomAccess();
		ra.setPosition(rand_coord);
//		System.out.println("new coord: " + rand_coord);
		return ra.get().getRealDouble();
	}

	private static <T extends RealType<T> & NativeType<T>> IntervalView<T> get_subpatch(IntervalView<T> patch, Point coord, int local_sub_patch_radius) {

		Point start = new Point(patch.numDimensions());
		Point end = new Point(patch.numDimensions());
		Point shift = new Point(patch.numDimensions());

		for (int i = 0; i < patch.numDimensions()-1; i++) {
			start.setPosition(Math.max(0, coord.getIntPosition(i) - local_sub_patch_radius), i);
			end.setPosition(start.getIntPosition(i) + local_sub_patch_radius*2+1, i);
			shift.setPosition(Math.min(0, patch.dimension(i)-end.getIntPosition(i)), i);
			start.setPosition(start.getIntPosition(i) + shift.getIntPosition(i), i);
			end.setPosition(end.getIntPosition(i) + shift.getIntPosition(i), i);
		}

		long[] startPos = new long[start.numDimensions()];
		long[] endPos = new long[start.numDimensions()];
		start.localize(startPos);
		end.localize(endPos);
		return Views.interval(patch, startPos, endPos);
//		slices = [ slice(s, e) for s, e in zip(start, end)]
//		return patch[tuple(slices)]

	}

	private Pair<RandomAccessibleInterval<T>, RandomAccessibleInterval<T>> subpatch_sampling(int[] idx) {

		List<RandomAccessibleInterval<T>> X_Patches = new ArrayList<>();
		List<RandomAccessibleInterval<T>> Y_Patches = new ArrayList<>();

		Random r = new Random();
		for (int i = 0; i < idx.length; i++) {
			int batchIndex = idx[i];

			long[] startX = new long[shape.numDimensions()+2];
			long[] endX = new long[startX.length];
			long[] endY = new long[startX.length];

			for (int dimIndex = 0; dimIndex < shape.numDimensions(); dimIndex++) {
				startX[dimIndex] = r.nextInt((int) (range.dimension(dimIndex) + 1));
				endX[dimIndex] = startX[dimIndex] + shape.dimension(dimIndex) -1;
				endY[dimIndex] = shape.dimension(dimIndex);
			}

			startX[shape.numDimensions()] = 0;
			endX[shape.numDimensions()] = 0;

			startX[shape.numDimensions()+1] = 0;
			endX[shape.numDimensions()+1] = 0; //TODO make multichannel work

			endY[shape.numDimensions()] = 1;

			endY[shape.numDimensions()+1] = 2; //TODO make multichannel work

			RandomAccessibleInterval<T> patch = getPatch(X.get(batchIndex), new FinalInterval(startX, endX));
			X_Patches.add(patch);
			FinalDimensions dimY = new FinalDimensions(endY);
//			Y_Patches.add(opService.create().img(dimY, patch.randomAccess().get()));
			Y_Patches.add(new ArrayImgFactory<>(patch.randomAccess().get()).create(dimY));
//	    Y_Batches[batchIndex] = Y[batchIndex, y_start:y_start + shape[0], x_start:x_start + shape[1]]
		}
		return new Pair<>(Views.concatenate(shape.numDimensions(), X_Patches), Views.concatenate(shape.numDimensions(), Y_Patches));
	}

	private RandomAccessibleInterval<T> getPatch(RandomAccessibleInterval<T> source, FinalInterval interval) {
		Img<T> res = new ArrayImgFactory<>(source.randomAccess().get()).create(Views.zeroMin(Views.interval(source, interval)));
		Cursor<T> inCursor = Views.zeroMin(Views.interval(source, interval)).localizingCursor();
		RandomAccess<T> outRA = res.randomAccess();
		while(inCursor.hasNext()) {
			inCursor.next();
			outRA.setPosition(inCursor);
			outRA.get().set(inCursor.get());
		}
		return res;
//		return opService.copy().rai(Views.zeroMin(Views.interval(source, interval)));
	}

}
