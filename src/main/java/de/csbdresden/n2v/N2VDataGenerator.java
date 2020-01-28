package de.csbdresden.n2v;

import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

import java.util.ArrayList;
import java.util.List;

public class N2VDataGenerator {

	public static <T extends RealType<T>> List<RandomAccessibleInterval<T>> generatePatchesFromList(List<RandomAccessibleInterval<T>> data, Interval shape) {

		List<RandomAccessibleInterval<T>> res = new ArrayList<>();

		data.forEach(img -> res.addAll(generatePatches(img, shape)));

//		Collections.shuffle(res);

		return res;

	}

	private static <T extends RealType<T>> List<RandomAccessibleInterval<T>> generatePatches(RandomAccessibleInterval<T> img, Interval shape) {

		List<RandomAccessibleInterval<T>> patches = extractPatches(img, shape);

		if(shape.dimension(0) == shape.dimension(1)) {
			//share in XY
			augmentPatches(patches);
		}

//		Collections.shuffle(patches);

		List<RandomAccessibleInterval<T>> res = new ArrayList<>();
		patches.forEach(patch -> {
			RandomAccessibleInterval<T> rai = Views.addDimension(patch, 0, 0);
			rai = Views.addDimension(rai, 0, 0);
			res.add(Views.zeroMin(rai));
		});

		return res;
	}

	private static <T extends RealType<T>> List<RandomAccessibleInterval<T>> extractPatches(RandomAccessibleInterval<T> img, Interval shape) {
		List<RandomAccessibleInterval<T>> res = new ArrayList<>();
		if(img.dimension(0) > shape.dimension(0) && img.dimension(1) > shape.dimension(1)) {
			for (int y = 0; y < img.dimension(1) - shape.dimension(1)-1; y+=shape.dimension(1)) {
				for (int x = 0; x < img.dimension(0) - shape.dimension(0)-1; x+=shape.dimension(0)) {
					long[] min = {x, y};
					long[] max = {x + shape.dimension(0)-1, y + shape.dimension(1)-1};
//					System.out.println(res.size() + ": " + Arrays.toString(min) + " -> " + Arrays.toString(max));
					res.add(Views.interval(img,
							min,
							max));
				}
			}
		} else {
			if(img.dimension(0) == shape.dimension(0) && img.dimension(1) == shape.dimension(1)) {
				res.add(img);
			} else {
				System.out.println("N2VDataGenerator::extractPatches: 'shape' is too big");
			}
		}
		return res;
	}

	private static <T extends RealType<T>> void augmentPatches(List<RandomAccessibleInterval<T>> patches) {
		List<RandomAccessibleInterval<T>> augmented = new ArrayList<>();
		patches.forEach(patch -> {
			IntervalView<T> r1 = Views.rotate(patch, 0, 1);
			IntervalView<T> r2 = Views.rotate(r1, 0, 1);
			augmented.add(r1);
			augmented.add(r2);
			augmented.add(Views.rotate(r2, 0, 1));
		});
		patches.addAll(augmented);
		augmented.clear();
//		patches.forEach(patch -> augmented.add(Views.invertAxis(patch, 0)));
//		patches.addAll(augmented);
	}

}
