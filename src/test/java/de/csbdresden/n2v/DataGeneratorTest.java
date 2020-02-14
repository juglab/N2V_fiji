package de.csbdresden.n2v;

import net.imglib2.FinalInterval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.integer.IntType;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class DataGeneratorTest {

	@Test
	public void testExtractPatches() {
		Img<IntType> img = new ArrayImgFactory<>(new IntType()).create(10, 10, 20);
		List<RandomAccessibleInterval<IntType>> patches = N2VDataGenerator.generateBatches(img, new FinalInterval(5, 5));
		assertEquals(4*20, patches.size());
		patches = N2VDataGenerator.generateBatches(img, new FinalInterval(10, 10));
		assertEquals(20, patches.size());
		patches = N2VDataGenerator.generateBatches(img, new FinalInterval(8, 8));
		assertEquals(20, patches.size());
	}

	@Test
	public void testAugmenter() {
		Img<IntType> img = new ArrayImgFactory<>(new IntType()).create(2, 2);
		RandomAccess<IntType> ra = img.randomAccess();
		ra.setPosition(new long[]{0, 0});
		ra.get().setOne();
		List<RandomAccessibleInterval<IntType>> patches = new ArrayList<>();
		patches.add(img);

		N2VDataGenerator.augmentBatches(patches);

		assertEquals(patches.size(), 4);

		testData(patches.get(0), new int[]{1, 0, 0, 0});
		testData(patches.get(1), new int[]{0, 1, 0, 0});
		testData(patches.get(2), new int[]{0, 0, 1, 0});
		testData(patches.get(3), new int[]{0, 0, 0, 1});
	}

	private void testData(RandomAccessibleInterval<IntType> img, int[] data) {
		RandomAccess<IntType> ra = img.randomAccess();
		ra.setPosition(new long[]{0, 0});
		assertEquals(data[0], ra.get().get());
		ra.setPosition(new long[]{0, 0});
		assertEquals(data[0], ra.get().get());
		ra.setPosition(new long[]{0, 0});
		assertEquals(data[0], ra.get().get());
		ra.setPosition(new long[]{0, 0});
		assertEquals(data[0], ra.get().get());
	}

}
