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
	public void testExtractPatches2D() {
		Img<IntType> img = new ArrayImgFactory<>(new IntType()).create(10, 10, 20);
		List<RandomAccessibleInterval<IntType>> patches = N2VDataGenerator.generateBatches(img, new FinalInterval(5, 5));
		assertEquals(4*20, patches.size());
		patches = N2VDataGenerator.generateBatches(img, new FinalInterval(10, 10));
		assertEquals(20, patches.size());
		patches = N2VDataGenerator.generateBatches(img, new FinalInterval(8, 8));
		assertEquals(20, patches.size());
	}

	@Test
	public void testExtractPatches3D() {
		Img<IntType> img = new ArrayImgFactory<>(new IntType()).create(10, 10, 10, 20);
		List<RandomAccessibleInterval<IntType>> patches = N2VDataGenerator.generateBatches(img, new FinalInterval(5, 5, 5));
		assertEquals(8*20, patches.size());
		patches = N2VDataGenerator.generateBatches(img, new FinalInterval(10, 10, 10));
		assertEquals(20, patches.size());
		patches = N2VDataGenerator.generateBatches(img, new FinalInterval(8, 8, 8));
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

		assertEquals(patches.size(), 8);

		testData(patches.get(0), new int[]{1, 0, 0, 0});
		testData(patches.get(1), new int[]{0, 1, 0, 0});
		testData(patches.get(2), new int[]{0, 0, 1, 0});
		testData(patches.get(3), new int[]{0, 0, 0, 1});
		testData(patches.get(4), new int[]{0, 1, 0, 0});
		testData(patches.get(5), new int[]{1, 0, 0, 0});
		testData(patches.get(6), new int[]{0, 0, 0, 1});
		testData(patches.get(7), new int[]{0, 0, 1, 0});
	}

	private void testData(RandomAccessibleInterval<IntType> img, int[] data) {
		RandomAccess<IntType> ra = img.randomAccess();
		ra.setPosition(new long[]{0, 0});
		assertEquals(data[0], ra.get().get());
		ra.setPosition(new long[]{1, 0});
		assertEquals(data[1], ra.get().get());
		ra.setPosition(new long[]{1, 1});
		assertEquals(data[2], ra.get().get());
		ra.setPosition(new long[]{0, 1});
		assertEquals(data[3], ra.get().get());
	}

}
