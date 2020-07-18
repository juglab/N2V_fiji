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
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Pair;
import net.imglib2.view.Views;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Enumeration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

public class TrainUtils {

	public static <T extends RealType<T>> RandomAccessibleInterval<FloatType> normalizeConverter(RandomAccessibleInterval<T> data, FloatType mean, FloatType stdDev) {
		Converter<? super T, ? super FloatType> converter = (Converter<T, FloatType>) (input, output)
				-> output.set((input.getRealFloat() - mean.get())/stdDev.get());
		return Converters.convert(data, converter, new FloatType());
	}

	public static void normalizeInplace(RandomAccessibleInterval<FloatType> data, FloatType mean, FloatType stdDev) {
		LoopBuilder.setImages( data ).forEachPixel( (pixel ) -> {
			pixel.sub(mean);
			pixel.div(stdDev);
		} );
	}

	public static <T extends RealType<T>> void denormalizeInplace(RandomAccessibleInterval<T> input, T mean, T stdDev, OpService opService) {
		opService.math().multiply(input, Views.iterable( input ), stdDev );
		opService.math().add( input, Views.iterable(input), mean );
	}

	public static File saveTrainedModel(File checkpointDir) throws IOException {
		Path out = Files.createTempFile("n2v-", ".bioimage.io.zip");
		FileOutputStream fos = new FileOutputStream(out.toFile());
		ZipOutputStream zipOut = new ZipOutputStream(fos);
		zipFile(checkpointDir, null, zipOut);
		zipOut.close();
		fos.close();
		return out.toFile();
	}

	static void zipFile(File fileToZip, String fileName, ZipOutputStream zipOut) throws IOException {
		if (fileToZip.isHidden()) {
			return;
		}
		if (fileToZip.isDirectory()) {
			if(fileName == null) {
				File[] children = fileToZip.listFiles();
				for (File childFile : children) {
					zipFile(childFile, childFile.getName(), zipOut);
				}
			} else {
				if (fileName.endsWith("/")) {
					zipOut.putNextEntry(new ZipEntry(fileName));
					zipOut.closeEntry();
				} else {
					zipOut.putNextEntry(new ZipEntry(fileName + "/"));
					zipOut.closeEntry();
				}
				File[] children = fileToZip.listFiles();
				for (File childFile : children) {
					zipFile(childFile, fileName + "/" + childFile.getName(), zipOut);
				}
			}
			return;
		}
		FileInputStream fis = new FileInputStream(fileToZip);
		ZipEntry zipEntry = new ZipEntry(fileName);
		zipOut.putNextEntry(zipEntry);
		byte[] bytes = new byte[1024];
		int length;
		while ((length = fis.read(bytes)) >= 0) {
			zipOut.write(bytes, 0, length);
		}
		fis.close();
	}

	public static void unZipAll(File source, File destination) throws IOException
	{
		System.out.println("Unzipping - " + source.getName());
		int BUFFER = 2048;

		ZipFile zip = new ZipFile(source);
		try{
			destination.getParentFile().mkdirs();
			Enumeration zipFileEntries = zip.entries();

			// Process each entry
			while (zipFileEntries.hasMoreElements())
			{
				// grab a zip file entry
				ZipEntry entry = (ZipEntry) zipFileEntries.nextElement();
				String currentEntry = entry.getName();
				File destFile = new File(destination, currentEntry);
				//destFile = new File(newPath, destFile.getName());
				File destinationParent = destFile.getParentFile();

				// create the parent directory structure if needed
				destinationParent.mkdirs();

				if (!entry.isDirectory())
				{
					BufferedInputStream is = null;
					FileOutputStream fos = null;
					BufferedOutputStream dest = null;
					try{
						is = new BufferedInputStream(zip.getInputStream(entry));
						int currentByte;
						// establish buffer for writing file
						byte data[] = new byte[BUFFER];

						// write the current file to disk
						fos = new FileOutputStream(destFile);
						dest = new BufferedOutputStream(fos, BUFFER);

						// read and write until last byte is encountered
						while ((currentByte = is.read(data, 0, BUFFER)) != -1) {
							dest.write(data, 0, currentByte);
						}
					} catch (Exception e){
						System.out.println("unable to extract entry:" + entry.getName());
						throw e;
					} finally{
						if (dest != null){
							dest.close();
						}
						if (fos != null){
							fos.close();
						}
						if (is != null){
							is.close();
						}
					}
				}else{
					//Create directory
					destFile.mkdirs();
				}

				if (currentEntry.endsWith(".zip"))
				{
					// found a zip file, try to extract
					unZipAll(destFile, destinationParent);
					if(!destFile.delete()){
						System.out.println("Could not delete zip");
					}
				}
			}
		} catch(Exception e){
			e.printStackTrace();
			System.out.println("Failed to successfully unzip:" + source.getName());
		} finally {
			zip.close();
		}
		System.out.println("Done Unzipping:" + source.getName());
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
