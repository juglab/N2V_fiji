package de.csbdresden.n2v;

import net.imagej.ops.OpService;
import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Enumeration;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

public class N2VUtils {

	private static int n2v_neighborhood_radius = 5;

	public static <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> normalize(RandomAccessibleInterval<T> input, T mean, T stdDev, OpService opService) {
		Img<T> rai = opService.create().img(input, input.randomAccess().get().copy());
		LoopBuilder.setImages( rai, input ).forEachPixel( (res, in ) -> {
			res.set(in);
			res.sub(mean);
			res.div(stdDev);

		} );
		return rai;
//		IterableInterval< T > rai = opService.math().subtract( Views.iterable( input ), mean );
//		return (RandomAccessibleInterval<T>) opService.math().divide( rai, stdDev );
	}

	public static <T extends RealType<T>> void denormalizeInplace(RandomAccessibleInterval<T> input, T mean, T stdDev, OpService opService) {
		opService.math().multiply(input, Views.iterable( input ), stdDev );
		opService.math().add( input, Views.iterable(input), mean );
	}

	static File saveTrainedModel(File checkpointDir) throws IOException {
		Path out = Files.createTempFile("n2v-trained-model", ".zip");
		FileOutputStream fos = new FileOutputStream(out.toFile());
		ZipOutputStream zipOut = new ZipOutputStream(fos);
		ZipDirectory.zipFile(checkpointDir, null, zipOut);
		zipOut.close();
		fos.close();
		return out.toFile();
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

	public static void normalize(List<RandomAccessibleInterval<FloatType>> data, FloatType mean, FloatType stdDev, OpService opService) {
		for (int i = 0; i < data.size(); i++) {
			data.set(i, normalize(data.get(i), mean, stdDev, opService));
		}
	}

	public static <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> copy(IntervalView<T> img) {
		Img<T> res = new ArrayImgFactory<>(img.randomAccess().get()).create(img);
		Cursor<T> inCursor = img.localizingCursor();
		RandomAccess<T> outRA = res.randomAccess();
		while(inCursor.hasNext()) {
			inCursor.next();
			outRA.setPosition(inCursor);
			outRA.get().set(inCursor.get());
		}
		return res;
	}
}
