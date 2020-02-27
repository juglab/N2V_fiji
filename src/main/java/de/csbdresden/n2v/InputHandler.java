package de.csbdresden.n2v;

import de.csbdresden.n2v.ui.N2VProgress;
import io.scif.services.DatasetIOService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.converter.RealFloatConverter;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import org.scijava.Context;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class InputHandler {

	@Parameter
	LogService logService;

	@Parameter
	DatasetIOService datasetIOService;

	private final N2VConfig config;
	private N2VProgress dialog;

	private final List< RandomAccessibleInterval< FloatType > > X = new ArrayList<>();
	private final List< RandomAccessibleInterval< FloatType > > validationX = new ArrayList<>();

	public InputHandler(Context context, N2VConfig config) {
		this.config = config;
		context.inject(this);
	}

	void setDialog(N2VProgress dialog) {
		this.dialog = dialog;
	}

	public void addTrainingAndValidationData(RandomAccessibleInterval<FloatType> training, double validationAmount) {
		if (Thread.interrupted()) return;

		logService.info( "Tile training and validation data.." );
		if(dialog != null) dialog.setCurrentTaskMessage("Tiling training and validation data" );

		List< RandomAccessibleInterval< FloatType > > tiles = N2VDataGenerator.createTiles( training, config.getTrainDimensions(), config.getTrainBatchDimLength(), logService );

		int trainEnd = (int) (tiles.size() * (1 - validationAmount));
		for (int i = 0; i < trainEnd; i++) {
			//TODO do I need to copy here?
			X.add( tiles.get( i ) );
		}
		int valEnd = tiles.size()-trainEnd % 2 == 1 ? tiles.size() - 1 : tiles.size();
		for (int i = trainEnd; i < valEnd; i++) {
			//TODO do I need to copy here?
			validationX.add( tiles.get( i ) );
		}
	}

	public void addTrainingAndValidationData(File trainingFolder, double validationAmount) {

		if(trainingFolder.isDirectory()) {
			File[] imgs = trainingFolder.listFiles();
			for (File file : imgs) {
				if (Thread.interrupted()) return;
				try {
					RandomAccessibleInterval img = datasetIOService.open(file.getAbsolutePath()).getImgPlus().getImg();
					addTrainingAndValidationData(convertToFloat(img), validationAmount);
				} catch (IOException e) {
					logService.warn("Could not load " + file.getAbsolutePath() + " as image");
				}
			}
		}
	}

	public static <T extends RealType<T>> RandomAccessibleInterval<FloatType> convertToFloat(RandomAccessibleInterval<T> img) {
		return Converters.convert(img, new RealFloatConverter<T>(), new FloatType());
	}

	public void addTrainingData(RandomAccessibleInterval<FloatType> training) {

		if (Thread.interrupted()) return;

		logService.info( "Tile training data.." );
		if(dialog != null) dialog.setCurrentTaskMessage("Tiling training data" );

		logService.info("Training image dimensions: " + Arrays.toString(Intervals.dimensionsAsIntArray(training)));

		X.addAll(N2VDataGenerator.createTiles( training, config.getTrainDimensions(), config.getTrainBatchDimLength(), logService ));
	}

	public void addTrainingData(File trainingFolder) {

		if(trainingFolder.isDirectory()) {
			File[] imgs = trainingFolder.listFiles();
			for (File file : imgs) {
				if (Thread.interrupted()) return;
				try {
					RandomAccessibleInterval img = datasetIOService.open(file.getAbsolutePath()).getImgPlus().getImg();
					addTrainingData(convertToFloat(img));
				} catch (IOException e) {
					logService.warn("Could not load " + file.getAbsolutePath() + " as image");
				}
			}
		}
	}

	public void addValidationData(RandomAccessibleInterval<FloatType> validation) {

		if (Thread.interrupted()) return;

		logService.info( "Tile validation data.." );
		if(dialog != null) dialog.setCurrentTaskMessage("Tiling validation data" );

		logService.info("Validation image dimensions: " + Arrays.toString(Intervals.dimensionsAsIntArray(validation)));

		validationX.addAll(N2VDataGenerator.createTiles( validation, config.getTrainDimensions(), config.getTrainBatchDimLength(), logService ));
	}

	public void addValidationData(File trainingFolder) {

		if(trainingFolder.isDirectory()) {
			File[] imgs = trainingFolder.listFiles();
			for (File file : imgs) {
				if (Thread.interrupted()) return;
				try {
					RandomAccessibleInterval img = datasetIOService.open(file.getAbsolutePath()).getImgPlus().getImg();
					addValidationData(convertToFloat(img));
				} catch (IOException e) {
					logService.warn("Could not load " + file.getAbsolutePath() + " as image");
				}
			}
		}
	}

	public List<RandomAccessibleInterval<FloatType>> getX() {
		return X;
	}

	public List<RandomAccessibleInterval<FloatType>> getValidationX() {
		return validationX;
	}
}
