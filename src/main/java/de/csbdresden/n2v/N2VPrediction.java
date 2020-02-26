package de.csbdresden.n2v;

import de.csbdresden.csbdeep.commands.GenericNetwork;
import net.imagej.Dataset;
import net.imagej.DefaultDataset;
import net.imagej.ImgPlus;
import net.imagej.ops.OpService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;
import org.scijava.Context;
import org.scijava.command.CommandModule;
import org.scijava.command.CommandService;
import org.scijava.plugin.Parameter;
import org.yaml.snakeyaml.Yaml;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.zip.ZipFile;

public class N2VPrediction {

	private File zippedModel;
	private FloatType mean;
	private FloatType stdDev;

	@Parameter
	private OpService opService;

	@Parameter
	private CommandService commandService;

	@Parameter
	private Context context;

	public N2VPrediction(Context context) {
		context.inject(this);
	}

	public void setModelFile(File zippedModel) {
		this.zippedModel = zippedModel;
		readConfig();
	}

	private void readConfig() {
		double mean = 0.0f;
		double stdDev = 1.0f;
		try {
			InputStream stream = extractFile(zippedModel, "config.yaml");
			Yaml yaml = new Yaml();
			Map<String, Object> obj = yaml.load(stream);
			System.out.println(obj);
			mean = (double) obj.get("mean");
			stdDev = (double) obj.get("stdDev");
		} catch (IOException e) {
			e.printStackTrace();
		}
		setMean(new FloatType((float) mean));
		setStdDev(new FloatType((float) stdDev));
	}

	public InputStream extractFile(File zipFile, String fileName) throws IOException {
		ZipFile zf = new ZipFile(zipFile);
		return zf.getInputStream(zf.getEntry(fileName));
	}

	public void setMean(FloatType mean) {
		this.mean = mean;
	}

	public void setStdDev(FloatType stdDev) {
		this.stdDev = stdDev;
	}

	public RandomAccessibleInterval predict(RandomAccessibleInterval input) {
		Img prediction = (Img) N2VUtils.normalize(input, mean, stdDev, opService);
		Dataset inputDataset = new DefaultDataset(context, new ImgPlus(prediction));
		try {
			final CommandModule module = commandService.run(
					GenericNetwork.class, false,
					"input", inputDataset,
					"normalizeInput", false,
					"modelFile", zippedModel.getAbsolutePath(),
					"blockMultiple", 8,
					"nTiles", 8,
					"overlap", 32,
					"showProgressDialog", false).get();
			if(module.isCanceled()) return null;
			RandomAccessibleInterval<FloatType> output = (RandomAccessibleInterval<FloatType>) module.getOutput("output");
			if(output == null) return null;
			N2VUtils.denormalizeInplace(output, mean, stdDev, opService);
			return output;
		} catch (InterruptedException | ExecutionException e) {
			e.printStackTrace();
		}
		return null;
	}

}
