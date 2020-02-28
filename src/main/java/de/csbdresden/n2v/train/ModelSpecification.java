package de.csbdresden.n2v.train;

import de.csbdresden.n2v.predict.N2VPrediction;
import net.imglib2.type.numeric.real.FloatType;
import org.yaml.snakeyaml.Yaml;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.ZipFile;

public class ModelSpecification {

	private final static String idName = "name";
	private final static String idDescription = "description";
	private final static String idCite = "cite";
	private final static String idCiteText = "text";
	private final static String idCiteDoi = "doi";
	private final static String idAuthors = "authors";
	private final static String idDocumentation = "documentation";
	private final static String idTags = "tags";
	private final static String idLicense = "license";
	private final static String idFormatVersion = "format_version";
	private final static String idLanguage = "language";
	private final static String idFramework = "framework";
	private final static String idSource = "source";
	private final static String idInputs = "inputs";
	private final static String idOutputs = "outputs";
	private final static String idNodeName = "name";
	private final static String idNodeAxes = "axes";
	private final static String idNodeDataType = "data_type";
	private final static String idNodeDataRange = "data_range";
	private final static String idNodeShape = "shape";
	private final static String idNodeShapeMin = "min";
	private final static String idNodeShapeStep = "step";
	private final static String idNodeHalo = "halo";
	private final static String idNodeShapeReferenceInput = "reference_input";
	private final static String idNodeShapeScale = "scale";
	private final static String idNodeShapeOffset = "offset";
	private final static String idPrediction = "prediction";
	private final static String idPredictionPreprocess = "preprocess";
	private final static String idPredictionProcessSpec = "spec";
	private final static String idPredictionProcessKwargs = "kwargs";
	private final static String idPredictionWeights = "weights";
	private final static String idPredictionWeightsSource = "source";
	private final static String idPredictionWeightsHash = "hash";
	private final static String idPredictionPostprocess = "postprocess";
	private final static String idPredictionDependencies = "dependencies";
	private final static String idMean = "mean";
	private final static String idStdDev = "stdDev";

	private final static String modelName = "N2V";
	private final static String modelFileName = "n2v.model.yaml";
	private final static String dependenciesFileName = "dependencies.yaml";
	private final static String modelDescription = "YOUR DESCRIPTION OF WHAT THIS MODEL WAS TRAINED ON";
	private final static String citationText = "Krull, A. and Buchholz, T. and Jug, F. Noise2void - learning denoising from single noisy images.\n" +
			"Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2019)";
	private final static String doiText = "arXiv:1811.10980";
	private final static List modelAuthors = Arrays.asList("YOUR NAMES HERE");
	private final static String modelDocumentation = "https://github.com/juglab/n2v/";
	private final static List tags = Arrays.asList("denoising", "unet2d");
	private final static String modelLicense = "BSD 3";
	private final static String modelFormatVersion = "0.1.0";
	private final static String modelLanguage = "java";
	private final static String modelFramework = "tensorflow";
	private final static String modelSource = N2VTraining.class.getCanonicalName();
	private final static String modelInputName = "raw";
	private final static String modelDataType = "float32";
	private final static List modelInputDataRange = Arrays.asList("-inf", "inf");
	private final static List modelOutputDataRange = Arrays.asList("-inf", "inf");
	private final static String modelOutputName = "denoised";
	private final static String modelPreprocessing = N2VPrediction.class.getCanonicalName() + "::preprocess";
	private final static String modelPostprocessing = N2VPrediction.class.getCanonicalName() + "::postprocess";
	private final static String modelWeightsSource = "./variables/variables";

	static void writeModelConfigFile(N2VConfig config, OutputHandler outputHandler, File targetDirectory) {
		writeDependenciesFile(targetDirectory);
		Map<String, Object> normalizeArgs = new LinkedHashMap<>();
		normalizeArgs.put(idMean, outputHandler.getMean().get());
		normalizeArgs.put(idStdDev, outputHandler.getStdDev().get());
		List modelInputMin;
		List modelInputStep;
		List modelOutputHalo;
		List modelOutputScale;
		List modelOutputOffset;
		String modelNodeAxes;
		if(config.getTrainDimensions() == 2) {
			modelNodeAxes = "byxc";
			modelInputMin = Arrays.asList(1,4,4,1);
			modelInputStep = Arrays.asList(1,4,4,0);
			modelOutputHalo = Arrays.asList(0, 32, 32, 0);
			modelOutputScale = Arrays.asList(1,1,1,1);
			modelOutputOffset = Arrays.asList(0,0,0,0);
		} else {
			modelNodeAxes = "bzyxc";
			modelInputMin = Arrays.asList(1,4,4,4,1);
			modelInputStep = Arrays.asList(1,4,4,4,0);
			modelOutputHalo = Arrays.asList(0, 32, 32, 32, 0);
			modelOutputScale = Arrays.asList(1,1,1,1, 1);
			modelOutputOffset = Arrays.asList(0,0,0,0, 0);
		}
		Map<String, Object> data = new LinkedHashMap<>();
		data.put(idName, modelName);
		data.put(idDescription, modelDescription);
		Map<String, Object> cite = new LinkedHashMap<>();
		cite.put(idCiteText, citationText);
		cite.put(idCiteDoi, doiText);
		data.put(idCite, cite);
		data.put(idAuthors, modelAuthors);
		data.put(idDocumentation, modelDocumentation);
		data.put(idTags, tags);
		data.put(idLicense, modelLicense);
		data.put(idFormatVersion, modelFormatVersion);
		data.put(idLanguage, modelLanguage);
		data.put(idFramework, modelFramework);
		data.put(idSource, modelSource);
		List<Object> inputs = new ArrayList<>();
		Map<String, Object> input = new LinkedHashMap<>();
		input.put(idNodeName, modelInputName);
		input.put(idNodeAxes, modelNodeAxes);
		input.put(idNodeDataType, modelDataType);
		input.put(idNodeDataRange, modelInputDataRange);
		Map<String, Object> inputShape = new LinkedHashMap<>();
		inputShape.put(idNodeShapeMin, modelInputMin);
		inputShape.put(idNodeShapeStep, modelInputStep);
		input.put(idNodeShape, inputShape);
		inputs.add(input);
		data.put(idInputs, inputs);
		List<Object> outputs = new ArrayList<>();
		Map<String, Object> output = new LinkedHashMap<>();
		output.put(idNodeName, modelOutputName);
		output.put(idNodeAxes, modelNodeAxes);
		output.put(idNodeDataType, modelDataType);
		output.put(idNodeDataRange, modelOutputDataRange);
		output.put(idNodeHalo, modelOutputHalo);
		Map<String, Object> outputShape = new LinkedHashMap<>();
		outputShape.put(idNodeShapeReferenceInput, modelInputName);
		outputShape.put(idNodeShapeScale, modelOutputScale);
		outputShape.put(idNodeShapeOffset, modelOutputOffset);
		output.put(idNodeShape, outputShape);
		outputs.add(output);
		data.put(idOutputs, outputs);
		Map<String, Object> prediction = new LinkedHashMap<>();
		Map<String, Object> preprocess = new LinkedHashMap<>();
		preprocess.put(idPredictionProcessSpec, modelPreprocessing);
		preprocess.put(idPredictionProcessKwargs, new LinkedHashMap<>(normalizeArgs));
		prediction.put(idPredictionPreprocess, preprocess);
		Map<String, Object> weights = new LinkedHashMap<>();
		weights.put(idPredictionWeightsSource, modelWeightsSource);
		prediction.put(idPredictionWeights, weights);
		Map<String, Object> postprocess = new LinkedHashMap<>();
		postprocess.put(idPredictionProcessSpec, modelPostprocessing);
		postprocess.put(idPredictionProcessKwargs, new LinkedHashMap<>(normalizeArgs));
		prediction.put(idPredictionPostprocess, postprocess);
		prediction.put(idPredictionDependencies, "./"+dependenciesFileName);
		data.put(idPrediction, prediction);
		Yaml yaml = new Yaml();
		FileWriter writer = null;
		try {
			writer = new FileWriter(new File(targetDirectory, modelFileName));
		} catch (IOException e) {
			e.printStackTrace();
		}
		yaml.dump(data, writer);
	}

	private static void writeDependenciesFile(File targetDirectory) {
		Map<String, Object> data = new LinkedHashMap<>();
		List<String> dependencies = new ArrayList<>();
		ClassLoader cl = ClassLoader.getSystemClassLoader();
		for(URL url: ((URLClassLoader)cl).getURLs()){
			dependencies.add(url.getPath());
		}
		data.put("classPath", dependencies);
		Yaml yaml = new Yaml();
		FileWriter writer = null;
		try {
			writer = new FileWriter(new File(targetDirectory, dependenciesFileName));
		} catch (IOException e) {
			e.printStackTrace();
		}
		yaml.dump(data, writer);
	}

	public static void readConfig(N2VPrediction prediction, File zippedModel) {
		boolean successful = _readConfig(prediction, zippedModel);
		if(!successful) {
			System.err.println("Could not read data from config file " + zippedModel.getAbsolutePath());
		}
	}

	private static boolean _readConfig(N2VPrediction prediction, File zippedModel) {
		double mean = 0.0f;
		double stdDev = 1.0f;
		int trainDimensions = 2;
		try {
			InputStream stream = extractFile(zippedModel, modelFileName);
			Yaml yaml = new Yaml();
			Map<String, Object> obj = yaml.load(stream);
			System.out.println(obj);
			if(obj == null) return false;
			List inputsObj = (List) obj.get(idInputs);
			Map<String, Object> inputObj = (Map<String, Object>) inputsObj.get(0);
			String axesObj = (String) inputObj.get(idNodeAxes);
			if(axesObj == null) return false;
			trainDimensions = axesObj.length() == 4 ? 2 : 3;
			Map<String, Object> predictionObj = (Map<String, Object>) obj.get(idPrediction);
			if(predictionObj == null) return false;
			Map<String, Object> preprocessObj = (Map<String, Object>) predictionObj.get(idPredictionPreprocess);
			if(preprocessObj == null) return false;
			Map<String, Object> kwargsObj = (Map<String, Object>) preprocessObj.get(idPredictionProcessKwargs);
			Object meanObj = kwargsObj.get(idMean);
			if(meanObj != null) mean = (double) meanObj;
			Object stdDevObj = kwargsObj.get(idStdDev);
			if(stdDevObj != null) stdDev = (double) stdDevObj;
		} catch (IOException e) {
			e.printStackTrace();
		}
		prediction.setMean(new FloatType((float) mean));
		prediction.setStdDev(new FloatType((float) stdDev));
		prediction.setTrainDimensions(trainDimensions);
		return true;
	}

	private static InputStream extractFile(File zipFile, String fileName) throws IOException {
		ZipFile zf = new ZipFile(zipFile);
		return zf.getInputStream(zf.getEntry(fileName));
	}
}
