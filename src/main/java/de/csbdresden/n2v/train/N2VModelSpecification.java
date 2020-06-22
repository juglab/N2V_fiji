package de.csbdresden.n2v.train;

import de.csbdresden.n2v.predict.N2VPrediction;
import net.imagej.modelzoo.specification.CitationSpecification;
import net.imagej.modelzoo.specification.DefaultCitationSpecification;
import net.imagej.modelzoo.specification.DefaultInputNodeSpecification;
import net.imagej.modelzoo.specification.DefaultModelSpecification;
import net.imagej.modelzoo.specification.DefaultOutputNodeSpecification;
import net.imagej.modelzoo.specification.DefaultTransformationSpecification;
import net.imagej.modelzoo.specification.InputNodeSpecification;
import net.imagej.modelzoo.specification.ModelSpecification;
import net.imagej.modelzoo.specification.OutputNodeSpecification;
import net.imagej.modelzoo.specification.TransformationSpecification;
import net.imglib2.type.numeric.real.FloatType;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class N2VModelSpecification extends DefaultModelSpecification {

	private final static String idTrainingKwargsTrainDimensions = "trainDimensions";
	private final static String idTrainingKwargsLearningRate = "learningRate";
	private final static String idTrainingKwargsNumEpochs = "numEpochs";
	private final static String idTrainingKwargsNumStepsPerEpoch = "numStepsPerEpoch";
	private final static String idTrainingKwargsBatchSize = "batchSize";
	private final static String idTrainingKwargsBatchDimLength = "batchDimLength";
	private final static String idTrainingKwargsPatchShape = "patchShape";
	private final static String idTrainingKwargsNeighborhoodRadius = "neighborhoodRadius";
	private final static String idTrainingKwargsStepsFinished = "stepsFinished";
	private final static String idMean = "mean";
	private final static String idStdDev = "stdDev";

	private final static String citationText = "Krull, A. and Buchholz, T. and Jug, F. Noise2void - learning denoising from single noisy images.\n" +
			"Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2019)";
	private final static String doiText = "arXiv:1811.10980";
	private final static List tags = Arrays.asList("denoising", "unet2d");
	private final static String modelSource = "n2v";
	private final static String modelTrainingSource = N2VTraining.class.getCanonicalName();
	private final static String modelInputName = N2VTraining.tensorXOpName;
	private final static String modelDataType = "float32";
	private final static List modelInputDataRange = Arrays.asList("-inf", "inf");
	private final static List modelOutputDataRange = Arrays.asList("-inf", "inf");
	private final static String modelOutputName = N2VTraining.predictionTargetOpName;
	private final static String modelPreprocessing = N2VPrediction.class.getCanonicalName() + "::preprocess";
	private final static String modelPostprocessing = N2VPrediction.class.getCanonicalName() + "::postprocess";

	void writeModelConfigFile(N2VConfig config, OutputHandler outputHandler, File targetDirectory, int stepsFinished) throws IOException {
		setMeta();
		setInputsOutputs(config);
		setTraining(config, stepsFinished);
		setPrediction(outputHandler);
		super.write(targetDirectory);
	}

	private void setPrediction(OutputHandler outputHandler) {
		TransformationSpecification preprocessing = new DefaultTransformationSpecification();
		preprocessing.setSpec(modelPreprocessing);
		Map<String, Object> normalizeArgs = new LinkedHashMap<>();
		normalizeArgs.put(idMean, Collections.singletonList(outputHandler.getMean().get()));
		normalizeArgs.put(idStdDev, Collections.singletonList(outputHandler.getStdDev().get()));
		preprocessing.setKwargs(new LinkedHashMap<>(normalizeArgs));
		addPredictionPreprocessing(preprocessing);
		TransformationSpecification postprocessing = new DefaultTransformationSpecification();
		postprocessing.setSpec(modelPostprocessing);
		postprocessing.setKwargs(new LinkedHashMap<>(normalizeArgs));
		addPredictionPostprocessing(postprocessing);
	}

	private void setTraining(N2VConfig config, int stepsFinished) {
		setTrainingSource(modelTrainingSource);
		Map<String, Object> trainingKwargs = new LinkedHashMap<>();
		trainingKwargs.put(idTrainingKwargsBatchSize, config.getTrainBatchSize());
		trainingKwargs.put(idTrainingKwargsLearningRate, config.getLearningRate());
		trainingKwargs.put(idTrainingKwargsTrainDimensions, config.getTrainDimensions());
		trainingKwargs.put(idTrainingKwargsNeighborhoodRadius, config.getNeighborhoodRadius());
		trainingKwargs.put(idTrainingKwargsNumEpochs, config.getNumEpochs());
		trainingKwargs.put(idTrainingKwargsNumStepsPerEpoch, config.getStepsPerEpoch());
		trainingKwargs.put(idTrainingKwargsPatchShape, config.getTrainPatchShape());
		trainingKwargs.put(idTrainingKwargsStepsFinished, stepsFinished);
		setTrainingKwargs(trainingKwargs);
	}

	private void setInputsOutputs(N2VConfig config) {
		List<Integer> modelInputMin;
		List<Integer> modelInputStep;
		List<Integer> modelInputHalo;
		List<Float> modelOutputScale;
		List<Integer> modelOutputOffset;
		String modelNodeAxes;
		int min = (int) Math.pow(2, config.getNetworkDepth());
		int halo = 22;
		if(config.getTrainDimensions() == 2) {
			modelNodeAxes = "byxc";
			modelInputMin = Arrays.asList(1, min, min, 1);
			modelInputStep = Arrays.asList(1, min, min, 0);
			modelInputHalo = Arrays.asList(0, halo, halo, 0);
			modelOutputScale = Arrays.asList(1f, 1f, 1f, 1f);
			modelOutputOffset = Arrays.asList(0, 0, 0, 0);
		} else {
			modelNodeAxes = "bzyxc";
			modelInputMin = Arrays.asList(1, min, min, min, 1);
			modelInputStep = Arrays.asList(1, min, min, min, 0);
			modelInputHalo = Arrays.asList(0, halo, halo, halo, 0);
			modelOutputScale = Arrays.asList(1f, 1f, 1f, 1f, 1f);
			modelOutputOffset = Arrays.asList(0, 0, 0, 0, 0);
		}
		InputNodeSpecification inputNode = new DefaultInputNodeSpecification();
		inputNode.setName(modelInputName);
		inputNode.setAxes(modelNodeAxes);
		inputNode.setDataType(modelDataType);
		inputNode.setDataRange(modelInputDataRange);
		inputNode.setHalo(modelInputHalo);
		inputNode.setShapeMin(modelInputMin);
		inputNode.setShapeStep(modelInputStep);
		addInputNode(inputNode);
		OutputNodeSpecification outputNode = new DefaultOutputNodeSpecification();
		outputNode.setName(modelOutputName);
		outputNode.setAxes(modelNodeAxes);
		outputNode.setDataType(modelDataType);
		outputNode.setDataRange(modelOutputDataRange);
		outputNode.setShapeReferenceInput(modelInputName);
		outputNode.setShapeScale(modelOutputScale);
		outputNode.setShapeOffset(modelOutputOffset);
		addOutputNode(outputNode);
	}

	private void setMeta() {
		CitationSpecification citation = new DefaultCitationSpecification();
		citation.setCitationText(citationText);
		citation.setDOIText(doiText);
		addCitation(citation);
		setTags(tags);
		setSource(modelSource);
	}


	public static boolean setFromSpecification(N2VPrediction prediction, ModelSpecification specification) {
		double mean = 0.0f;
		double stdDev = 1.0f;

		List<TransformationSpecification> predictionPreprocessing = specification.getPredictionPreprocessing();
		if(predictionPreprocessing.size() > 0) {
			Map<String, Object> kwargs = predictionPreprocessing.get(0).getKwargs();
			if(kwargs != null) {
				List<? extends Number> meanObj = (List<? extends Number>) kwargs.get(idMean);
				if(meanObj != null && meanObj.size() > 0) mean = meanObj.get(0).doubleValue();
				List<? extends Number> stdDevObj = (List<? extends Number>) kwargs.get(idStdDev);
				if(stdDevObj != null && stdDevObj.size() > 0) stdDev = stdDevObj.get(0).doubleValue();
			}
		}

		prediction.setMean(new FloatType((float) mean));
		prediction.setStdDev(new FloatType((float) stdDev));
		System.out.println("N2V prediction mean  : " + mean);
		System.out.println("N2V prediction stdDev: " + stdDev);

		return true;
	}

}
