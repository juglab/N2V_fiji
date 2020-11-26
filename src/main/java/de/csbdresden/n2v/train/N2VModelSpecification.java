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

import io.bioimage.specification.CitationSpecification;
import io.bioimage.specification.DefaultCitationSpecification;
import io.bioimage.specification.DefaultInputNodeSpecification;
import io.bioimage.specification.DefaultModelSpecification;
import io.bioimage.specification.DefaultOutputNodeSpecification;
import io.bioimage.specification.InputNodeSpecification;
import io.bioimage.specification.OutputNodeSpecification;
import io.bioimage.specification.WeightsSpecification;
import io.bioimage.specification.transformation.ImageTransformation;
import io.bioimage.specification.transformation.ScaleLinearTransformation;
import io.bioimage.specification.transformation.ZeroMeanUnitVarianceTransformation;
import io.bioimage.specification.weights.TensorFlowSavedModelBundleSpecification;

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
	private final static String idTrainingKwargsPatchShape = "patchShape";
	private final static String idTrainingKwargsNeighborhoodRadius = "neighborhoodRadius";
	private final static String idTrainingKwargsStepsFinished = "stepsFinished";

	private final static String citationText = "Krull, A. and Buchholz, T. and Jug, F. Noise2void - learning denoising from single noisy images.\n" +
			"Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2019)";
	private final static String doiText = "arXiv:1811.10980";
	private final static String source = "n2v";
	private final static List tags = Arrays.asList("denoising", "unet2d");
	private final static String modelTrainingSource = N2VTraining.class.getCanonicalName();
	private final static String modelInputName = N2VTraining.tensorXOpName;
	private final static String modelDataType = "float32";
	private final static List modelInputDataRange = Arrays.asList("-inf", "inf");
	private final static List modelOutputDataRange = Arrays.asList("-inf", "inf");
	private final static String modelOutputName = N2VTraining.predictionTargetOpName;


	void writeModelConfigFile(N2VConfig config, N2VOutputHandler outputHandler, File targetDirectory, int stepsFinished) throws IOException {
		super.write(targetDirectory);
	}

	void update(N2VConfig config, N2VOutputHandler outputHandler, int stepsFinished) {
		setMeta(outputHandler);
		setInputsOutputs(config, outputHandler);
		setTraining(config, stepsFinished);
		setWeights(outputHandler);
	}

	private void setWeights(N2VOutputHandler outputHandler) {
		WeightsSpecification weights = new TensorFlowSavedModelBundleSpecification();
		weights.setSource(outputHandler.getSavedModelBundlePackage());
		addWeights(weights);
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

	public void setInputsOutputs(N2VConfig config, N2VOutputHandler outputHandler) {
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
		ZeroMeanUnitVarianceTransformation preprocessing = new ZeroMeanUnitVarianceTransformation();
		preprocessing.setMode(ImageTransformation.Mode.FIXED);
		preprocessing.setMean(outputHandler.getMean().get());
		preprocessing.setStd(outputHandler.getStdDev().get());
		inputNode.setPreprocessing(Collections.singletonList(preprocessing));
		addInputNode(inputNode);
		OutputNodeSpecification outputNode = new DefaultOutputNodeSpecification();
		outputNode.setName(modelOutputName);
		outputNode.setAxes(modelNodeAxes);
		outputNode.setDataType(modelDataType);
		outputNode.setDataRange(modelOutputDataRange);
		outputNode.setShapeReferenceInput(modelInputName);
		outputNode.setShapeScale(modelOutputScale);
		outputNode.setShapeOffset(modelOutputOffset);
		ScaleLinearTransformation postprocessing = new ScaleLinearTransformation();
		postprocessing.setMode(ImageTransformation.Mode.FIXED);
		postprocessing.setOffset(outputHandler.getMean().get());
		postprocessing.setGain(outputHandler.getStdDev().get());
		outputNode.setPostprocessing(Collections.singletonList(postprocessing));
		addOutputNode(outputNode);
	}

	public void setMeta(N2VOutputHandler outputHandler) {
		CitationSpecification citation = new DefaultCitationSpecification();
		citation.setCitationText(citationText);
		citation.setDOIText(doiText);
		addCitation(citation);
		setTags(tags);
		setSource(source);
		setSampleInputs(outputHandler.getSampleInputNames());
		setSampleOutputs(outputHandler.getSampleOutputNames());
	}

}
