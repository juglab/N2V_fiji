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

import de.csbdresden.n2v.util.N2VUtils;
import io.scif.img.ImgSaver;
import net.imagej.modelzoo.specification.DefaultModelSpecification;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.numeric.real.FloatType;
import org.apache.commons.compress.utils.IOUtils;
import org.apache.commons.io.FileUtils;
import org.scijava.Context;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Date;

public class OutputHandler {
	private final N2VConfig config;
	private final N2VTraining training;
	private FloatType mean = new FloatType();

	private FloatType stdDev = new FloatType();

	private float currentLearningRate = 0.0004f;
	private float currentLoss = Float.MAX_VALUE;
	private float currentAbs = Float.MAX_VALUE;
	private float currentMse = Float.MAX_VALUE;
	private float currentValidationLoss = Float.MAX_VALUE;
	private float bestValidationLoss = Float.MAX_VALUE;

	private File mostRecentModelDir;
	private File bestModelDir;
	private boolean noCheckpointSaved = true;
	private Tensor< String > checkpointPrefix;
	private boolean checkpointExists;
	private ImgSaver imgSaver;

	public OutputHandler(N2VConfig config, N2VTraining training, Context context) {
		this.config = config;
		this.currentLearningRate = config.getLearningRate();
		this.training = training;
		imgSaver = new ImgSaver(context);
	}

	public File exportLatestTrainedModel() throws IOException {
		if(noCheckpointSaved) return null;
		N2VModelSpecification spec = new N2VModelSpecification();
		spec.setName(new Date().toString() + " last checkpoint");
		spec.writeModelConfigFile(config, this, mostRecentModelDir, training.getStepsFinished());
		return N2VUtils.saveTrainedModel(mostRecentModelDir);
	}

	public File exportBestTrainedModel() throws IOException {
		if(noCheckpointSaved) return null;
		return N2VUtils.saveTrainedModel(bestModelDir);
	}

	void copyBestModel(N2VTraining training) {
		if(bestValidationLoss > currentValidationLoss) {
			bestValidationLoss = currentValidationLoss;
			try {
				FileUtils.copyDirectory(mostRecentModelDir, bestModelDir);
				N2VModelSpecification spec = new N2VModelSpecification();
				spec.setName(new Date().toString() + " lowest loss");
				spec.writeModelConfigFile(config, this, bestModelDir, training.getStepsFinished());
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	void createSavedModelDirs() throws IOException {
		bestModelDir = Files.createTempDirectory("n2v-best-").toFile();
		String checkpointDir = Files.createTempDirectory("n2v-latest-").toAbsolutePath().toString() + File.separator + "variables";
		checkpointPrefix = Tensors.create(Paths.get(checkpointDir, "variables").toString());
		mostRecentModelDir = new File(checkpointDir).getParentFile();

		checkpointExists = false;

		String predictionGraphDir = config.getTrainDimensions() == 2 ? "n2v_prediction_2d" : "n2v_prediction_3d";
		byte[] predictionGraphDef = IOUtils.toByteArray( getClass().getResourceAsStream("/" + predictionGraphDir + "/saved_model.pb") );
		FileUtils.writeByteArrayToFile(new File(mostRecentModelDir, "saved_model.pb"), predictionGraphDef);
		FileUtils.writeByteArrayToFile(new File(mostRecentModelDir, "training_model.pb"), predictionGraphDef);
	}

	public void createSavedModelDirsFromExisting(File trainedModel) throws IOException {
		mostRecentModelDir = trainedModel;
		bestModelDir = Files.createTempDirectory("n2v-best-").toFile();
		String checkpointDir = mostRecentModelDir.getAbsolutePath() + File.separator + "variables";
		checkpointPrefix = Tensors.create(Paths.get(checkpointDir, "variables").toString());

		checkpointExists = true;

		byte[] predictionGraphDef = IOUtils.toByteArray( new FileInputStream(new File(trainedModel, "saved_model.pb")));
		FileUtils.writeByteArrayToFile(new File(mostRecentModelDir, "saved_model.pb"), predictionGraphDef);
		FileUtils.writeByteArrayToFile(new File(mostRecentModelDir, "training_graph.pb"), predictionGraphDef);
	}

	void loadUntrainedGraph(Graph graph) throws IOException {
		String graphName = config.getTrainDimensions() == 2 ? "n2v_graph_2d.pb" : "n2v_graph_3d.pb";
		byte[] graphDef = IOUtils.toByteArray( getClass().getResourceAsStream("/" + graphName) );
		graph.importGraphDef( graphDef );
//		graph.operations().forEachRemaining( op -> {
//			for ( int i = 0; i < op.numOutputs(); i++ ) {
//				Output< Object > opOutput = op.output( i );
//				String name = opOutput.op().name();
//				logService.info( name );
//			}
//		} );
	}

	File loadTrainedGraph(Graph graph, File zipFile) throws IOException {

		File trainedModel = Files.createTempDirectory("n2v-imported-model").toFile();
		N2VUtils.unZipAll(zipFile, trainedModel);

		byte[] graphDef = new byte[ 0 ];
		try {
			graphDef = IOUtils.toByteArray( new FileInputStream(new File(trainedModel, "training_graph.pb")));
		} catch ( IOException e ) {
			e.printStackTrace();
		}
		graph.importGraphDef( graphDef );

//		graph.operations().forEachRemaining( op -> {
//			for ( int i = 0; i < op.numOutputs(); i++ ) {
//				Output< Object > opOutput = op.output( i );
//				String name = opOutput.op().name();
//				logService.info( name );
//			}
//		} );
		return trainedModel;
	}


	public void initTensors(Session sess) {
		if (checkpointExists) {
			sess.runner()
					.feed("save/Const", checkpointPrefix)
					.addTarget("save/restore_all").run();
		} else {
			sess.runner().addTarget("init").run();
		}
	}

	void saveCheckpoint(Session sess, RandomAccessibleInterval<FloatType> input, RandomAccessibleInterval<FloatType> output) {
		sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/control_dependency").run();
		noCheckpointSaved = false;
		if(input != null && output != null) {
			File imgIn = new File(mostRecentModelDir, new DefaultModelSpecification().getTestInput());
			if(imgIn.exists()) imgIn.delete();
			imgSaver.saveImg(imgIn.getAbsolutePath(),
					toImg(input));
			File imgOut = new File(mostRecentModelDir, new DefaultModelSpecification().getTestOutput());
			if(imgOut.exists()) imgOut.delete();
			imgSaver.saveImg(imgOut.getAbsolutePath(),
					toImg(output));
		}
	}

	private Img<?> toImg(RandomAccessibleInterval<FloatType> input) {
		ArrayImg<FloatType, ?> res = new ArrayImgFactory<>(new FloatType()).create(input);
		LoopBuilder.setImages(input, res).forEachPixel((in, out) -> {
			out.set(in);
		});
		return res;
	}

	public float getCurrentLoss() {
		return currentLoss;
	}

	public float getCurrentMse() {
		return currentMse;
	}

	public float getCurrentAbs() {
		return currentAbs;
	}

	public float getCurrentLearningRate() {
		return currentLearningRate;
	}

	public void setCurrentLearningRate(float rate) {
		currentLearningRate = rate;
	}

	public FloatType getMean() {
		return mean;
	}

	public FloatType getStdDev() {
		return stdDev;
	}

	public void setCurrentLoss(float loss) {
		this.currentLoss = loss;
	}

	public void setCurrentAbs(float abs) {
		this.currentAbs = abs;
	}

	public void setCurrentMse(float mse) {
		this.currentMse = mse;
	}

	public float getCurrentValidationLoss() {
		return currentValidationLoss;
	}

	public void setCurrentValidationLoss(float loss) {
		this.currentValidationLoss = loss;
	}

	public void dispose() {
		if(checkpointPrefix != null) checkpointPrefix.close();
	}
}
