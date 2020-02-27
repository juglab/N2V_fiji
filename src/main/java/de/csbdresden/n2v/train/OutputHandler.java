package de.csbdresden.n2v.train;

import de.csbdresden.n2v.util.N2VUtils;
import net.imglib2.type.numeric.real.FloatType;
import org.apache.commons.compress.utils.IOUtils;
import org.apache.commons.io.FileUtils;
import org.scijava.util.POM;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import org.yaml.snakeyaml.Yaml;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class OutputHandler {
	private FloatType mean = new FloatType();

	private FloatType stdDev = new FloatType();

	private float currentLearningRate = 0.0004f;
	private float currentLoss = Float.MAX_VALUE;
	private float currentAbs = Float.MAX_VALUE;
	private float currentMse = Float.MAX_VALUE;
	private float currentValidationLoss = Float.MAX_VALUE;
	private float bestValidationLoss = Float.MAX_VALUE;
	private final int trainDimensions;

	private File mostRecentModelDir;
	private File bestModelDir;
	private boolean noCheckpointSaved = true;
	private Tensor< String > checkpointPrefix;
	private boolean checkpointExists;

	public OutputHandler(int trainDimensions) {
		this.trainDimensions = trainDimensions;
	}

	void writeModelConfigFile() {
		Map<String, Object> data = new HashMap<>();
		data.put("name", "N2V");
		POM pom = POM.getPOM(N2VTraining.class);
		data.put("version", pom != null ? pom.getVersion() : "");
		data.put("mean", mean.get());
		data.put("stdDev", stdDev.get());
		data.put("trainDimensions", trainDimensions);
		Yaml yaml = new Yaml();
		FileWriter writer = null;
		try {
			writer = new FileWriter(new File(mostRecentModelDir, "config.yaml"));
		} catch (IOException e) {
			e.printStackTrace();
		}
		yaml.dump(data, writer);
	}

	public File exportLatestTrainedModel() throws IOException {
		if(noCheckpointSaved) return null;
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

		String predictionGraphDir = trainDimensions == 2 ? "prediction_2d" : "prediction_3d";
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
		String graphName = trainDimensions == 2 ? "graph_2d.pb" : "graph_3d.pb";
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

	void saveCheckpoint(Session sess) {
		sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/control_dependency").run();
		noCheckpointSaved = false;
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
