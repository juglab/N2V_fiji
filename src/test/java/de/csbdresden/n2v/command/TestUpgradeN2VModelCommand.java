package de.csbdresden.n2v.command;

import com.google.common.io.Files;
import net.imagej.ImageJ;
import net.imagej.modelzoo.ModelZooArchive;
import org.junit.Test;

import java.io.File;
import java.net.URISyntaxException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class TestUpgradeN2VModelCommand {

	@Test
	public void testUpgrade() {
		final ImageJ ij = new ImageJ();
		File modelFile = new File(getClass().getResource("/format-0.1.0_model.zip").getPath());
		UpgradeN2VModelCommand upgrader = new UpgradeN2VModelCommand();
		ij.context().inject(upgrader);
		File destinationFolder = Files.createTempDir();
		ModelZooArchive model = upgrader.tryUpgrade(modelFile, destinationFolder, "tmp");
		assertNotNull(model);
		assertNotNull(model.getSpecification());
		assertNotNull(model.getSpecification().getPredictionPreprocessing());
		assertNotNull(model.getSpecification().getPredictionPreprocessing().get(0));
		assertNotNull(model.getSpecification().getPredictionPreprocessing().get(0).getKwargs());
		assertNotNull(model.getSpecification().getPredictionPreprocessing().get(0).getKwargs().get("mean"));
		assertEquals(100, model.getSpecification().getPredictionPreprocessing().get(0).getKwargs().get("mean"));
		assertNotNull(model.getSpecification().getPredictionPreprocessing().get(0).getKwargs().get("stdDev"));
		assertEquals(200, model.getSpecification().getPredictionPreprocessing().get(0).getKwargs().get("stdDev"));
		ij.context().dispose();
	}

}
