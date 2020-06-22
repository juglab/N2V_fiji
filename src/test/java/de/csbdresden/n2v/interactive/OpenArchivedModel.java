package de.csbdresden.n2v.interactive;

import net.imagej.ImageJ;
import org.junit.After;
import org.junit.Test;

import java.io.IOException;

public class OpenArchivedModel {
	private ImageJ ij;

	@After
	public void tearDown() {
		ij.context().dispose();
	}

	@Test
	public void run() throws IOException {

		ij = new ImageJ();
		ij.launch();

		// resource paths
		String modelPath = "/home/random/Documents/2020-06 NEUBIAS/models/n2v.bioimage.io.zip";

		Object model = ij.io().open(modelPath);
		ij.ui().show(model);

	}

	public static void main(String... args) throws IOException {
		new OpenArchivedModel().run();
	}
}
