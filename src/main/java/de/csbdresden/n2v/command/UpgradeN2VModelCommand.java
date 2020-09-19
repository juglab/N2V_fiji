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
package de.csbdresden.n2v.command;

import de.csbdresden.n2v.train.N2VConfig;
import de.csbdresden.n2v.train.N2VModelSpecification;
import net.imagej.modelzoo.ModelZooArchive;
import net.imagej.modelzoo.ModelZooService;
import net.imagej.modelzoo.specification.ModelSpecification;
import org.scijava.Context;
import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.io.location.Location;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

import static org.scijava.widget.FileWidget.DIRECTORY_STYLE;

@Plugin( type = Command.class, menuPath = "Plugins>CSBDeep>N2V>Upgrade old N2V model" )
public class UpgradeN2VModelCommand implements Command {

	@Parameter(label = "Old trained model file (bioimage.io.zip)")
	private File modelFile;

	@Parameter(label = "Destination directory", style = DIRECTORY_STYLE)
	private File destinationFolder;

	@Parameter(label = "Destination file name (without .zip ending)")
	private String destinationFileName;

	@Parameter( type = ItemIO.OUTPUT )
	private ModelZooArchive output;

	@Parameter
	private Context context;

	@Parameter
	private ModelZooService modelZooService;

	@Parameter
	private LogService logService;

	@Override
	public void run() {
		output = tryUpgrade(modelFile, destinationFolder, destinationFileName);
	}

	ModelZooArchive tryUpgrade(File modelFile, File destinationFolder, String destinationFileName) {
		ModelZooArchive model = null;
		try {
			model = modelZooService.open(modelFile);
		} catch (IOException e) {
			e.printStackTrace();
		}
		ModelSpecification oldSpec = model.getSpecification();
		if(oldSpec.getFormatVersion().equals("0.1.0")) {
			try {
				return upgrade(model, destinationFolder, destinationFileName);
			} catch (IOException e) {
				e.printStackTrace();
			}
		} else {
			if(oldSpec.getFormatVersion().equals("0.2.0-csbdeep")) {
				logService.info("Model format is already the newest version.");
			} else {
				logService.error("Unknown model format version " + oldSpec.getFormatVersion());
			}
		}
		return null;
	}

	private ModelZooArchive upgrade(ModelZooArchive model, File destinationFolder, String destinationFileName) throws IOException {
		ModelSpecification oldSpec = model.getSpecification();
		File destination = new File(destinationFolder, destinationFileName + ".bioimage.io.zip");
		if (destination.getAbsolutePath().equals(getAbsolutePath(model.getLocation()))) {
			logService.error("Destination file cannot be the same as the deprecated model ZIP file");
			return model;
		}
		Files.copy(new File(model.getLocation().getURI()).toPath(), destination.toPath(), StandardCopyOption.REPLACE_EXISTING);
		N2VModelSpecification newSpec = new N2VModelSpecification();
		newSpec.setMeta();
		newSpec.setInputsOutputs(new N2VConfig());
		newSpec.setName(oldSpec.getName());
		newSpec.setDescription(oldSpec.getDescription());
		newSpec.setAuthors(oldSpec.getAuthors());
		newSpec.setTags(oldSpec.getTags());
		newSpec.getPredictionPreprocessing().addAll(oldSpec.getPredictionPreprocessing());
		newSpec.getPredictionPostprocessing().addAll(oldSpec.getPredictionPostprocessing());
		try (FileSystem fileSystem = FileSystems.newFileSystem(destination.toPath(), null)) {
			Path specPath = fileSystem.getPath(oldSpec.getModelFileName());
			Files.delete(specPath);
			newSpec.write(fileSystem.getPath(newSpec.getModelFileName()));
		}
		return modelZooService.open(destination);
	}

	private String getAbsolutePath(Location source) {
		return new File(source.getURI()).getAbsolutePath();
	}

}
