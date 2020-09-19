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
import de.csbdresden.n2v.train.N2VTraining;
import net.imagej.modelzoo.ModelZooArchive;
import net.imagej.modelzoo.ModelZooService;
import org.scijava.Cancelable;
import org.scijava.Context;
import org.scijava.ItemIO;
import org.scijava.ItemVisibility;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.widget.NumberWidget;

import java.io.File;
import java.io.IOException;

import static org.scijava.widget.FileWidget.DIRECTORY_STYLE;

@Plugin( type = Command.class, menuPath = "Plugins>CSBDeep>N2V>N2V train on folder" )
public class N2VTrainOnFolderCommand implements Command, Cancelable {

	@Parameter(label = "Folder containing images used for training", style = DIRECTORY_STYLE)
	private File training;

	@Parameter(label = "Folder containing images used for validation", style = DIRECTORY_STYLE)
	private File validation;

	@Parameter(label = "Use 3D model instead of 2D")
	private boolean mode3D = false;

	@Parameter(required = false, visibility = ItemVisibility.MESSAGE)
	private String advancedLabel = "<html><br/><span style='font-weight: normal'>Advanced options</span></html>";

	@Parameter(label = "Number of epochs")
	private int numEpochs = 300;

	@Parameter(label = "Number of steps per epoch")
	private int numStepsPerEpoch = 200;

	@Parameter(label = "Batch size per step")
	private int batchSize = 180;

	@Parameter(label = "Patch shape", min = "16", max = "512", stepSize = "16", style= NumberWidget.SLIDER_STYLE)
	private int patchShape = 64;

	@Parameter(label = "Neighborhood radius")
	private int neighborhoodRadius = 5;

	@Parameter(type = ItemIO.OUTPUT, label = "model from last training step")
	private ModelZooArchive latestTrainedModel;

	@Parameter(type = ItemIO.OUTPUT, label = "model with lowest validation loss")
	private ModelZooArchive bestTrainedModel;

	@Parameter
	private Context context;

	@Parameter
	private ModelZooService modelZooService;

	private boolean canceled;

	@Override
	public void run() {
		N2VTraining n2v = new N2VTraining(context);
		n2v.init(new N2VConfig()
				.setTrainDimensions(mode3D ? 3 : 2)
				.setNumEpochs(numEpochs)
				.setStepsPerEpoch(numStepsPerEpoch)
				.setBatchSize(batchSize)
				.setPatchShape(patchShape)
				.setNeighborhoodRadius(neighborhoodRadius));
		try {
			if(training.equals(validation)) {
				System.out.println("Using 10% of training data for validation");
				n2v.input().addTrainingAndValidationData(training, 0.1);
			} else {
				n2v.input().addTrainingData(training);
				n2v.input().addValidationData(validation);
			}
			n2v.train();
		}
		catch(Exception e) {
			n2v.dispose();
			e.printStackTrace();
			return;
		}
		try {
			File savedModel = n2v.output().exportLatestTrainedModel();
			if(savedModel == null) return;
			openSavedModels(n2v, savedModel);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void openSavedModels(N2VTraining training, File savedModel) throws IOException {
		latestTrainedModel = modelZooService.open(savedModel);
		savedModel = training.output().exportBestTrainedModel();
		bestTrainedModel = modelZooService.open(savedModel);
	}

	@Override
	public boolean isCanceled() {
		return canceled;
	}

	@Override
	public void cancel(String reason) {
		canceled = true;
	}

	@Override
	public String getCancelReason() {
		return null;
	}

}
