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

import de.csbdresden.n2v.predict.N2VPrediction;
import net.imagej.modelzoo.ModelZooArchive;
import net.imagej.modelzoo.consumer.DefaultModelZooPrediction;
import net.imagej.modelzoo.consumer.commands.AbstractSingleImagePredictionCommand;
import net.imagej.modelzoo.consumer.commands.DefaultSingleImagePredictionCommand;
import net.imagej.modelzoo.consumer.commands.SingleImagePredictionCommand;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import org.scijava.ItemIO;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.File;
import java.io.IOException;

@Plugin( type = SingleImagePredictionCommand.class, name = "n2v", menuPath = "Plugins>CSBDeep>N2V>N2V predict" )
public class N2VPredictCommand <T extends RealType<T> & NativeType<T>> extends AbstractSingleImagePredictionCommand<T, N2VPrediction> {

	@Parameter(type = ItemIO.OUTPUT)
	private RandomAccessibleInterval<T> output;

	@Override
	public void run() {
		try {
			validateTrainedModel(getModelFile());
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
		super.run();
	}

	@Override
	protected N2VPrediction createPrediction() {
		return new N2VPrediction(getContext());
	}

	@Override
	protected void createOutput(N2VPrediction prediction) {
		output = (RandomAccessibleInterval<T>) prediction.getOutput().getImage();
	}

	private void validateTrainedModel(File trainedModel) throws IOException {
		ModelZooArchive model = modelZooService().io().open(trainedModel);
		if(model.getSpecification().getFormatVersion().equals("0.1.0")) {
			log().error("Deprecated model format - please call Plugins > CSBDeep > N2V > Upgrade N2V model.");
			return;
		}
		if(isMultiChannel()) {
			log().error("Can't predict multichannel images. This will be implemented in the future.");
		}
	}

	private boolean isMultiChannel() {
		int channelIndex = getAxes().indexOf("C");
		if(channelIndex < 0) return false;
		if(getPrediction().getInput().getImage().numDimensions() <= channelIndex) return false;
		return getPrediction().getInput().getImage().dimension(channelIndex) > 1;
	}
}
