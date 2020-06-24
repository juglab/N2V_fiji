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
package de.csbdresden.n2v.predict;

import de.csbdresden.n2v.train.N2VModelSpecification;
import de.csbdresden.n2v.train.TrainUtils;
import io.scif.MissingLibraryException;
import net.imagej.modelzoo.ModelZooArchive;
import net.imagej.modelzoo.consumer.DefaultSingleImagePrediction;
import net.imagej.modelzoo.consumer.ModelZooPrediction;
import net.imagej.ops.OpService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import org.scijava.Context;
import org.scijava.command.CommandService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.FileNotFoundException;

@Plugin(type = ModelZooPrediction.class, name = "n2v")
public class N2VPrediction extends DefaultSingleImagePrediction<FloatType, FloatType> {

	private FloatType mean;
	private FloatType stdDev;

	@Parameter
	private OpService opService;

	@Parameter
	private CommandService commandService;

	@Parameter
	private Context context;

	public N2VPrediction(Context context) {
		super(context);
	}

	@Override
	public void setTrainedModel(ModelZooArchive trainedModel) {
		super.setTrainedModel(trainedModel);
		N2VModelSpecification.setFromSpecification(this, trainedModel.getSpecification());
	}

	public void setMean(FloatType mean) {
		this.mean = mean;
	}

	public void setStdDev(FloatType stdDev) {
		this.stdDev = stdDev;
	}

	@Override
	public void setInput(String name, RandomAccessibleInterval<?> value, String axes) {
		preprocessInput(value, mean, stdDev);
		super.setInput(name, value, axes);
	}


	@Override
	public void run() throws OutOfMemoryError, FileNotFoundException, MissingLibraryException {
		super.run();
		postprocessOutput(getOutput(), mean, stdDev);
	}

	private void preprocessInput(RandomAccessibleInterval input, FloatType mean, FloatType stdDev) {
		TrainUtils.normalizeInplace(input, mean, stdDev);
	}

	private void postprocessOutput(RandomAccessibleInterval<FloatType> output, FloatType mean, FloatType stdDev) {
		if(output == null) return;
		TrainUtils.denormalizeInplace(output, mean, stdDev, opService);
	}

	public RandomAccessibleInterval<FloatType> predictPadded(RandomAccessibleInterval<FloatType> input, String axes) throws FileNotFoundException, MissingLibraryException {
		if(getTrainedModel() == null) return null;
		setInput(input, axes);
		run();
		if(getOutput() == null) return null;
		return getOutput();
	}
}
