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

public class N2VConfig {
	private int numEpochs = 300;
	private int trainBatchSize = 180;
	private int trainPatchShape = 60;
	private int stepsPerEpoch = 200;
	private int neighborhoodRadius = 5;
	private int trainDimensions = 2;
	private float learningRate = 0.0004f;

	private int[] maskSpan = new int[]{0, 0};
	private int networkDepth = 4;

	public N2VConfig setStepsPerEpoch(final int steps) {
		stepsPerEpoch = steps;
		return this;
	}

	public N2VConfig setNumEpochs(final int numEpochs) {
		this.numEpochs = numEpochs;
		return this;
	}

	public N2VConfig setBatchSize(final int batchSize) {
		trainBatchSize = batchSize;
		return this;
	}

	public N2VConfig setPatchShape(final int patchShape) {
		trainPatchShape = patchShape;
		return this;
	}

	public N2VConfig setTrainDimensions(int trainDimensions) {
		this.trainDimensions = trainDimensions;
		return this;
	}

	public N2VConfig setNeighborhoodRadius(int radius) {
		this.neighborhoodRadius = radius;
		return this;
	}

	public N2VConfig setMaskSpan(int[] maskSpan) {
		this.maskSpan = maskSpan;
		return this;
	}

	public N2VConfig setLearningRate(float learningRate) {
		this.learningRate = learningRate;
		return this;
	}

	public int getTrainDimensions() {
		return trainDimensions;
	}

	public int getNumEpochs() {
		return numEpochs;
	}

	public int getStepsPerEpoch() {
		return stepsPerEpoch;
	}

	public int getTrainBatchSize() {
		return trainBatchSize;
	}

	public long getTrainPatchShape() {
		return trainPatchShape;
	}

	public int getNeighborhoodRadius() {
		return neighborhoodRadius;
	}

	public float getLearningRate() {
		return learningRate;
	}

	public int[] getMaskSpan() {
		return maskSpan;
	}

	public int getNetworkDepth() {
		return networkDepth;
	}
}
