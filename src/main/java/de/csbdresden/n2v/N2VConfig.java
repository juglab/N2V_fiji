package de.csbdresden.n2v;

public class N2VConfig {
	private int numEpochs = 300;
	private int trainBatchSize = 180;
	private int trainBatchDimLength = 180;
	private int trainPatchDimLength = 60;
	private int stepsPerEpoch = 200;
	private int neighborhoodRadius = 5;
	private int trainDimensions = 2;

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

	public N2VConfig setPatchDimLength(final int patchDimLength) {
		trainPatchDimLength = patchDimLength;
		return this;
	}

	public N2VConfig setBatchDimLength(final int batchDimLength) {
		trainBatchDimLength = batchDimLength;
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

	public long getTrainBatchDimLength() {
		return trainBatchDimLength;
	}

	public long getTrainPatchDimLength() {
		return trainPatchDimLength;
	}

	public int getNeighborhoodRadius() {
		return neighborhoodRadius;
	}
}
