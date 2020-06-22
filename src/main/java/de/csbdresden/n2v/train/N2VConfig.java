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
