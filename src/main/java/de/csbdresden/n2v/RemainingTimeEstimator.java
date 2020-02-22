package de.csbdresden.n2v;

import java.util.Arrays;

public class RemainingTimeEstimator {

	private int numSteps;
	private long time;
	private long[] history = new long[5];
	private int cursor = 0;
	private boolean firstVal = true;
	private long remainingTime; // in ms
	private int currentStep = 0;

	public void setCurrentStep(int step) {
		currentStep = step;
		long currentTime = System.currentTimeMillis();
		if(step > 0) {
			long dif = currentTime - time;
			int h = (int) ((dif / 1000) / 3600);
			int m = (int) (((dif / 1000) / 60) % 60);
			int s = (int) ((dif / 1000) % 60);
			System.out.println(String.format("time of step: %02d:%02d:%02d", h, m, s));
			if(firstVal) {
				firstVal = false;
				Arrays.fill(history, dif);
			} else {
				history[cursor] = dif;
			}
			cursor = (cursor + 1) % history.length;
		}
		time = currentTime;
	}

	private static long average(long[] history) {
		long res = 0;
		for (int i = 0; i < history.length; i++) {
			res += history[i];
		}
		return res / history.length;
	}

	public void setNumSteps(int numSteps) {
		this.numSteps = numSteps;
	}

	public String getRemainingTimeString() {
		if(currentStep == 0) return "";
		long average = average(history);
		remainingTime = (long) (average * (numSteps-1 - currentStep));
		int h = (int) ((remainingTime / 1000) / 3600);
		int m = (int) (((remainingTime / 1000) / 60) % 60);
		int s = (int) ((remainingTime / 1000) % 60);
		return String.format("remaining training time: %02d:%02d:%02d", h, m, s);
	}
}
