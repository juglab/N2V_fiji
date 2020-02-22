package de.csbdresden.n2v;

import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;

public class RemainingTimeEstimator {

	private int numSteps;
	private long time;
	private long[] history = new long[5];
	private int cursor = 0;
	private boolean firstVal = true;
	private long remainingTime; // in ms

	public void setCurrentStep(int step) {
		long currentTime = System.currentTimeMillis();
		long dif = currentTime - time;
		if(firstVal) {
			firstVal = false;
			Arrays.fill(history, dif);
		} else {
			history[cursor] = dif;
		}
		cursor = (cursor + 1) % history.length;
		remainingTime = average(history) * (numSteps - step);
		time = currentTime;
	}

	private static long average(long[] history) {
		long res = 0;
		for (long l : history) {
			res += l;
		}
		return res / history.length;
	}

	public void setNumSteps(int numSteps) {
		this.numSteps = numSteps;
	}

	public void setHistoryLength(int length) {
		history = new long[length];
		firstVal = true;
	}

	public String getRemainingTimeString() {
		int h = (int) ((remainingTime / 1000) / 3600);
		int m = (int) (((remainingTime / 1000) / 60) % 60);
		int s = (int) ((remainingTime / 1000) % 60);
		return String.format("%02d:%02d:%02d", h, m, s);
	}

	public void start() {
		time = System.currentTimeMillis();
	}
}
