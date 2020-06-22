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
		remainingTime = average * (numSteps - currentStep);
		int h = (int) ((remainingTime / 1000) / 3600);
		int m = (int) (((remainingTime / 1000) / 60) % 60);
		int s = (int) ((remainingTime / 1000) % 60);
		return String.format("remaining training time: %02d:%02d:%02d", h, m, s);
	}
}
