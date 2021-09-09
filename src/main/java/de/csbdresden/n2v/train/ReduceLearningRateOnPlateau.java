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

public class ReduceLearningRateOnPlateau {

	private float factor = 0.5f;
	private int min_lr = 0;
	private float min_delta = 0.0001f;
	private int patience = 10;
	private boolean verbose = true;
	private int cooldown = 0;
	private int cooldown_counter = 0;
	private int wait = 0;
	private float best = 0;
	MonitorOp monitorOp;

	interface MonitorOp {
		boolean accept(float a, float b);
	}

	public ReduceLearningRateOnPlateau() {
		reset();
	}

	public void reduceLearningRateOnPlateau(N2VTraining training) {
		if(inCooldown()) {
			cooldown_counter -= 1;
			wait = 0;
		}
		if(monitorOp.accept(training.output().getCurrentValidationLoss(), best)) {
			best = training.output().getCurrentValidationLoss();
			wait = 0;
		} else {
			if(!inCooldown()) {
				wait += 1;
				if(wait >= patience) {
					float oldLR = training.output().getCurrentLearningRate();
					if(oldLR > min_lr) {
						float newLR = oldLR * factor;
						newLR = Math.max(newLR, min_lr);
						training.setLearningRate(newLR);
						if(verbose) {
							System.out.println("Reducing learning rate to " + newLR);
						}
						cooldown_counter = cooldown;
						wait = 0;
					}
				}
			}
		}
	}

	private boolean monitorOp1(float a, float b) {
		return a < b - min_delta;
	}

	private boolean inCooldown() {
		return cooldown_counter > 0;
	}

	private void reset() {
		monitorOp = this::monitorOp1;
		best = Float.MAX_VALUE;
		cooldown_counter = 0;
		wait = 0;
	}

}
