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

import com.google.common.io.Files;
import net.imagej.ImageJ;
import net.imagej.modelzoo.ModelZooArchive;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.net.URISyntaxException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class TestUpgradeN2VModelCommand {

	@Rule
	public TemporaryFolder folder = new TemporaryFolder();

	@Test
	public void testUpgrade() {
		final ImageJ ij = new ImageJ();
		File modelFile = new File(getClass().getResource("/format-0.1.0_model.zip").getPath());
		UpgradeN2VModelCommand upgrader = new UpgradeN2VModelCommand();
		ij.context().inject(upgrader);
		File destinationFolder = folder.getRoot();
		ModelZooArchive model = upgrader.tryUpgrade(modelFile, destinationFolder, "tmp");
		assertNotNull(model);
		assertNotNull(model.getSpecification());
		assertEquals("0.2.1-csbdeep", model.getSpecification().getFormatVersion());
		assertNotNull(model.getSpecification().getPredictionPreprocessing());
		assertEquals(1, model.getSpecification().getPredictionPreprocessing().size());
		assertNotNull(model.getSpecification().getPredictionPreprocessing().get(0));
		assertNotNull(model.getSpecification().getPredictionPreprocessing().get(0).getKwargs());
		assertNotNull(model.getSpecification().getPredictionPreprocessing().get(0).getKwargs().get("mean"));
		assertEquals(100, model.getSpecification().getPredictionPreprocessing().get(0).getKwargs().get("mean"));
		assertNotNull(model.getSpecification().getPredictionPreprocessing().get(0).getKwargs().get("stdDev"));
		assertEquals(200, model.getSpecification().getPredictionPreprocessing().get(0).getKwargs().get("stdDev"));
		ij.context().dispose();
	}

}
