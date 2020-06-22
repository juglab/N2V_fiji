###
# #%L
# N2V plugin
# %%
# Copyright (C) 2019 - 2020 Center for Systems Biology Dresden
# %%
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# #L%
###
# @File(label="Input path (directory with .tifs)", style="directory") input
# @File(label="Output path (directory)", style="directory") output
# @File(label="Model file") modelFile
# @DatasetIOService io
# @CommandService command
# @ModuleService module

from java.io import File
import sys
from de.csbdresden.n2v.command import N2VPredictCommand

def getFileName(path):
	fileparts = path.split("/")
	return fileparts[len(fileparts)-1]

def runNetwork(inputFile, outputFile):
	print("input: " + inputFile.getAbsolutePath() + ", output: " + outputFile.getAbsolutePath())
	img = io.open(inputFile.getAbsolutePath())
	mymod = (command.run(N2VPredictCommand, False,
		"input", img,
		"modelFile", modelFile,
		"showProgressDialog", False)).get()
	myoutput = mymod.getOutput("output")
	io.save(myoutput, outputFile.getAbsolutePath())

if(output == input):
	print("ERROR: please provide an output directory that is not the same as the input directory")
	sys.exit()

for file in input.listFiles():
	if file.toString().endswith(".tif"):
		runNetwork(file, File(output, getFileName(file.toString())))
