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
