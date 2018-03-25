import numpy as np
import re


def getVariable(fileName, variable, type):
    # Return value for variable from parameters.hpp
    with open(fileName) as parameterFile:
        for line in parameterFile:
            splitLine = line.split()
            if (len(splitLine) > 2) and (splitLine[1] == variable):
                if type == 'int':
                    return int(splitLine[2])
                if type == 'long':
                    return long(splitLine[2])
                if type == 'float':
                    return float(splitLine[2])
                if type == 'str':
                    strings = re.findall('\"([^,\"]+)\"',line)
                    if (len(strings) > 1):
                        return strings
                    else:
                        return strings[0]
                if type == 'bool':
                    if splitLine[2] == 'true':
                        return True
                    if splitLine[2] == 'false':
                        return False


def readScalar(fileName):
    # Returns vector with scalar value at each entry
    return np.loadtxt(fileName)


def projectFolder():
    # Returns current project folder
    with open('../src/directories.hpp') as parameterFile:
        for line in parameterFile:
            line = line.split()
            if (len(line) is 3) and (line[1] == 'PROJECT_FOLDER'):
                return str(re.findall('\"([^,\"]+)\"',line[2])[0])


def outputDirectory():
    # Returns current output folder
    with open('../src/directories.hpp') as parameterFile:
        for line in parameterFile:
            line = line.split()
            if (len(line) is 3) and (line[1] == 'OUTPUT_DIRECTORY'):
                return str(re.findall('\"([^,\"]+)\"',line[2])[0])


def nOutputs(outputDirectory):
    # Returns number of outputs (denoted by their tags)
    with open(outputDirectory + 'output.log') as parameterFile:
        for line in parameterFile:
            line = line.split()
            if (len(line) is 2) and (line[0] == 'tag_nr'):
                return int(line[1])


class cell():
    # Python copy of the cell class used in Magritte
    def __init__(self, outputDirectory, tag):
        if (tag != ''): tag = '_' + tag
        # inputFile = getVariable(outputDirectory+'parameters.hpp', 'INPUTFILE', 'str')
        # Initialize cells by reading Magritte output
        self.temperatureGas     = readScalar(outputDirectory + 'temperature_gas' + tag + '.txt')
        self.thermalRatio       = readScalar(outputDirectory + 'thermal_ratio'   + tag + '.txt')
        self.temperatureGasPrev = readScalar(outputDirectory + 'temperature_gas_prev' + tag + '.txt')
        self.thermalRatioPrev   = readScalar(outputDirectory + 'thermal_ratio_prev'   + tag + '.txt')
        # self.abundances         = readVector(outputDirectory + 'the00rmal_ratio'   + tag + '.txt')
        self.ncells             = len(self.temperatureGas)
