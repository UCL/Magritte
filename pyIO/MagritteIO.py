import numpy as np
import re
import os


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


def readVector(fileName):
    # Returns matrix with vector value at each entry
    return np.loadtxt(fileName)

def projectFolder():
    # Returns current project folder
    dirPath = os.path.dirname(os.path.realpath(__file__))
    with open(dirPath+'/../src/directories.hpp') as parameterFile:
        for line in parameterFile:
            line = line.split()
            if (len(line) is 3) and (line[1] == 'PROJECT_FOLDER'):
                return str(re.findall('\"([^,\"]+)\"',line[2])[0])


def outputDirectory():
    # Returns current output folder
    dirPath = os.path.dirname(os.path.realpath(__file__))
    with open(dirPath+'/../src/directories.hpp') as parameterFile:
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

def fileExtension(fileName):
    # Returns the file extension given a fileName
    return os.path.splitext(fileName)[1]

def deVectorize(vector, length):
    # Devectorize vector of length L into an array containing 'length' arrays of length L / 'length'
    deVector = []
    for i in range(len(vector)/length):
        deVector.append([])
        for j in range(length):
            deVector[i].append(vector[j+i*length])
    return deVector


def trans(M):
    # "Transposes" array M
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]


class cell():
    # Python copy of the cell class used in Magritte
    def __init__(self, outputDirectory, tag):
        # Read Magritte's output in outputDirectory
        self.readMagritteOutput(outputDirectory, tag)
        # Devectorize vectorized quantities
        # self.arrangeData()


    def readMagritteOutput(self, outputDirectory, tag):
        if (tag != ''): tag = '_' + tag
        # inputFile = getVariable(outputDirectory+'parameters.hpp', 'INPUTFILE', 'str')
        # Initialize cells by reading Magritte output
        self.temperatureGas     = readScalar(outputDirectory + 'temperature_gas' + tag + '.txt')
        self.temperatureDust    = readScalar(outputDirectory + 'temperature_dust' + tag + '.txt')
        self.thermalRatio       = readScalar(outputDirectory + 'thermal_ratio'   + tag + '.txt')
        self.temperatureGasPrev = readScalar(outputDirectory + 'temperature_gas_prev' + tag + '.txt')
        self.thermalRatioPrev   = readScalar(outputDirectory + 'thermal_ratio_prev'   + tag + '.txt')
        self.abundances         = readVector(outputDirectory + 'abundances' + tag + '.txt')
        self.ncells             = len(self.temperatureGas)
        # self.popVec             = readScalar(outputDirectory + 'level_populations' + tag + '.txt')

    # def arrangeData(self):
        # self.pop = deVectorize(self.popVec, self.ncells)
        # self.pop = trans(self.pop)


        # totNlev = len(self.popVec) / self.ncells
        # self.pop = []
        # for i in range(totNlev):
        #     self.pop.append([])
        #     for p in range(self.ncells):
        #         self.pop[i].append(self.popVec[i+p*totNlev])





def get_ncells(vtkGrid, gridType):
    '''
    Get the number of cells
    '''
    if gridType == 'cell_based':
        return grid.GetNumberOfCells()
    if gridType == 'point_based':
        return grid.GetNumberOfPoints()


def get_vtk_neighbors(vtkGrid, cellNr):
    '''
    Get the numbers of the neighbouring cells of cell cellNr
    '''
    neighbors = []
    # Get the points surrounding the cell with nr cellNr
    surroundingPoints = vtk.vtkIdList()
    vtkGrid.GetCellPoints(cellNr, surroundingPoints)
    # For each of the surrounding points
    for p in range(vtkCellPoints.GetNumberOfIds()):
        # Extract one of the surrounding points (and put in vtkIdList)
        oneSurroundingPoint = vtk.vtkIdList()
        oneSurroundingPoint.SetNumberOfIds(1)
        oneSurroundingPoint.SetId(0,surroundingPoints.GetId(p))
        # Find other cells surrounding that one point
        otherCellsAroundPoint = vtk.vtkIdList()
        grid.GetCellNeighbors(cellNr, oneSurroundingPoint, otherCellsAroundPoint)
        nOtherCells = otherCellsAroundPoint.GetNumberOfIds()
        # Put the Id's of these cells in the neighbors list
        for n in range(nOtherCells):
            neighbors.append(otherCellsAroundPoint.GetId(n))
    # Remove duplicates
    neighbors = list(set(neighbors))
    # Return list with cell numbers of neighbors of cell cellNr
    return neighbors
