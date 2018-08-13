# Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
#
# Developed by: Frederik De Ceuster - University College London & KU Leuven
# _________________________________________________________________________


import setupFunctions
getVariable      = setupFunctions.getVariable
getFilePath      = setupFunctions.getFilePath
fileExtension    = setupFunctions.fileExtension
numberOfLines    = setupFunctions.numberOfLines
writeHeader      = setupFunctions.writeHeader
writeDefinition  = setupFunctions.writeDefinition
readSpeciesNames = setupFunctions.readSpeciesNames
getSpeciesNumber = setupFunctions.getSpeciesNumber
getNcells        = setupFunctions.getNcells
import lineData
LineData  = lineData.LineData
import os
import time
import sys
import numpy as np

from subprocess import call


def getMagritteFolder():
    '''
    Get folder of Magritte source code assuming this piece of code is in it.
    '''
    return os.path.dirname(os.path.realpath(__file__)) + '/../RadiativeTransfer/'


def setupLinedata(inputFolder):
    """
    Set up Line data
    """
    # Get number of chemical species
    specDataFile = inputFolder + 'species.txt'
    nspec        = numberOfLines(specDataFile) + 2
    speciesNames = readSpeciesNames(specDataFile)
    # Get number of data files
    lineDataFolder = inputFolder + 'linedata/'
    lineDataFiles  = [lineDataFolder + lineDataFile for lineDataFile in os.listdir(lineDataFolder)]
    nlspec = len(lineDataFiles)
    # Read line data files
    lineData   = [LineData(fileName) for fileName in lineDataFiles]
    # Get species numbers of line producing species
    name   = [ld.name for ld in lineData]
    number = getSpeciesNumber(speciesNames, name)
    # Get species numbers of collision partners
    partner   = [ld.partner   for ld in lineData]
    partnerNr = getSpeciesNumber(speciesNames, partner)
    # Format as C strings
    name      = ['\"' + ld.name + '\"' for ld in lineData]
    orthoPara = [['\'' + op + '\'' for op in ld.orthoPara] for ld in lineData]
    # Write Magritte_config.hpp file
    fileName = getMagritteFolder() + '../Lines/src/linedata_config.hpp'
    writeHeader(fileName)
    with open(fileName, 'a') as config:
        config.write('#ifndef __LINEDATA_CONFIG_HPP_INCLUDED__\n')
        config.write('#define __LINEDATA_CONFIG_HPP_INCLUDED__\n')
        config.write('\n')
    writeDefinition(fileName, nspec,                             'NSPEC')
    writeDefinition(fileName, nlspec,                            'NLSPEC')
    writeDefinition(fileName, [ld.nlev     for ld in lineData],  'NLEV')
    writeDefinition(fileName, [ld.nrad     for ld in lineData],  'NRAD')
    writeDefinition(fileName, [ld.ncolpar  for ld in lineData],  'NCOLPAR')
    writeDefinition(fileName, [ld.ncoltemp for ld in lineData],  'NCOLTEMP')
    writeDefinition(fileName, [ld.ncoltran for ld in lineData],  'NCOLTRAN')
    writeDefinition(fileName, number,                            'NUMBER')
    writeDefinition(fileName, name,                              'NAME')
    writeDefinition(fileName, partnerNr,                         'PARTNER_NR')
    writeDefinition(fileName, orthoPara,                         'ORTHO_PARA')
    writeDefinition(fileName, [ld.energy    for ld in lineData], 'ENERGY')
    writeDefinition(fileName, [ld.weight    for ld in lineData], 'WEIGHT')
    writeDefinition(fileName, [ld.irad      for ld in lineData], 'IRAD')
    writeDefinition(fileName, [ld.jrad      for ld in lineData], 'JRAD')
    #writeDefinition(fileName, [ld.frequency for ld in lineData], 'FREQUENCY')
    writeDefinition(fileName, [ld.frequency for ld in lineData], 'FREQ')
    writeDefinition(fileName, [ld.A         for ld in lineData], 'A_COEFF')
    writeDefinition(fileName, [ld.B         for ld in lineData], 'B_COEFF')
    writeDefinition(fileName, [ld.icol      for ld in lineData], 'ICOL')
    writeDefinition(fileName, [ld.jcol      for ld in lineData], 'JCOL')
    writeDefinition(fileName, [ld.coltemp   for ld in lineData], 'COLTEMP')
    writeDefinition(fileName, [ld.C_data    for ld in lineData], 'C_DATA')
    with open(fileName, 'a') as config:
        config.write('\n')
        config.write('#endif // __LINEDATA_CONFIG_HPP_INCLUDED__\n')
    # Done


def getDimensions(cellsFile):
    '''
    Get number of dimensions of grid
    '''
    # Read the cells file
    (x,y,z, vx,vy,vz) = np.loadtxt(cellsFile, unpack=True)
    # Determine dimension by spread along axis
    dimension = 0
    for r in [x,y,z]:
        if (np.var(r) > 0.0):
            dimension += 1
    # Done
    return dimension





def setupMagritte(projectFolder, runName=''):
    """
    Main setup for Magritte
    """
    # Create an io folder if is does not exist yet
    ioFolder = projectFolder + 'io/'
    if not os.path.isdir(ioFolder):
        os.mkdir(ioFolder)
    # Get a date stamp to name current folder
    dateStamp = time.strftime("%y-%m-%d_%H:%M:%S", time.gmtime())
    # Create run folder
    if runName == '':
        runFolder = ioFolder + dateStamp + '/'
    else:
        runFolder = ioFolder + runName + '/'
    os.mkdir(runFolder)
    # Create input and output folders
    inputFolder  = runFolder + 'input/'
    os.mkdir(inputFolder)
    outputFolder = runFolder + 'output/'
    os.mkdir(outputFolder)
    # Call script to prepare the input folder
    call(['python ' + projectFolder +'createInput.py ' + inputFolder], shell=True)
    # Extract Magritte (source) folder
    MagritteFolder = getMagritteFolder()
    # Get dimension and number of cells
    cellsFile = inputFolder + 'cells.txt'
    dimension = getDimensions(cellsFile)
    ncells    = numberOfLines(cellsFile)
    # Get number of rays
    rayFile   = inputFolder + 'rays.txt'
    nrays     = numberOfLines(rayFile)
    # Get number of chemical species
    specsFile = inputFolder + 'species.txt'
    nspec     = numberOfLines(specsFile) + 2
    # Set up linedata
    setupLinedata(inputFolder)
    # Write configure.hpp file
    with open(MagritteFolder + 'src/configure.hpp', 'w') as config:
        config.write('#ifndef __CONFIGURE_HPP_INCLUDED__\n')
        config.write('#define __CONFIGURE_HPP_INCLUDED__\n')
        config.write('\n')
        config.write('#include <string>\n')
        config.write('using namespace std;\n')
        config.write('#include "folders.hpp"\n')
        config.write('\n')
        config.write('const int  Dimension = {};\n'.format(dimension))
        config.write('const long     Nrays = {};\n'.format(nrays))
        config.write('const long    Ncells = {};\n'.format(ncells))
        config.write('const int      Nspec = {};\n'.format(nspec))
        config.write('\n')
        config.write('#endif // __CONFIGURE_HPP_INCLUDED__\n')
    # Write folders.hpp file
    with open(MagritteFolder + 'src/folders.hpp', 'w') as folder:
        folder.write('#ifndef __FOLDERS_HPP_INCLUDED__\n')
        folder.write('#define __FOLDERS_HPP_INCLUDED__\n')
        folder.write('\n')
        folder.write('#include <string>\n')
        folder.write('using namespace std;\n')
        folder.write('\n')
        folder.write('const string Magritte_folder = \"{}\";\n'.format(MagritteFolder))
        folder.write('\n')
        folder.write('const string  input_folder = \"{}\";\n'.format(inputFolder))
        folder.write('const string output_folder = \"{}\";\n'.format(outputFolder))
        folder.write('\n')
        folder.write('#endif // __FOLDERS_HPP_INCLUDED__\n')
    # Done


# Main
# ----

if (__name__ == '__main__'):
    # Setup Magritte if necessary
    projectFolder = str(sys.argv[1])
    if len(sys.argv) > 2:
        ioName    = str(sys.argv[2])
    else:
        ioName    = ''
    print('Setting up Magritte...')
    # Run setup
    setupMagritte(projectFolder, ioName)
    # Done
    print('Setup done. Magritte can be compiled now.')
