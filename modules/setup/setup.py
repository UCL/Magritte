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
import quadrature
write_quadrature_file = quadrature.write_quadrature_file
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
    name    = [ld.name for ld in lineData]
    numbers = getSpeciesNumber(speciesNames, name)
    # Get species numbers of collision partners
    partner   = [ld.partner   for ld in lineData]
    partnerNr = getSpeciesNumber(speciesNames, partner)
    # Format as C strings
    #name      = ['\"' + ld.name + '\"' for ld in lineData]
    orthoPara = [['\'' + op + '\'' for op in ld.orthoPara] for ld in lineData]
    # Write linedata files
    with open(lineDataFolder + 'nlspec.txt'    , 'w') as dataFile:
        dataFile.write(str(nlspec))
    with open(lineDataFolder + 'nlev.txt'      , 'w') as dataFile:
        for ld in lineData:
            dataFile.write(str(ld.nlev)     + '\t')
    with open(lineDataFolder + 'nrad.txt'      , 'w') as dataFile:
        for ld in lineData:
            dataFile.write(str(ld.nrad)     + '\t')
    with open(lineDataFolder + 'ncolpar.txt'   , 'w') as dataFile:
        for ld in lineData:
            dataFile.write(str(ld.ncolpar)  + '\t')
    with open(lineDataFolder + 'ntmp.txt'      , 'w') as dataFile:
        for ld in lineData:
            for ntmps in ld.ncoltemp:
                dataFile.write(str(ntmps)   + '\t')
            dataFile.write('\n')
    with open(lineDataFolder + 'ncol.txt'      , 'w') as dataFile:
        for ld in lineData:
            for ncols in ld.ncoltran:
                dataFile.write(str(ncols)   + '\t')
            dataFile.write('\n')
    with open(lineDataFolder + 'num.txt'       , 'w') as dataFile:
        for number in numbers:
            dataFile.write(str(number)      + '\t')
    with open(lineDataFolder + 'sym.txt'       , 'w') as dataFile:
        for ld in lineData:
            dataFile.write(ld.name          + '\t')
    with open(lineDataFolder + 'irad.txt'      , 'w') as dataFile:
        for ld in lineData:
            for i in ld.irad:
                dataFile.write(str(i)       + '\t')
            dataFile.write('\n')
    with open(lineDataFolder + 'jrad.txt'      , 'w') as dataFile:
        for ld in lineData:
            for j in ld.jrad:
                dataFile.write(str(j)       + '\t')
            dataFile.write('\n')
    with open(lineDataFolder + 'energy.txt'    , 'w') as dataFile:
        for ld in lineData:
            for E in ld.energy:
                dataFile.write(str(E)       + '\t')
            dataFile.write('\n')
    with open(lineDataFolder + 'weight.txt'    , 'w') as dataFile:
        for ld in lineData:
            for W in ld.weight:
                dataFile.write(str(W)       + '\t')
            dataFile.write('\n')
    with open(lineDataFolder + 'frequency.txt' , 'w') as dataFile:
        for ld in lineData:
            for F in ld.frequency:
                dataFile.write(str(F)       + '\t')
            dataFile.write('\n')
    for l in range(nlspec):
        with open(lineDataFolder + f'A_{l}.txt', 'w') as dataFile:
            for i in range(lineData[l].nlev):
                for j in range(lineData[l].nlev):
                    dataFile.write(str(lineData[l].A[i][j]) + '\t')
                dataFile.write('\n')
    for l in range(nlspec):
        with open(lineDataFolder + f'B_{l}.txt', 'w') as dataFile:
            for i in range(lineData[l].nlev):
                for j in range(lineData[l].nlev):
                    dataFile.write(str(lineData[l].B[i][j]) + '\t')
                dataFile.write('\n')
    with open(lineDataFolder + 'num_col_partner.txt', 'w') as dataFile:
        for l in range(nlspec):
            for num in partnerNr[l]:
                dataFile.write(str(num) + '\t')
            dataFile.write('\n')
    with open(lineDataFolder + 'orth_or_para_H2.txt', 'w') as dataFile:
        for ld in lineData:
            for op in ld.orthoPara:
                dataFile.write(str(op) + '\t')
            dataFile.write('\n')
    for l in range(nlspec):
        with open(lineDataFolder + f'temperature_col_{l}.txt', 'w') as dataFile:
            for c in range(lineData[l].ncolpar):
                for temp in lineData[l].coltemp[c]:
                    dataFile.write(str(temp) + '\t')
                dataFile.write('\n')
    for l in range(nlspec):
        for c in range(lineData[l].ncolpar):
            for t in range(lineData[l].ncoltemp[c]):
                with open(lineDataFolder + f'C_data_{l}_{c}_{t}.txt', 'w') as dataFile:
                    for i in range(lineData[l].nlev):
                        for j in range(lineData[l].nlev):
                            dataFile.write(str(lineData[l].C_data[c][t][i][j]) + '\t')
                        dataFile.write('\n')
    # Write Magritte_config.hpp file
    #fileName = getMagritteFolder() + '../Lines/src/linedata_config.hpp'
    #writeHeader(fileName)
    #with open(fileName, 'a') as config:
    #    config.write('#ifndef __LINEDATA_CONFIG_HPP_INCLUDED__\n')
    #    config.write('#define __LINEDATA_CONFIG_HPP_INCLUDED__\n')
    #    config.write('\n')
    #writeDefinition(fileName, nlspec,                            'NLSPEC')
    #writeDefinition(fileName, [ld.nlev     for ld in lineData],  'NLEV')
    #writeDefinition(fileName, [ld.nrad     for ld in lineData],  'NRAD')
    #writeDefinition(fileName, [ld.ncolpar  for ld in lineData],  'NCOLPAR')
    #writeDefinition(fileName, [ld.ncoltemp for ld in lineData],  'NCOLTEMP')
    #writeDefinition(fileName, [ld.ncoltran for ld in lineData],  'NCOLTRAN')
    #writeDefinition(fileName, number,                            'NUMBER')
    #writeDefinition(fileName, name,                              'NAME')
    #writeDefinition(fileName, partnerNr,                         'PARTNER_NR')
    #writeDefinition(fileName, orthoPara,                         'ORTHO_PARA')
    #writeDefinition(fileName, [ld.energy    for ld in lineData], 'ENERGY')
    #writeDefinition(fileName, [ld.weight    for ld in lineData], 'WEIGHT')
    #writeDefinition(fileName, [ld.irad      for ld in lineData], 'IRAD')
    #writeDefinition(fileName, [ld.jrad      for ld in lineData], 'JRAD')
    #writeDefinition(fileName, [ld.frequency for ld in lineData], 'FREQ')
    #writeDefinition(fileName, [ld.A         for ld in lineData], 'A_COEFF')
    #writeDefinition(fileName, [ld.B         for ld in lineData], 'B_COEFF')
    #writeDefinition(fileName, [ld.coltemp   for ld in lineData], 'COLTEMP')
    #writeDefinition(fileName, [ld.C_data    for ld in lineData], 'C_DATA')
    #with open(fileName, 'a') as config:
    #    config.write('\n')
    #    config.write('#endif // __LINEDATA_CONFIG_HPP_INCLUDED__\n')
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
    dateStamp = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime())
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
        config.write('const int  DIMENSION = {};\n'.format(dimension))
        config.write('const long     NRAYS = {};\n'.format(nrays))
        config.write('const long    NCELLS = {};\n'.format(ncells))
        config.write('const int      NSPEC = {};\n'.format(nspec))
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
    # Write quadrature file
    n_quadrature_points = 39
    print(n_quadrature_points)
    write_quadrature_file(MagritteFolder + 'src/quadrature.hpp', n_quadrature_points)
    # Done
    return runFolder


# Main
# ----

if (__name__ == '__main__'):
    # Setup Magritte if necessary
    projectFolder = str(sys.argv[1])
    if (len(sys.argv) > 2):
        ioName    = str(sys.argv[2])
    else:
        ioName    = ''
    print('Setting up Magritte...')
    # Run setup
    runFolder = setupMagritte(projectFolder, ioName)
    # Done
    print('Setup done for :')
    print(runFoder)
    print('Magritte can be compiled now.')
