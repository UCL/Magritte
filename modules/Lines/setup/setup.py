# Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
#
# Developed by: Frederik De Ceuster - University College London & KU Leuven
# _________________________________________________________________________


import setupFunctions
getProjectFolder = setupFunctions.getProjectFolder
getVariable      = setupFunctions.getVariable
getFilePath      = setupFunctions.getFilePath
fileExtension    = setupFunctions.fileExtension
numberOfLines    = setupFunctions.numberOfLines
writeHeader      = setupFunctions.writeHeader
writeDefinition  = setupFunctions.writeDefinition
readSpeciesNames = setupFunctions.readSpeciesNames
getSpeciesNumber = setupFunctions.getSpeciesNumber
import lineData
LineData  = lineData.LineData
import filecmp
import os
import shutil
import time
import sys


def setupMagritte():
    """Main setup for Magritte"""
    # Get number of chemical species
    specDataFile = getFilePath('SPEC_DATAFILE')
    nspec        = numberOfLines(specDataFile) + 2
    speciesNames = readSpeciesNames(specDataFile)
    # Get number of data files
    lineDataFiles = getFilePath('LINE_DATAFILES')
    if not isinstance(lineDataFiles, list):
        lineDataFiles = [lineDataFiles]
    nlspec = len(lineDataFiles)
    # Read line data files
    dataFormat = getVariable('DATA_FORMAT', 'str')
    lineData   = [LineData(fileName, dataFormat) for fileName in lineDataFiles]
    # Get species numbers of line producing species
    name   = [ld.name for ld in lineData]
    number = getSpeciesNumber(speciesNames, name)
    # Get species numbers of collision partners
    partner   = [ld.partner   for ld in lineData]
    partnerNr = getSpeciesNumber(speciesNames, partner)
    # Setup definitions
    name      = ['\"' + ld.name + '\"' for ld in lineData]                       # format as C strings
    orthoPara = [['\'' + op + '\'' for op in ld.orthoPara] for ld in lineData]   # format as C strings
    # Write Magritte_config.hpp file
    fileName = '../src/linedata_config.hpp'
    writeHeader(fileName)
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
    writeDefinition(fileName, [ld.frequency for ld in lineData], 'FREQUENCY')
    writeDefinition(fileName, [ld.A         for ld in lineData], 'A_COEFF')
    writeDefinition(fileName, [ld.B         for ld in lineData], 'B_COEFF')
    writeDefinition(fileName, [ld.icol      for ld in lineData], 'ICOL')
    writeDefinition(fileName, [ld.jcol      for ld in lineData], 'JCOL')
    writeDefinition(fileName, [ld.coltemp   for ld in lineData], 'COLTEMP')
    writeDefinition(fileName, [ld.C_data    for ld in lineData], 'C_DATA')


# Main
# ----

if (__name__ == '__main__'):

    # Setup Magritte if necessary
    projectFolder = str(sys.argv[1])
    print('Setting up Magritte...')
    # If parameter file is not up to date, run setup
#    if not filecmp.cmp(projectFolder+'parameters.hpp','../src/parameters.hpp'):
#        print('parameters.hpp was out of date, updating...')
        # Copy parameter file from project to Magritte folder
       # shutil.copyfile(projectFolder+'parameters.hpp','../src/parameters.hpp')
        # Get date date stamp for output directory
       # dateStamp       = time.strftime("%y-%m-%d_%H:%M:%S", time.gmtime())
       # outputDirectory = projectFolder + 'output/files/' + dateStamp + '/'
        # Write directories to cpp header
    fileName = '../src/directories.hpp'
    writeHeader(fileName)
#    writeDefinition(fileName, '\"'+outputDirectory+'\"', 'OUTPUT_DIRECTORY')
    writeDefinition(fileName, '\"'+projectFolder+'\"', 'PROJECT_FOLDER')
        # Run setup
    setupMagritte()
        # Create output directory
       # os.mkdir(outputDirectory)
       # os.mkdir(outputDirectory + 'plots/')
       # shutil.copyfile(projectFolder+'parameters.hpp',outputDirectory+'parameters.hpp')
    print('parameter.hpp is up to date.')
    print('Setup done. Levels can be compiled now.')
