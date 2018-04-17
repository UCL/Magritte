# Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
#
# Developed by: Frederik De Ceuster - University College London & KU Leuven
# _________________________________________________________________________


import setupFunctions
getProjectFolder = setupFunctions.getProjectFolder
getVariable      = setupFunctions.getVariable
getFilePath      = setupFunctions.getFilePath
fileExtension    = setupFunctions.fileExtension
getNCELLS        = setupFunctions.getNCELLS
numberOfLines    = setupFunctions.numberOfLines
vectorize        = setupFunctions.vectorize
writeHeader      = setupFunctions.writeHeader
writeDefinition  = setupFunctions.writeDefinition
readSpeciesNames = setupFunctions.readSpeciesNames
getSpeciesNumber = setupFunctions.getSpeciesNumber
import lineData
LineData  = lineData.LineData
import makeRates
import filecmp
import os
import shutil
import time
import sys


def setupMagritte():
    """Main setup for Magritte"""
    # Get number of spacial dimensions
    dimensions = getVariable('DIMENSIONS', 'int')
    # Get number of rays
    if (dimensions == 3):
        # In 3D the number of rays is determined by nsides (HEALPix)
        nsides = getVariable('NSIDES', 'long')
        nrays  = 12*nsides**2
    else:
        # In 1D and 2D the number of rays is user defined
        nrays  = getVariable('NRAYS', 'long')
    # Read grid data
    inputFile = getFilePath('INPUTFILE')
    gridType  = getVariable('GRID_TYPE', 'str')
    fixedGrid = getVariable('FIXED_NCELLS', 'bool')
    # Get number of (Magritte) grid cells
    ncells     = getNCELLS(inputFile, gridType)
    ncellsInit = ncells
    if not fixedGrid:
        ncells = 'cells->ncells'
    # Get number of chemical species
    specDataFile = getFilePath('SPEC_DATAFILE')
    nspec        = numberOfLines(specDataFile) + 2
    speciesNames = readSpeciesNames(specDataFile)
    # Get number of chemical reactions
    reacDataFile = getFilePath('REAC_DATAFILE')
    nreac        = numberOfLines(reacDataFile)
    # Try to make species and rates
    try:
        makeRates.makeRates(specDataFile, reacDataFile)
    except:
        writeHeader('../src/sundials/rate_equations.cpp')
        writeHeader('../src/sundials/jacobian.cpp')
        print('\n\nWARNING: makeRates failed!\n\n')
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
    partner   = vectorize([ld.partner   for ld in lineData])
    partnerNr = getSpeciesNumber(speciesNames, partner)
    # Setup data structures
    tot_nlev                 = 0
    tot_nrad                 = 0
    tot_nlev2                = 0
    tot_ncolpar              = 0
    cum_nlev                 = [0]
    cum_nlev2                = [0]
    cum_nrad                 = [0]
    cum_ncolpar              = [0]
    cum_tot_ncoltemp         = [0]
    cum_tot_ncoltran         = [0]
    cum_tot_ncoltrantemp     = [0]
    for ld in lineData:
        # Add numbers
        tot_nlev         += ld.nlev
        tot_nrad         += ld.nrad
        tot_nlev2        += ld.nlev*ld.nlev
        tot_ncolpar      += ld.ncolpar
    # Vectorize variables
    nlev             = vectorize([ld.nlev             for ld in lineData])
    nrad             = vectorize([ld.nrad             for ld in lineData])
    ncolpar          = vectorize([ld.ncolpar          for ld in lineData])
    ncoltemp         = vectorize([ld.ncoltemp         for ld in lineData])
    ncoltran         = vectorize([ld.ncoltran         for ld in lineData])
    cum_ncoltemp     = vectorize([ld.cum_ncoltemp     for ld in lineData])
    cum_ncoltran     = vectorize([ld.cum_ncoltran     for ld in lineData])
    cum_ncoltrantemp = vectorize([ld.cum_ncoltrantemp for ld in lineData])
    tot_ncoltemp     = vectorize([ld.tot_ncoltemp     for ld in lineData])
    tot_ncoltran     = vectorize([ld.tot_ncoltran     for ld in lineData])
    tot_ncoltrantemp = vectorize([ld.tot_ncoltrantemp for ld in lineData])
    # Calculate cumulatives
    for i in range(nlspec-1):
        cum_nlev.append(cum_nlev[i]+lineData[i].nlev)
        cum_nlev2.append(cum_nlev2[i]+lineData[i].nlev*lineData[i].nlev)
        cum_nrad.append(cum_nrad[i]+lineData[i].nrad)
        cum_ncolpar.append(cum_ncolpar[i]+lineData[i].ncolpar)
        cum_tot_ncoltemp.append(cum_tot_ncoltemp[i]+tot_ncoltemp[i])
        cum_tot_ncoltran.append(cum_tot_ncoltran[i]+tot_ncoltran[i])
        cum_tot_ncoltrantemp.append(cum_tot_ncoltrantemp[i]+tot_ncoltrantemp[i])
    tot_cum_tot_ncoltran     = cum_tot_ncoltran[-1]+tot_ncoltran[-1]
    tot_cum_tot_ncoltemp     = cum_tot_ncoltemp[-1]+tot_ncoltemp[-1]
    tot_cum_tot_ncoltrantemp = cum_tot_ncoltrantemp[-1]+tot_ncoltrantemp[-1]
    # Write Magritte_config.hpp file
    fileName = '../src/Magritte_config.hpp'
    writeHeader(fileName)
    writeDefinition(fileName, ncells,                   'NCELLS')
    if (dimensions == 3):
        writeDefinition(fileName, nrays,                'NRAYS')
    writeDefinition(fileName, ncellsInit,               'NCELLS_INIT')
    writeDefinition(fileName, nspec,                    'NSPEC')
    writeDefinition(fileName, nreac,                    'NREAC')
    writeDefinition(fileName, nlspec,                   'NLSPEC')
    writeDefinition(fileName, tot_nlev,                 'TOT_NLEV')
    writeDefinition(fileName, tot_nrad,                 'TOT_NRAD')
    writeDefinition(fileName, tot_nlev2,                'TOT_NLEV2')
    writeDefinition(fileName, tot_ncolpar,              'TOT_NCOLPAR')
    writeDefinition(fileName, tot_cum_tot_ncoltran,     'TOT_CUM_TOT_NCOLTRAN')
    writeDefinition(fileName, tot_cum_tot_ncoltemp,     'TOT_CUM_TOT_NCOLTEMP')
    writeDefinition(fileName, tot_cum_tot_ncoltrantemp, 'TOT_CUM_TOT_NCOLTRANTEMP')
    writeDefinition(fileName, nlev,                     'NLEV')
    writeDefinition(fileName, nrad,                     'NRAD')
    writeDefinition(fileName, cum_nlev,                 'CUM_NLEV')
    writeDefinition(fileName, cum_nlev2,                'CUM_NLEV2')
    writeDefinition(fileName, cum_nrad,                 'CUM_NRAD')
    writeDefinition(fileName, ncolpar,                  'NCOLPAR')
    writeDefinition(fileName, cum_ncolpar,              'CUM_NCOLPAR')
    writeDefinition(fileName, ncoltemp,                 'NCOLTEMP')
    writeDefinition(fileName, ncoltran,                 'NCOLTRAN')
    writeDefinition(fileName, cum_ncoltemp,             'CUM_NCOLTEMP')
    writeDefinition(fileName, cum_ncoltran,             'CUM_NCOLTRAN')
    writeDefinition(fileName, cum_ncoltrantemp,         'CUM_NCOLTRANTEMP')
    writeDefinition(fileName, tot_ncoltemp,             'TOT_NCOLTEMP')
    writeDefinition(fileName, tot_ncoltran,             'TOT_NCOLTRAN')
    writeDefinition(fileName, tot_ncoltrantemp,         'TOT_NCOLTRANTEMP')
    writeDefinition(fileName, cum_tot_ncoltemp,         'CUM_TOT_NCOLTEMP')
    writeDefinition(fileName, cum_tot_ncoltran,         'CUM_TOT_NCOLTRAN')
    writeDefinition(fileName, cum_tot_ncoltrantemp,     'CUM_TOT_NCOLTRANTEMP')
    # Setup definitions
    name      = ['\"' + ld.name + '\"' for ld in lineData]   # format as C strings
    orthoPara = vectorize([ld.orthoPara for ld in lineData])
    orthoPara = ['\'' + op + '\'' for op in orthoPara]       # format as C strings
    # Vectorize all variables
    energy    = vectorize([ld.energy    for ld in lineData])
    weight    = vectorize([ld.weight    for ld in lineData])
    irad      = vectorize([ld.irad      for ld in lineData])
    jrad      = vectorize([ld.jrad      for ld in lineData])
    frequency = vectorize([ld.frequency for ld in lineData])
    A         = vectorize([ld.A         for ld in lineData])
    B         = vectorize([ld.B         for ld in lineData])
    icol      = vectorize([ld.icol      for ld in lineData])
    jcol      = vectorize([ld.jcol      for ld in lineData])
    coltemp   = vectorize([ld.coltemp   for ld in lineData])
    C_data    = vectorize([ld.C_data    for ld in lineData])
    # Write definitions.hpp
    fileName = '../src/line_data.hpp'
    writeHeader(fileName)
    writeDefinition(fileName, number,    'NUMBER')
    writeDefinition(fileName, name,      'NAME')
    writeDefinition(fileName, partnerNr, 'PARTNER_NR')
    writeDefinition(fileName, orthoPara, 'ORTHO_PARA')
    writeDefinition(fileName, energy,    'ENERGY')
    writeDefinition(fileName, weight,    'WEIGHT')
    writeDefinition(fileName, irad,      'IRAD')
    writeDefinition(fileName, jrad,      'JRAD')
    writeDefinition(fileName, frequency, 'FREQUENCY')
    writeDefinition(fileName, A,         'A_COEFF')
    writeDefinition(fileName, B,         'B_COEFF')
    writeDefinition(fileName, icol,      'ICOL')
    writeDefinition(fileName, jcol,      'JCOL')
    writeDefinition(fileName, coltemp,   'COLTEMP')
    writeDefinition(fileName, C_data,    'C_DATA')


# Main
# ----

if (__name__ == '__main__'):

    # Setup Magritte if necessary
    projectFolder = str(sys.argv[1])
    print('Setting up Magritte...')
    # If parameter file is not up to date, run setup
    if not filecmp.cmp(projectFolder+'parameters.hpp','../src/parameters.hpp'):
        print('parameters.hpp was out of date, updating...')
        # Copy parameter file from project to Magritte folder
        shutil.copyfile(projectFolder+'parameters.hpp','../src/parameters.hpp')
        # Get date date stamp for output directory
        dateStamp       = time.strftime("%y-%m-%d_%H:%M:%S", time.gmtime())
        outputDirectory = projectFolder + 'output/files/' + dateStamp + '/'
        # Write directories to cpp header
        fileName = '../src/directories.hpp'
        writeHeader(fileName)
        writeDefinition(fileName, '\"'+outputDirectory+'\"', 'OUTPUT_DIRECTORY')
        writeDefinition(fileName, '\"'+projectFolder+'\"', 'PROJECT_FOLDER')
        # Run setup
        setupMagritte()
        # Create output directory
        os.mkdir(outputDirectory)
        os.mkdir(outputDirectory + 'plots/')
        shutil.copyfile(projectFolder+'parameters.hpp',outputDirectory+'parameters.hpp')
    print('parameter.hpp is up to date.')
    print('Setup done. Magritte can be compiled now.')
