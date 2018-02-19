#! /usr/bin/python

# Frederik De Ceuster - University College London & KU Leuven                                     #
#                                                                                                 #
# ----------------------------------------------------------------------------------------------- #
#                                                                                                 #
# rate_equations: defines the (chemical) rate equations                                           #
#                                                                                                 #
#  ( based on odes in 3D-PDR                                                                      #
#    and the cvRobers_dns example that comes with Sundials )                                      #
#                                                                                                 #
# ----------------------------------------------------------------------------------------------- #
#                                                                                                 #
# ----------------------------------------------------------------------------------------------- #



import math
import os
import string
import struct
import sys
import time

import numpy as np



useTtk = False
useTkinter = False
useTtk = False

# Default options
reactionFile = ''
speciesFile  = ''
outputPrefix = ''
# THIS HAS TO BE FALSE !!!!
sortSpecies = False
logForm = False
fileFormat = 'Rate05'
codeFormat = 'C'





###   Definitions of the helper functions   ###
###   -----------------------------------   ###



# Determine the appropriate file format and read the entries in the specified reaction file
def read_reaction_file(fileName):
    input = open(fileName, mode='r')
    input.seek(0)
    data = input.readline().strip('\r\n')

    # Ignore header entries
    while len(data) < 5:
        data = input.readline().strip('\r\n')

    # Determine the format of the reaction file
    data = data.split(',')
    if len(data) > 10: fileFormat = 'rate05'
    else:
        data = data[0]
        try:
            formatCode = '4sx8sx8sx8sx8sx8sx4sx4sx8sxxx5sxx8s1s5s5s1s4s'
            test = struct.unpack(formatCode, data)
            fileFormat = 'rate99'
        except:
            try:
                formatCode = 'x4sx7sx7sx7sx7sx3sx3s8sx5sx8s1s4s4s'
                test = struct.unpack(formatCode, data)
                fileFormat = 'rate95'
            except:
                sys.exit('\nERROR! Unrecognised file format in input reaction file\n')
                return

    reactants = [] ; products = [] ; alpha = [] ; beta = [] ; gamma = [] ; labels = []
    while len(data) != 0 and data != ['']:
        if fileFormat == 'rate05':
            reactants.append(data[1:4])
            products.append(data[4:8])
            alpha.append(float(data[8]))
            beta.append(float(data[9]))
            gamma.append(float(data[10]))
            labels.append([value.strip('\r\n') for value in data[11:]])
            if len(labels[-1]) < 4: labels[-1] = ['','10','41000','','']
            data = input.readline().split(',')
        elif fileFormat == 'rate99':
            data = data[:102]
            data = struct.unpack(formatCode, data)
            reactants.append([convert_species(value.strip()) for value in data[1:4]])
            products.append([convert_species(value.strip())  for value in data[4:8]])
            alpha.append(float(data[8]))
            beta.append(float(data[9]))
            gamma.append(float(data[10]))
            labels.append([value.strip() for value in data[11:]])
            if len(labels[-1]) < 4: labels[-1] = ['','10','41000','','']
            data = input.readline().strip('\r\n')
        elif fileFormat == 'rate95':
            data = data[:77]
            data = struct.unpack(formatCode, data)
            reactants.append([convert_species(value.strip()) for value in data[1:3]])
            products.append([convert_species(value.strip())  for value in data[3:7]])
            alpha.append(float(data[7]))
            beta.append(float(data[8]))
            gamma.append(float(data[9]))
            labels.append([data[10].strip(),'10','41000'])
            labels[-1].append(['A','B','C','D','E',''][['1','2','3','4','5','9'].index(data[11][3])])
            labels[-1].append(data[12].strip())
            labels[-1].append(data[11].strip())
            reactants[-1].append('')
            data = input.readline().strip('\r\n')
    nReactions = len(reactants)
    input.close()
    return nReactions, reactants, products, alpha, beta, gamma, labels



# Determine the appropriate file format and read the entries in the specified species file
def read_species_file(fileName):
    input = open(fileName, mode='r')
    input.seek(0)
    data = input.readline().strip('\r\n')

    # Ignore header entries
    while len(data) < 5:
        data = input.readline().strip('\r\n')

    # Determine the format of the species file
    data = data.split(',')
    if len(data) > 3: fileFormat = 'rate05'
    else:
        data = data[0]
        try:
            formatCode = 'xx3sx8sxxx8sxx5s'
            test = struct.unpack(formatCode, data)
            fileFormat = 'rate99'
        except:
            try:
                formatCode = 'xxx3sxx8sxxxx3sxx8sxxxx3sxx8sxxxx3sxx8sxxxx3sxx8sx'
                test = struct.unpack(formatCode, data)
                fileFormat = 'rate95'
            except:
                if useTkinter:
                    app.statusMessage('  ERROR! Unrecognised file format in input species file.', replace=True, error=True)
                else:
                    sys.exit('\n  ERROR! Unrecognised file format in input species file\n')
                return

    species = [] ; abundance = [] ; mass = []
    while len(data) != 0 and data != ['']:
        if fileFormat == 'rate05':
            if data[1].lower() != 'e-' and data[1].upper() != 'ELECTR':
                species.append(convert_species(data[1]))
                abundance.append(float(data[2]))
                mass.append(float(data[3]))
            data = input.readline().split(',')
        elif fileFormat == 'rate99':
            data = struct.unpack(formatCode, data)
            if data[1].strip().lower() != 'e-' and data[1].strip().upper() != 'ELECTR':
                species.append(convert_species(data[1].strip()))
                abundance.append(float(data[2]))
                mass.append(float(data[3]))
            data = input.readline().strip('\r\n')
        elif fileFormat == 'rate95':
            while len(data) > 9:
                data = data[8:]
                if data[:8].strip().lower() != 'e-' and data[:8].strip().upper() != 'ELECTR':
                    species.append(convert_species(data[:8].strip()))
                    abundance.append(float(0))
                    mass.append(float(0))
            data = input.readline().strip('\r\n')
    nSpecies = len(species)
    input.close()
    return nSpecies, species, abundance, mass



# Find all the species involved in a list of reactions
def find_all_species(reactantList, productList):
    ignoreList = ['','#','e-','ELECTR','PHOTON','CRP','CRPHOT','XRAY','XRSEC','XRLYA','XRPHOT','FREEZE','CRH','PHOTD','THERM']
    speciesList = []
    for reactionReactants in reactantList:
        for reactant in reactionReactants:
            if ignoreList.count(reactant) == 0:
                if speciesList.count(reactant) == 0: speciesList.append(reactant)
    for reactionProducts in productList:
        for product in reactionProducts:
            if ignoreList.count(product) == 0:
                if speciesList.count(product) == 0: speciesList.append(product)
    nSpecies = len(speciesList)
    return nSpecies, speciesList



# Remove entries from a list of reactions when none of the listed species are involved
def find_all_reactions(speciesList, reactants, products, alpha, beta, gamma, labels):
    ignoreList = ['','#','e-','ELECTR','PHOTON','CRP','CRPHOT','XRAY','XRSEC','XRLYA','XRPHOT','FREEZE','CRH','PHOTD','THERM']
    n = 0
    while n < len(reactants):
        for reactant in reactants[n]:
            if ignoreList.count(reactant) == 0:
                if speciesList.count(reactant) == 0:
                    reactants.pop(n) ; products.pop(n) ; alpha.pop(n) ; beta.pop(n) ; gamma.pop(n) ; labels.pop(n)
                    n -= 1
                    break
        n += 1
    n = 0
    while n < len(reactants):
        for product in products[n]:
            if ignoreList.count(product) == 0:
                if speciesList.count(product) == 0:
                    reactants.pop(n) ; products.pop(n) ; alpha.pop(n) ; beta.pop(n) ; gamma.pop(n) ; labels.pop(n)
                    n -= 1
                    break
        n += 1
    nReactions = len(reactants)
    return nReactions, reactants, products, alpha, beta, gamma, labels



# Return a list of "orphan" species that are either never formed or never detroyed
def check_orphan_species(speciesList, reactants, products):
    nSpecies = len(speciesList)

    # Count the total number of formation and destruction routes for each species
    nFormation = [] ; nDestruction = []
    if useTtk:
        app.progressValue.set(0)
        Style().configure('Horizontal.TProgressbar', background='#dcdad5')
    for n, species in enumerate(speciesList):
        if useTtk:
            app.progressValue.set(80.0*float(n+1)/float(nSpecies))
            app.status.update_idletasks()
        nFormation.append(sum([reactionProducts.count(species) for reactionProducts in products]))
        nDestruction.append(sum([reactionReactants.count(species) for reactionReactants in reactants]))

    # Check for species that are never formed or destroyed and produce an error message
    orphanList = [] ; missingList = []
    for n, species in enumerate(speciesList):
        if useTtk:
            app.progressValue.set(20.0*float(n+1)/float(nSpecies)+80.0)
            app.status.update_idletasks()
        if nFormation[n] == 0 and nDestruction[n] == 0:
             missingList.append(species)
        elif nFormation[n] == 0:
            orphanList.append(species)
            if useTkinter:
                app.statusMessage('\n\n  ERROR! Species "'+species+'" has destruction reaction(s) but no formation route.', error=True)
            else:
                sys.exit('\n  ERROR! Species "'+species+'" has destruction reaction(s) but no formation route\n')
        elif nDestruction[n] == 0:
            orphanList.append(species)
            if useTkinter:
                app.statusMessage('\n\n  ERROR! Species "'+species+'" has formation reaction(s) but no destruction route.', error=True)
            else:
                sys.exit('\n  ERROR! Species "'+species+'" has formation reaction(s) but no destruction route\n')
    if len(orphanList) > 0:
        return
    if len(missingList) > 0:
        if useTkinter:
            app.statusMessage('\n\n  WARNING! The following species are missing from the reaction network:\n\n'+string.join(missingList,', ')+'.', error=True)
        else:
            print '\n  WARNING! The following species are missing from the reaction network:\n'+string.join(missingList,', ')
    return nFormation, nDestruction, missingList



# Convert a species name to the appropriate mixed-case form
def convert_species(species):
    upperCaseList = ['ELECTR','H','D','HE','LI','C','N','O','F','NA','MG','SI','P','S','CL','CA','FE']
    mixedCaseList = ['e-','H','D','He','Li','C','N','O','F','Na','Mg','Si','P','S','Cl','Ca','Fe']
    for n in range(len(upperCaseList)):
        species = species.replace(upperCaseList[n], mixedCaseList[n])
    return species



# Sort a list of species first by their total number of destruction
# reactions and then by their total number of formation reactions
def sort_species(speciesList, nFormation, nDestruction):
    zippedList = zip(nDestruction, nFormation, speciesList)
    zippedList.sort()
    nDestruction, nFormation, speciesList = zip(*zippedList)
    zippedList = zip(nFormation, nDestruction, speciesList)
    zippedList.sort()
    nFormation, nDestruction, speciesList = zip(*zippedList)
    return speciesList



# Find the elemental constituents and molecular mass of each species in the supplied list
def find_constituents(speciesList):
    elementList = ['PAH','He','Li','Na','Mg','Si','Cl','Ca','Fe','H','D','C','N','O','F','P','S','#','+','-']
    elementMass = [420.0,4.0,7.0,23.0,24.0,28.0,35.0,40.0,56.0,1.0,2.0,12.0,14.0,16.0,19.0,31.0,32.0,0,0,0]
    nElements = len(elementList)
    speciesConstituents = []
    speciesMass = []

    for species in speciesList:
        constituents = []
        for element in elementList:
            constituents.append(0)
            for n in range(species.count(element)):
                index = species.index(element)+len(element)
                if species[index:index+2].isdigit():
                    constituents[-1] += int(species[index:index+2])
                    species = species[:index-len(element)]+species[index+2:]
                elif species[index:index+1].isdigit():
                    constituents[-1] += int(species[index:index+1])
                    species = species[:index-len(element)]+species[index+1:]
                else:
                    constituents[-1] += 1
                    species = species[:index-len(element)]+species[index:]

        # Calculate the total molecular mass as the sum of the elemental masses of each constituent
        speciesMass.append(sum([float(constituents[i])*float(elementMass[i]) for i in range(nElements)]))

        # Sort the elements in the consituent list by their atomic mass
        zippedList = zip(elementMass, elementList, constituents)
        zippedList.sort()
        sortedMasses, sortedElements, constituents = zip(*zippedList)
        speciesConstituents.append(constituents)

    # Sort the list of elements by their atomic mass
    zippedList = zip(elementMass, elementList)
    zippedList.sort()
    sortedMasses, sortedElements = zip(*zippedList)
    return speciesMass,speciesConstituents,sortedElements



# Create the conservation term for the desired species (an element, electron or dust grain)
def conserve_species(species, speciesConstituents, codeFormat='C'):
    elementList = ['#','+','-','H','D','He','Li','C','N','O','F','Na','Mg','Si','P','S','Cl','Ca','Fe','PAH']
    nSpecies = len(speciesConstituents)
    conservationEquation = ''
    # Handle the special case of electrons (i.e., charge conservation with both anions and cations)
    if species == 'e-':
        indexPos = elementList.index('+')
        indexNeg = elementList.index('-')
        for n in range(nSpecies):
            if speciesConstituents[n][indexPos] > 0:
                if len(conservationEquation) > 0: conservationEquation += '+'
                if codeFormat == 'C':   conservationEquation += multiple(speciesConstituents[n][indexPos])+'Ith(y,'+str(n)+')'

            if speciesConstituents[n][indexNeg] > 0:
                conservationEquation += '-'
                if codeFormat == 'C':   conservationEquation += multiple(speciesConstituents[n][indexNeg])+'Ith(y,'+str(n)+')'
    else:
        index = elementList.index(species)
        for n in range(nSpecies):
            if speciesConstituents[n][index] > 0:
                if len(conservationEquation) > 0: conservationEquation += '+'
                if codeFormat == 'C':   conservationEquation += multiple(speciesConstituents[n][index])+'Ith(y,'+str(n)+')'
    if len(conservationEquation) > 0:
        if codeFormat == 'C':   conservationEquation = '  x_e = '+conservationEquation+';\n'

    else:
        if codeFormat == 'C':   conservationEquation = '  x_e = 0;\n'
    return conservationEquation



# Create the equations for the additional parameters needed in certain X-ray reaction rates
def xray_parameters(speciesList, codeFormat='C'):
    # Find the index numbers for H, H2, He and e- in the species list
    indexH  = speciesList.index('H')
    indexH2 = speciesList.index('H2')
    indexHe = speciesList.index('He')
    indexEl = len(speciesList)

    # Create the code string to calculate the parameters zeta_H, zeta_H2 and zeta_He,
    # i.e., 1/(W_i.x_i) in equation D.12 of Meijerink & Spaans (2005, A&A, 436, 397)
    if codeFormat == 'C':   xrayParameterEquations = '\n  /* The X-ray secondary ionization rates depend on the mean energies\n   * required to ionize H or H2 in a neutral gas mixture, 1/(W_i*x_i) */\n  zeta_H  = 1.0/(39.8*(1.0+12.2*pow(x_e,0.866))*(Ith(y,'+str(indexH)+')+1.89*Ith(y,'+str(indexH2)+')));\n  zeta_H2 = 1.0/(41.9*(1.0+6.72*pow((1.83*x_e/(1.0+0.83*x_e)),0.824))*(Ith(y,'+str(indexH2)+')+0.53*Ith(y,'+str(indexH)+')));\n  zeta_He = 1.0/(487.*(1.0+12.5*pow(x_e,0.994))*(Ith(y,'+str(indexHe)+')));\n'
    return xrayParameterEquations



# Determine if the specified reactants and products represent the grain surface formation of H2
def is_H2_formation(reactants, products):
    nReactants = len([species for species in reactants if species != ''])
    nProducts  = len([species for species in products  if species != ''])
    if nReactants == 2 and nProducts == 1:
        if reactants[0] == 'H' and reactants[1] == 'H' and products[0] == 'H2': return True
    if nReactants == 3 and nProducts == 2:
        if reactants[0] == 'H' and reactants[1] == 'H' and reactants[2] == '#' and products[0] == 'H2' and products[1] == '#': return True
    return False



# Create the appropriate multiplication string for a given number
def multiple(number):
    if number == 1: return ''
    else: return str(number)+'*'



# Write the ODEs file in C language format
def write_odes_c(fileName, speciesList, constituentList, reactants, products, logForm=False):
    nSpecies = len(speciesList)
    nReactions = len(reactants)
    output = open(fileName, mode='w')

    # Determine if X-ray reactions are present in the chemical network
    if sum([reactantList.count('XRAY')+reactantList.count('XRSEC') for reactantList in reactants]) > 0:
        xrayReactions = True
    else:
        xrayReactions = False

    # Find the index numbers for H, H2 and He in the species list
    indexH  = speciesList.index('H')
    indexH2 = speciesList.index('H2')
    indexHe = speciesList.index('He')

    # Write the comments and function header

    # if logForm:
        # if xrayReactions:
            # fileHeader = '/*=======================================================================\n\n User-supplied f (ODEs) routine. Compute the function ydot = f(t,y)\n\n-----------------------------------------------------------------------*/\nstatic int f(realtype t, N_Vector y, N_Vector ydot, void *user_data)\n{\n  realtype x['+str(nSpecies)+'], *ode, *rate;\n  realtype n_H, x_e, loss, form, zeta_H, zeta_H2, zeta_He;\n  User_Data data;\n\n  /* Obtain the pointer to the ydot vector data array */\n  ode = NV_DATA_S(ydot);\n\n  /* Retrieve the array of reaction rate coefficients and\n   * the total number density from the user-supplied data */\n  data = (User_Data) user_data;\n  rate = data->rate;\n  n_H = data->n_H;\n\n  /* Convert the abundances from logarithmic to normal form */\n  for (int i = 0; i < neq; i++) {\n    x[i] = exp(NV_Ith_S(y,i));\n  }\n\n  /* The electron abundance is a conserved quantity, given by the sum\n   * of the abundances of all ionized species in the chemical network */\n'
        # else:
            # fileHeader = '/*=======================================================================\n\n User-supplied f (ODEs) routine. Compute the function ydot = f(t,y)\n\n-----------------------------------------------------------------------*/\nstatic int f(realtype t, N_Vector y, N_Vector ydot, void *user_data)\n{\n  realtype x['+str(nSpecies)+'], *ode, *rate;\n  realtype n_H, x_e, loss, form;\n  User_Data data;\n\n  /* Obtain the pointer to the ydot vector data array */\n  ode = NV_DATA_S(ydot);\n\n  /* Retrieve the array of reaction rate coefficients and\n   * the total number density from the user-supplied data */\n  data = (User_Data) user_data;\n  rate = data->rate;\n  n_H = data->n_H;\n\n  /* Convert the abundances from logarithmic to normal form */\n  for (int i = 0; i < neq; i++) {\n    x[i] = exp(NV_Ith_S(y,i));\n  }\n\n  /* The electron abundance is a conserved quantity, given by the sum\n   * of the abundances of all ionized species in the chemical network */\n'
    # else:
        # if xrayReactions:
            # fileHeader = '/*=======================================================================\n\n User-supplied f (ODEs) routine. Compute the function ydot = f(t,y)\n\n-----------------------------------------------------------------------*/\nstatic int f(realtype t, N_Vector y, N_Vector ydot, void *user_data)\n{\n  realtype *x, *ode, *rate;\n  realtype n_H, x_e, loss, form, zeta_H, zeta_H2, zeta_He;\n  User_Data data;\n\n  /* Obtain pointers to the y and ydot vector data arrays */\n  x = NV_DATA_S(y);\n  ode = NV_DATA_S(ydot);\n\n  /* Retrieve the array of reaction rate coefficients and\n   * the total number density from the user-supplied data */\n  data = (User_Data) user_data;\n  rate = data->rate;\n  n_H = data->n_H;\n\n  /* The electron abundance is a conserved quantity, given by the sum\n   * of the abundances of all ionized species in the chemical network */\n'
        # else:
            # fileHeader = '/*=======================================================================\n\n User-supplied f (ODEs) routine. Compute the function ydot = f(t,y)\n\n-----------------------------------------------------------------------*/\nstatic int f(realtype t, N_Vector y, N_Vector ydot, void *user_data)\n{\n  realtype *x, *ode, *rate;\n  realtype n_H, x_e, loss, form;\n  User_Data data;\n\n  /* Obtain pointers to the y and ydot vector data arrays */\n  x = NV_DATA_S(y);\n  ode = NV_DATA_S(ydot);\n\n  /* Retrieve the array of reaction rate coefficients and\n   * the total number density from the user-supplied data */\n  data = (User_Data) user_data;\n  rate = data->rate;\n  n_H = data->n_H;\n\n  /* The electron abundance is a conserved quantity, given by the sum\n   * of the abundances of all ionized species in the chemical network */\n'

    with open("standard_code/rate_equations_hd.txt") as header:
        data=header.readlines()
    fileHeader = ""
    for line in data:
        fileHeader = fileHeader + line
    output.write(fileHeader)

    # Prepare and write the electron conservation equation
    output.write(conserve_species('e-', constituentList, codeFormat='C'))

    # If X-ray reactions are present, write the additional terms needed to calculate their rates
    if xrayReactions:
        output.write(xray_parameters(speciesList, codeFormat='C'))

    output.write("  data->electron_abundance = x_e;\n")

    # Prepare and write the loss and formation terms for each ODE
    output.write('\n\n  /* The ODEs created by MakeRates begin here... */\n')
    if useTtk:
        app.progressValue.set(0)
        Style().configure('Horizontal.TProgressbar', background='#dcdad5')
    for n in range(nSpecies):
        if useTtk:
            app.progressValue.set(100.0*float(n)/float(nSpecies))
            app.status.update_idletasks()
        species = speciesList[n]
        lossString = '' ; formString = ''

        # Loss terms
        for i in range(nReactions):
            if reactants[i].count(species) > 0:
                if is_H2_formation(reactants[i], products[i]):
                    lossString += '-2*cell[o].rate['+str(i)+']*n_H'
                    continue
                lossString += '-'+multiple(reactants[i].count(species))+'cell[o].rate['+str(i)+']'
                for reactant in speciesList:
                    if reactant == species:
                        for j in range(reactants[i].count(reactant)-1):
                            lossString += '*Ith(y,'+str(speciesList.index(reactant))+')*n_H'
                        continue
                    for j in range(reactants[i].count(reactant)):
                        lossString += '*Ith(y,'+str(speciesList.index(reactant))+')*n_H'
                for j in range(reactants[i].count('e-')):
                    lossString += '*x_e*n_H'

                # X-ray induced secondary ionization
                if reactants[i].count('XRSEC') == 1:
                    if reactants[i].count('H') == 1:
                        lossString += '*zeta_H'
                    elif reactants[i].count('H2') == 1:
                        lossString += '*zeta_H2'
                    elif reactants[i].count('He') == 1:
                        lossString += '*zeta_He'
                    else:
                        lossString += '*zeta_H'

                # Photoreactions due to X-ray induced secondary photons (Lyman-alpha from excited H)
                if reactants[i].count('XRLYA') == 1:
                    lossString += '*Ith(y,'+str(indexH)+')*zeta_H'

                # Photoreactions due to X-ray induced secondary photons (Lyman-Werner from excited H2)
                if reactants[i].count('XRPHOT') == 1:
                    lossString += '*Ith(y,'+str(indexH2)+')*zeta_H2'

            # Formation terms
            if products[i].count(species) > 0:
                if is_H2_formation(reactants[i], products[i]):
                    formString += '+cell[o].rate['+str(i)+']*Ith(y,'+str(speciesList.index('H'))+')*n_H'
                    continue
                formString += '+'+multiple(products[i].count(species))+'cell[o].rate['+str(i)+']'
                for reactant in speciesList:
                    for j in range(reactants[i].count(reactant)):
                        formString += '*Ith(y,'+str(speciesList.index(reactant))+')'
                for j in range(reactants[i].count('e-')):
                    formString += '*x_e'
                if sum([speciesList.count(reactant) for reactant in reactants[i]]) > 1 or reactants[i].count('e-') > 0:
                    formString += '*n_H'

                # X-ray induced secondary ionization
                if reactants[i].count('XRSEC') == 1:
                    if reactants[i].count('H') == 1:
                        formString += '*zeta_H'
                    elif reactants[i].count('H2') == 1:
                        formString += '*zeta_H2'
                    elif reactants[i].count('He') == 1:
                        formString += '*zeta_He'
                    else:
                        formString += '*zeta_H'

                # Photoreactions due to X-ray induced secondary photons (Lyman-alpha from excited H)
                if reactants[i].count('XRLYA') == 1:
                    formString += '*Ith(y,'+str(indexH)+')*zeta_H'

                # Photoreactions due to X-ray induced secondary photons (Lyman-Werner from excited H2)
                if reactants[i].count('XRPHOT') == 1:
                    formString += '*Ith(y,'+str(indexH2)+')*zeta_H2'

        if lossString != '':
            lossString = '\n  loss = '+lossString+';\n'
            output.write(lossString)
        if formString != '':
            formString = '  form = '+formString+';\n'
            output.write(formString)
        ydotString = '  Ith(ydot,'+str(n)+') = '
        if formString != '':
            ydotString += 'form'
            if lossString != '': ydotString += '+'
        if lossString != '':
            ydotString += 'Ith(y,'+str(n)+')*loss'
        ydotString += ';\n'
        output.write(ydotString)

    # If the logarithmic form of the ODEs is to be used, divide each by its abundance
    if logForm:
        output.write('\n')
        output.write('\n  /* Convert the ODEs from dy/dt to d[ln(y)]/dt by dividing each by its abundance */\n')
        for n in range(nSpecies):
            output.write('  Ith(ydot,'+str(n)+') = Ith(ydot,'+str(n)+')/Ith(y,'+str(n)+');\n')

    # Write the function footer
    fileFooter = '\n\n  return(0);\n}\n /*-----------------------------------------------------------------------------------------------*/\n\n'
    output.write(fileFooter)
    output.close()



# Write the Jacobian matrix file in C language format
def write_jac_c(fileName, speciesList, reactants, products, logForm=False):
    nSpecies = len(speciesList)
    nReactions = len(reactants)
    output = open(fileName, mode='w')

    # Determine if X-ray reactions are present in the chemical network
    if sum([reactantList.count('XRAY')+reactantList.count('XRSEC') for reactantList in reactants]) > 0:
        xrayReactions = True
    else:
        xrayReactions = False

    # Find the index numbers for H, H2 and He in the species list
    indexH  = speciesList.index('H')
    indexH2 = speciesList.index('H2')
    indexHe = speciesList.index('He')

    # Specify the appropriate format code to represent the matrix indices
    if nSpecies >= 1000:
        formatCode = '%4i'
    elif nSpecies >=100:
        formatCode = '%3i'
    elif nSpecies >= 10:
        formatCode = '%2i'
    else:
        formatCode = '%i'

#    jacobian_header = np.loadtxt("jacobian_header.txt", dtype=str)

#    print jacobian_header

#    for line in jacobian_header:
#        print line
#        fileHeader = fileHeader + line




    # Write the comments and function header

    # if xrayReactions:
       # fileHeader = '/*=======================================================================\n\n User-supplied Jacobian routine. Compute the function J(t,y) = df/dy\n\n-----------------------------------------------------------------------*/\nstatic int Jac(long int N, realtype t, N_Vector y, N_Vector fy, DlsMat J,\n               void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)\n{\n  realtype *x, *rate;\n  realtype n_H, x_e, zeta_H, zeta_H2, zeta_He;\n  User_Data data;\n\n  /* Obtain a pointer to the y vector data array */\n  x = NV_DATA_S(y);\n\n  /* Retrieve the array of reaction rate coefficients, total number\n   * density and the electron abundance from the user-supplied data */\n  data = (User_Data) user_data;\n  rate = data->rate;\n  n_H = data->n_H;\n  x_e = data->x_e;\n'
    # else:
       # fileHeader = '/*=======================================================================\n\n User-supplied Jacobian routine. Compute the function J(t,y) = df/dy\n\n-----------------------------------------------------------------------*/\nstatic int Jac(long int N, realtype t, N_Vector y, N_Vector fy, DlsMat J,\n               void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)\n{\n  realtype *x, *rate;\n  realtype n_H, x_e;\n  User_Data data;\n\n  /* Obtain a pointer to the y vector data array */\n  x = NV_DATA_S(y);\n\n  /* Retrieve the array of reaction rate coefficients, total number\n   * density and the electron abundance from the user-supplied data */\n  data = (User_Data) user_data;\n  rate = data->rate;\n  n_H = data->n_H;\n  x_e = data->x_e;\n'

    with open("standard_code/jacobian_hd.txt") as header:
        data=header.readlines()
    fileHeader = ""
    for line in data:
        fileHeader = fileHeader + line
    output.write(fileHeader)

    # If X-ray reactions are present, write the additional terms needed to calculate their rate partial derivatives
    if xrayReactions:
        output.write(xray_parameters(speciesList, codeFormat='C'))
        additionalString = '\n  /* The additional partial derivative terms for the X-ray secondary ionization reactions begin here... */\n'

    # Prepare and write the terms for each Jacobian matrix element
    output.write('\n  /* The Jacobian matrix created by MakeRates begin here... */\n')

    # Prepare and write the electron conservation equation
    output.write(conserve_species('e-', constituentList, codeFormat='C'))

    output.write("  data->electron_abundance = x_e;\n\n")

    for n in range(nSpecies):
        species1 = speciesList[n]
        for m in range(nSpecies):
            species2 = speciesList[m]
            matrixString = ''

            # Loss terms for species1
            for i in range(nReactions):
                if reactants[i].count(species1) > 0 and reactants[i].count(species2) > 0:
                    if is_H2_formation(reactants[i], products[i]):
                        matrixString += '-2*cell[o].rate['+str(i)+']*n_H'
                        continue
                    matrixString += '-'+multiple(reactants[i].count(species1))+multiple(reactants[i].count(species1))+'cell[o].rate['+str(i)+']'
                    for reactant in speciesList:
                        if reactant == species2:
                            for j in range(reactants[i].count(reactant)-1):
                                matrixString += '*Ith(y,'+str(speciesList.index(reactant))+')*n_H'
                        else:
                            if reactant == species1:
                                for j in range(reactants[i].count(reactant)):
                                    matrixString += '*Ith(y,'+str(speciesList.index(reactant))+')*n_H'
                            else:
                                for j in range(reactants[i].count(reactant)):
                                    matrixString += '*Ith(y,'+str(speciesList.index(reactant))+')*n_H'
                    for j in range(reactants[i].count('e-')):
                        matrixString += '*x_e*n_H'

                    # X-ray induced secondary ionization
                    if reactants[i].count('XRSEC') == 1:
                        if reactants[i].count('H') == 1:
                            matrixString = matrixString[:-len('-cell[o].rate['+str(i)+']')]
                            additionalString += '  IJth(J,'+str(formatCode % indexH2)+','+str(formatCode % n)+') -= cell[o].rate['+str(i)+']*Ith(y,'+str(indexH) +')*zeta_H*(-1.89/(Ith(y,'+str(indexH)+')+1.89*Ith(y,'+str(indexH2)+')));\n'
                            additionalString += '  IJth(J,'+str(formatCode % indexH) +','+str(formatCode % n)+') -= cell[o].rate['+str(i)+']*Ith(y,'+str(indexH2)+')*zeta_H*(+1.89/(Ith(y,'+str(indexH)+')+1.89*Ith(y,'+str(indexH2)+')));\n'
                        elif reactants[i].count('H2') == 1:
                            matrixString = matrixString[:-len('-cell[o].rate['+str(i)+']')]
                            additionalString += '  IJth(J,'+str(formatCode % indexH2)+','+str(formatCode % n)+') -= cell[o].rate['+str(i)+']*Ith(y,'+str(indexH) +')*zeta_H2*(+0.53/(Ith(y,'+str(indexH2)+')+0.53*Ith(y,'+str(indexH)+')));\n'
                            additionalString += '  IJth(J,'+str(formatCode % indexH) +','+str(formatCode % n)+') -= cell[o].rate['+str(i)+']*Ith(y,'+str(indexH2)+')*zeta_H2*(-0.53/(Ith(y,'+str(indexH2)+')+0.53*Ith(y,'+str(indexH)+')));\n'
                        elif reactants[i].count('He') == 1:
                            matrixString += '*zeta_He'
                        else:
                            matrixString += '*zeta_H'
                            additionalString += '  IJth(J,'+str(formatCode % indexH2)+','+str(formatCode % n)+') -= cell[o].rate['+str(i)+']*Ith(y,'+str(n)+')*zeta_H*(-1.89/(Ith(y,'+str(indexH)+')+1.89*Ith(y,'+str(indexH2)+')));\n'
                            additionalString += '  IJth(J,'+str(formatCode % indexH) +','+str(formatCode % n)+') -= cell[o].rate['+str(i)+']*Ith(y,'+str(n)+')*zeta_H*(-1.00/(Ith(y,'+str(indexH)+')+1.89*Ith(y,'+str(indexH2)+')));\n'

                    # Photoreactions due to X-ray induced secondary photons (Lyman-alpha from excited H)
                    if reactants[i].count('XRLYA') == 1:
                        matrixString += '*Ith(y,'+str(indexH)+')*zeta_H'
#                        additionalString += '  IJth(J,'+str(formatCode % indexH2)+','+str(formatCode % n)+') -= cell[o].rate['+str(i)+']*Ith(y,'+str(n)+')*zeta_H*(-1.89*Ith(y,'+str(indexH)+ ')/(Ith(y,'+str(indexH)+')+1.89*Ith(y,'+str(indexH2)+')));\n'
#                        additionalString += '  IJth(J,'+str(formatCode % indexH) +','+str(formatCode % n)+') -= cell[o].rate['+str(i)+']*Ith(y,'+str(n)+')*zeta_H*(+1.89*Ith(y,'+str(indexH2)+')/(Ith(y,'+str(indexH)+')+1.89*Ith(y,'+str(indexH2)+')));\n'

                    # Photoreactions due to X-ray induced secondary photons (Lyman-Werner from excited H2)
                    if reactants[i].count('XRPHOT') == 1:
                        matrixString += '*Ith(y,'+str(indexH2)+')*zeta_H2'
#                        additionalString += '  IJth(J,'+str(formatCode % indexH2)+','+str(formatCode % n)+') -= cell[o].rate['+str(i)+']*Ith(y,'+str(n)+')*zeta_H2*(+0.53*Ith(y,'+str(indexH)+ ')/(Ith(y,'+str(indexH2)+')+0.53*Ith(y,'+str(indexH)+')));\n'
#                        additionalString += '  IJth(J,'+str(formatCode % indexH) +','+str(formatCode % n)+') -= cell[o].rate['+str(i)+']*Ith(y,'+str(n)+')*zeta_H2*(-0.53*Ith(y,'+str(indexH2)+')/(Ith(y,'+str(indexH2)+')+0.53*Ith(y,'+str(indexH)+')));\n'

            # Formation terms for species1
            for i in range(nReactions):
                if products[i].count(species1) > 0 and reactants[i].count(species2) > 0:
                    if is_H2_formation(reactants[i], products[i]):
                        matrixString += '+cell[o].rate['+str(i)+']*n_H'
                        continue
                    matrixString += '+'+multiple(products[i].count(species1))+'cell[o].rate['+str(i)+']'
                    for reactant in speciesList:
                        if reactant == species2:
                            for j in range(reactants[i].count(reactant)-1):
                                matrixString += '*Ith(y,'+str(speciesList.index(reactant))+')*n_H'
                        else:
                            for j in range(reactants[i].count(reactant)):
                                matrixString += '*Ith(y,'+str(speciesList.index(reactant))+')*n_H'
                    for j in range(reactants[i].count('e-')):
                        matrixString += '*x_e*n_H'

                    # X-ray induced secondary ionization
                    if reactants[i].count('XRSEC') == 1:
                        if reactants[i].count('H') == 1:
                            matrixString = matrixString[:-len('+cell[o].rate['+str(i)+']')]
                            additionalString += '  IJth(J,'+str(formatCode % indexH2)+','+str(formatCode % n)+') += cell[o].rate['+str(i)+']*Ith(y,'+str(indexH) +')*zeta_H*(-1.89/(Ith(y,'+str(indexH)+')+1.89*Ith(y,'+str(indexH2)+')));\n'
                            additionalString += '  IJth(J,'+str(formatCode % indexH) +','+str(formatCode % n)+') += cell[o].rate['+str(i)+']*Ith(y,'+str(indexH2)+')*zeta_H*(+1.89/(Ith(y,'+str(indexH)+')+1.89*Ith(y,'+str(indexH2)+')));\n'
                        elif reactants[i].count('H2') == 1:
                            matrixString = matrixString[:-len('+cell[o].rate['+str(i)+']')]
                            additionalString += '  IJth(J,'+str(formatCode % indexH2)+','+str(formatCode % n)+') += cell[o].rate['+str(i)+']*Ith(y,'+str(indexH) +')*zeta_H2*(+0.53/(Ith(y,'+str(indexH2)+')+0.53*Ith(y,'+str(indexH)+')));\n'
                            additionalString += '  IJth(J,'+str(formatCode % indexH) +','+str(formatCode % n)+') += cell[o].rate['+str(i)+']*Ith(y,'+str(indexH2)+')*zeta_H2*(-0.53/(Ith(y,'+str(indexH2)+')+0.53*Ith(y,'+str(indexH)+')));\n'
                        elif reactants[i].count('He') == 1:
                            matrixString += '*zeta_He'
                        else:
                            matrixString += '*zeta_H'
                            additionalString += '  IJth(J,'+str(formatCode % indexH2)+','+str(formatCode % n)+') += cell[o].rate['+str(i)+']*Ith(y,'+str(m)+')*zeta_H*(-1.89/(Ith(y,'+str(indexH)+')+1.89*Ith(y,'+str(indexH2)+')));\n'
                            additionalString += '  IJth(J,'+str(formatCode % indexH) +','+str(formatCode % n)+') += cell[o].rate['+str(i)+']*Ith(y,'+str(m)+')*zeta_H*(-1.00/(Ith(y,'+str(indexH)+')+1.89*Ith(y,'+str(indexH2)+')));\n'

                    # Photoreactions due to X-ray induced secondary photons (Lyman-alpha from excited H)
                    if reactants[i].count('XRLYA') == 1:
                        matrixString += '*Ith(y,'+str(indexH)+')*zeta_H'
#                        additionalString += '  IJth(J,'+str(formatCode % indexH2)+','+str(formatCode % n)+') += cell[o].rate['+str(i)+']*Ith(y,'+str(m)+')*zeta_H*(-1.89*Ith(y,'+str(indexH)+ ')/(Ith(y'+str(indexH)+')+1.89*Ith(y'+str(indexH2)+')));\n'
#                        additionalString += '  IJth(J,'+str(formatCode % indexH) +','+str(formatCode % n)+') += cell[o].rate['+str(i)+']*Ith(y,'+str(m)+')*zeta_H*(+1.89*Ith(y,'+str(indexH2)+')/(Ith(y'+str(indexH)+')+1.89*Ith(y'+str(indexH2)+')));\n'

                    # Photoreactions due to X-ray induced secondary photons (Lyman-Werner from excited H2)
                    if reactants[i].count('XRPHOT') == 1:
                        matrixString += '*Ith(y,'+str(indexH2)+')*zeta_H2'
#                        additionalString += '  IJth(J,'+str(formatCode % indexH2)+','+str(formatCode % n)+') += cell[o].rate['+str(i)+']*Ith(y,'+str(m)+')*zeta_H2*(+0.53*Ith(y,'+str(indexH)+ ')/(Ith(y'+str(indexH2)+')+0.53*Ith(y'+str(indexH)+')));\n'
#                        additionalString += '  IJth(J,'+str(formatCode % indexH) +','+str(formatCode % n)+') += cell[o].rate['+str(i)+']*Ith(y,'+str(m)+')*zeta_H2*(-0.53*Ith(y,'+str(indexH2)+')/(Ith(y'+str(indexH2)+')+0.53*Ith(y'+str(indexH)+')));\n'

            if matrixString != '':
                matrixString = '  IJth(J,'+str(formatCode % m)+','+str(formatCode % n)+') = '+matrixString+';\n'
                output.write(matrixString)

    # If X-ray reactions are present, write their additional partial derivative terms
    if xrayReactions:
        output.write(additionalString)

    # If the logarithmic form of the ODEs is to be used, multiply each matrix element J_ij by y_j/y_i
    if logForm:
        output.write('\n')
        for n in range(nSpecies):
            for m in range(nSpecies):
                if n != m:
                    output.write('  IJth(J,'+str(formatCode % m)+','+str(formatCode % n)+') = IJth(J,'+str(m)+','+str(n)+')*Ith(y,'+str(m)+')/Ith(y,'+str(n)+');\n')

    # Write the function footer
    fileFooter = '\n  return(0);\n}\n/*-----------------------------------------------------------------------------------------------*/\n\n'
    output.write(fileFooter)
    output.close()





###   Global code   ###
###   -----------   ###



# Read and verify the command line keyword options (if any have been specified)
# usageString = sys.argv[0]+' [reactionFile=ReactionFileName] [speciesFile=SpeciesFileName] [outputPrefix=OutputFilePrefix] [sortSpecies=True|False] [logForm=True|False] [fileFormat=Rate95|Rate99|Rate05] [codeFormat=F77|F90|C]\n'
# keywordNames = ['reactionFile=','speciesFile=','outputPrefix=','sortSpecies=','logForm=','fileFormat=','codeFormat=']
# keywordValue = [reactionFile,speciesFile,outputPrefix,('True'if sortSpecies else 'False'),('True'if logForm else 'False'),fileFormat,codeFormat]
#
# for n in range(1,len(sys.argv)):
#     if sum([sys.argv[n].lower().count(keyword.lower()) for keyword in keywordNames]) == 0:
#         sys.exit('\n  ERROR! Unrecognised keyword: '+sys.argv[n]+'\n\nUsage: '+usageString)
#
#     for i, keyword in enumerate(keywordNames):
#         if sys.argv[n].lower().count(keyword.lower()) != 0:
#             index = sys.argv[n].lower().index(keyword.lower())+len(keyword)
#             keywordValue[i] = sys.argv[n][index:].strip()



# if keywordValue[0] != '':
#     reactionFile = keywordValue[0]
# else:
#     reactionFile = '<None>'
#
# if keywordValue[1] != '':
#     speciesFile = keywordValue[1]
# else:
#     speciesFile = '<None>'


print " "
print "MakeRates for Magritte"
print "----------------------"
print " "


# Get the input files from parameters.txt

with open("../src/parameters.hpp") as parameters_file:
    for line in parameters_file:
        line = line.split()
        if len(line) is 3:
            if line[1] == 'SPEC_DATAFILE':
                speciesFile = "../" + line[2].split("\"")[1]
            if line[1] == 'REAC_DATAFILE':
                reactionFile = "../" + line[2].split("\"")[1]


outputPrefix = '<None>'

sortSpecies = False

logForm = False

fileFormat = "Rate05"

codeFormat = "C"



if (os.stat(reactionFile).st_size > 0 or os.stat(speciesFile).st_size > 0):

    # Read the reactants, products, Arrhenius equation parameters and measurement labels for each reaction
    print '\n  Reading reaction file...'
    nReactions, reactants, products, alpha, beta, gamma, labels = read_reaction_file(reactionFile)

    # Read the name, abundance and molecular mass for each species
    if speciesFile != '':
        print '  Reading species file...'
        nSpecies, speciesList, abundanceList, massList = read_species_file(speciesFile)

    # Find the total number and full list of reactions containing only these species
        print '\n  Finding all reactions involving these species...'
        nReactions, reactants, products, alpha, beta, gamma, labels = find_all_reactions(speciesList, reactants, products, alpha, beta, gamma, labels)

    # Find the total number and full list of unique species contained in the reactions
    else:
        print '\n  Finding all species involved in the reactions...'
        nSpecies, speciesList = find_all_species(reactants, products)
        abundanceList = [float(0) for i in range(nSpecies)]

    print '\n  Number of reactions:',nReactions
    print '  Number of species:',nSpecies


    # Check for "orphan" species that are either never formed or never destroyed
    print '\n  Checking for species without formation/destruction reactions...'
    nFormation, nDestruction, missingList = check_orphan_species(speciesList, reactants, products)

    # Sort the species first by number of destruction reactions, then by number of formation reactions
    if sortSpecies:
        print '\n  Sorting the species by number of formation reactions...'
        speciesList = sort_species(speciesList, nFormation, nDestruction)

    # Calculate the molecular mass and elemental constituents of each species
    print '\n  Calculating molecular masses and elemental constituents...'
    massList, constituentList, elementList = find_constituents(speciesList)

    # Write the ODEs in the appropriate language format
    if codeFormat == 'C':
        print '  Writing system of ODEs in C format...'
        filename = '../src/sundials/rate_equations.cpp'
        write_odes_c(filename, speciesList, constituentList, reactants, products, logForm=logForm)

    # Write the Jacobian matrix in the appropriate language format
    if codeFormat == 'C':
        print '  Writing Jacobian matrix in C format...'
        filename = '../src/sundials/jacobian.cpp'
        write_jac_c(filename, speciesList, reactants, products, logForm=False)

print '\n  Finished! \n\n'
print " "
