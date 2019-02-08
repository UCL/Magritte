import numpy as np
import re

from pyMagritte import Linedata, CollisionPartner
from pyMagritte import Long1,   Long2,   Long3
from pyMagritte import Double1, Double2, Double3
from pyMagritte import vLinedata, vCollisionPartner

# Physical constants
CC = 2.99792458E+8    # [m/s] speed of light
HH = 6.62607004E-34   # [J*s] Planck's constant
KB = 1.38064852E-23   # [J/K] Boltzmann's constant



def getProperName(name):
    '''
    Return the standard name for the species
    '''
    if name in ['e']: return 'e-'
    if name in ['pH2', 'oH2', 'p-H2', 'o-H2']: return 'H2'
    # If none of the above special cases, it should be fine
    return name


def getSpeciesNumber (species, name):
    '''
    Returns number of species given by 'name'
    '''
    # Note that there are dummy species in Magritte at places 0 and NLSPEC
    if isinstance (name, list):
        return [getSpeciesNumber (species,elem) for elem in name]
    else:
        for i in range (len (species.sym)):
            if (species.sym[i] == getProperName (name)):
                return i+1
        return 0


def extractCollisionPartner (fileName, line, species, elem):
    '''
    Returns collision partner and whether it is ortho or para (for H2)
    '''
    with open (fileName) as dataFile:
        data = dataFile.readlines ()
    partner   = re.findall (elem.replace ('+','\+')+'\s*[\+\-]?\s*([\w\+\-]+)\s*', data[line])[0]
    excess    = re.findall ('[op]\-?', partner)
    if (len (excess) > 0):
        orthoPara = re.findall ('[op]', partner)[0]
        partner   = partner.replace (excess[0],'')
    else:
        orthoPara = 'n'
    return [getSpeciesNumber (species, partner), orthoPara]


class Reader ():
    def __init__ (self ,fileName):
        self.fileName = fileName

    def readColumn (self, start, nElem, columnNr, type):
        '''
        Returns a column of data as a list
        '''
        with open (self.fileName) as dataFile:
            lineNr = 0
            column = []
            for line in dataFile:
                if (lineNr >= start) and (lineNr < start+nElem):
                    if type == 'float':
                        column.append (float(line.split()[columnNr]))
                    if type == 'int':
                        column.append (int  (line.split()[columnNr]))
                    if type == 'str':
                        column.append (str  (line.split()[columnNr]))
                lineNr += 1
            if type == 'float':
                column = Double1 (column)
            if type == 'int':
                column = Long1   (column)
        return column

    def extractCollisionPartner (self, line, species, elem):
        '''
        Returns collision partner and whether it is ortho or para (for H2)
        '''
        with open (self.fileName) as dataFile:
            data = dataFile.readlines ()
        partner   = re.findall (elem.replace ('+','\+')+'\s*[\+\-]?\s*([\w\+\-]+)\s*', data[line])[0]
        excess    = re.findall ('[op]\-?', partner)
        if (len (excess) > 0):
            orthoPara = re.findall ('[op]', partner)[0]
            partner   = partner.replace (excess[0],'')
        else:
            orthoPara = 'n'
        return [getSpeciesNumber (species, partner), orthoPara]


def linedata_from_LAMDA_file (fileName, species):
    """
    Read line data in LAMDA format
    """
    # Create Lindata object
    ld = Linedata ()
    # Create reader for data file
    rd = Reader (fileName)
    # Read radiative data
    ld.sym    = rd.readColumn(start= 1,         nElem=1,       columnNr=0, type='str')[0]
    #ld.mass   = rd.readColumn(start= 3,         nElem=1,       columnNr=0, type='float')[0]
    ld.nlev   = rd.readColumn(start= 5,         nElem=1,       columnNr=0, type='int')[0]
    ld.energy = rd.readColumn(start= 7,         nElem=ld.nlev, columnNr=1, type='float')
    ld.weight = rd.readColumn(start= 7,         nElem=ld.nlev, columnNr=2, type='float')
    ld.nrad   = rd.readColumn(start= 8+ld.nlev, nElem=1,       columnNr=0, type='int')[0]
    ld.irad   = rd.readColumn(start=10+ld.nlev, nElem=ld.nrad, columnNr=1, type='int')
    ld.jrad   = rd.readColumn(start=10+ld.nlev, nElem=ld.nrad, columnNr=2, type='int')
    ld.A      = rd.readColumn(start=10+ld.nlev, nElem=ld.nrad, columnNr=3, type='float')
    # Start reading collisional data
    nlr = ld.nlev + ld.nrad

    ld.ncolpar = rd.readColumn(start=11+nlr, nElem=1,  columnNr=0, type='int')[0]
    ind        = 13 + nlr

    # Create list of CollisionPartners and cast to vCollisionPartner
    ld.colpar = vCollisionPartner ([CollisionPartner() for _ in range(ld.ncolpar)])

    # Loop over the collision partners
    for c in range(ld.ncolpar):
        ld.colpar[c].num_col_partner = rd.extractCollisionPartner(line=ind, species=species, elem=ld.sym)[0]
        ld.colpar[c].orth_or_para_H2 = rd.extractCollisionPartner(line=ind, species=species, elem=ld.sym)[1]
        ld.colpar[c].ncol =  rd.readColumn(start=ind+2, nElem=1,                 columnNr=0,   type='int')[0]
        ld.colpar[c].ntmp =  rd.readColumn(start=ind+4, nElem=1,                 columnNr=0,   type='int')[0]
        ld.colpar[c].icol =  rd.readColumn(start=ind+8, nElem=ld.colpar[c].ncol, columnNr=1,   type='int')
        ld.colpar[c].jcol =  rd.readColumn(start=ind+8, nElem=ld.colpar[c].ncol, columnNr=2,   type='int')
        tmp = []
        Cd  = []
        for t in range (ld.colpar[c].ntmp):
            tmp.append (rd.readColumn(start=ind+6, nElem=1,                 columnNr=t,   type='float')[0])
            Cd .append (rd.readColumn(start=ind+8, nElem=ld.colpar[c].ncol, columnNr=3+t, type='float'))
        ld.colpar[c].tmp = Double1 (tmp)
        ld.colpar[c].Cd  = Double2 (Cd)

        ind += 9 + ld.colpar[c].ncol

    # Change index range from [1, nlev] to [0, nlev-1]
    for k in range(ld.nrad):
        ld.irad[k] = ld.irad[k] - 1
        ld.jrad[k] = ld.jrad[k] - 1
    for c, colpar in enumerate(ld.colpar):
        for k in range(colpar.ncol):
            ld.colpar[c].icol[k] = ld.colpar[c].icol[k] - 1
            ld.colpar[c].jcol[k] = ld.colpar[c].jcol[k] - 1

    # Convert to SI units
    for i in range(ld.nlev):
        # Energy from [cm^-1] to [J]
        ld.energy[i] = ld.energy[i] * 1.0E+2 * HH*CC
    for c, colpar in enumerate(ld.colpar):
        for t in range(colpar.ntmp):
            for k in range(colpar.ncol):
                # Cd from [cm^3] to [m^3]
                ld.colpar[c].Cd[t][k] = ld.colpar[c].Cd[t][k] * 1.0E-6

    # Set derived quantities
    ld.Bs        = Double1 ([0.0 for _ in range(ld.nrad)])
    ld.Ba        = Double1 ([0.0 for _ in range(ld.nrad)])
    ld.frequency = Double1 ([0.0 for _ in range(ld.nrad)])
    for k in range(ld.nrad):
        i = ld.irad[k]
        j = ld.jrad[k]
        ld.frequency[k] = (ld.energy[i]-ld.energy[j]) / HH
        ld.Bs[k]        = ld.A[k] * CC**2 / (2.0*HH*(ld.frequency[k])**3)
        ld.Ba[k]        = ld.weight[i] / ld.weight[j] * ld.Bs[k]
    for c, colpar in enumerate(ld.colpar):
        ld.colpar[c].Ce = Double2 ([Double1([0.0 for _ in range (colpar.ncol)]) for _ in range(colpar.ntmp)])
        for t in range(colpar.ntmp):
            for k in range(colpar.ncol):
                i = colpar.icol[k]
                j = colpar.jcol[k]
                ld.colpar[c].Ce[t][k] = colpar.Cd[t][k] * ld.weight[i] / ld.weight[j] * np.exp(-(ld.energy[i]-ld.energy[j])/(KB * colpar.tmp[t]))
    # Done
    return ld
