
import numpy as np
import re


# Defs
# ----

def readDataInt(fileName, start, stop, regex):
    # Read integers in 'data' from line 'start' until line 'stop' following 'regex'
    with open(fileName) as dataFile:
        data = dataFile.readlines()
    variable = []
    for i in range(start,stop):
        variable +=  [int(word) for word in re.findall(regex, data[i])]
    return variable


def readDataFlt(fileName, start, stop, regex):
    # Read floats in 'data' from line 'start' until line 'stop' following 'regex'
    with open(fileName) as dataFile:
        data = dataFile.readlines()
    variable = []
    for i in range(start,stop):
        variable += [float(word) for word in re.findall(regex, data[i])]
    return variable


def extractCollisionPartner(fileName, line, elem):
    # Returns collision partner and whether it is ortho or para (for H2)
    with open(fileName) as dataFile:
        data = dataFile.readlines()
    partner   = re.findall(elem.replace('+','\+')+'\s*[\+\-]?\s*([\w\+\-]+)\s*', data[line])[0]
    excess    = re.findall('[op]\-?', partner)
    if (len(excess) > 0):
        orthoPara = re.findall('[op]', partner)[0]
        partner   = partner.replace(excess[0],'')
    else:
        orthoPara = 'n'
    return [partner, orthoPara]


def readColumn(fileName, start, nElem, columnNr, type):
    with open(fileName) as dataFile:
        lineNr = 0
        column = []
        for line in dataFile:
            if (lineNr >= start) and (lineNr < start+nElem):
                if type == 'float':
                    column.append(float(line.split()[columnNr]))
                if type == 'long':
                    column.append(long(line.split()[columnNr]))
                if type == 'int':
                    column.append(int(line.split()[columnNr]))
                if type == 'str':
                    column.append(str(line.split()[columnNr]))
            lineNr += 1
    return column


def zero2(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]




# LineData class
# --------------

class LineData():

    """ Molecular line data """

    def __init__(self, fileName, dataFormat):
        # Read fileName which contains line data in dataFormat
        if dataFormat == 'LAMDA':
            self.readLamdaFile(fileName)
        if dataFormat == 'VanZadelhoff':
            self.readRadexFile(fileName)
        # Setup transition matrices with Einstein coefficients
        self.setupMatrices()

    def readLamdaFile(self, fileName):

        """ Read line data in LAMDA format """

        self.name      = readColumn(fileName, start=1,            nElem=1,         columnNr=0, type='str')[0]
        self.mass      = readColumn(fileName, start=3,            nElem=1,         columnNr=0, type='float')[0]
        self.nlev      = readColumn(fileName, start=5,            nElem=1,         columnNr=0, type='int')[0]
        self.energy    = readColumn(fileName, start=7,            nElem=self.nlev, columnNr=1, type='float')
        self.weight    = readColumn(fileName, start=7,            nElem=self.nlev, columnNr=2, type='float')
        self.nrad      = readColumn(fileName, start=8+self.nlev,  nElem=1,         columnNr=0, type='int')[0]
        self.irad      = readColumn(fileName, start=10+self.nlev, nElem=self.nrad, columnNr=1, type='int')
        self.jrad      = readColumn(fileName, start=10+self.nlev, nElem=self.nrad, columnNr=2, type='int')
        self.A_coeff   = readColumn(fileName, start=10+self.nlev, nElem=self.nrad, columnNr=3, type='float')

        nlr = self.nlev + self.nrad

        self.partner   = []
        self.orthoPara = []
        self.ncoltran  = []
        self.ncoltemp  = []
        self.coltemp   = []
        self.icol      = []
        self.jcol      = []
        self.C_coeff   = []

        self.ncolpar = readColumn(fileName, start=11+nlr, nElem=1,  columnNr=0, type='int')[0]
        index        = 13 + nlr

        # Loop over the collision partners
        for colpar in range(self.ncolpar):
            self.partner    += [extractCollisionPartner(fileName, line=index, elem=self.name)[0]]
            self.orthoPara  += [extractCollisionPartner(fileName, line=index, elem=self.name)[1]]

            self.ncoltran += readColumn(fileName, start=index+2, nElem=1, columnNr=0, type='int')
            self.ncoltemp += readColumn(fileName, start=index+4, nElem=1, columnNr=0, type='int')

            self.coltemp  += [[readColumn(fileName, start=index+6, nElem=1, columnNr=temp, type='float')[0] for temp in range(self.ncoltemp[colpar])]]

            self.icol     += [readColumn(fileName, start=index+8, nElem=self.ncoltran[colpar], columnNr=1, type='int')]
            self.jcol     += [readColumn(fileName, start=index+8, nElem=self.ncoltran[colpar], columnNr=2, type='int')]

            self.C_coeff  += [[readColumn(fileName, start=index+8, nElem=self.ncoltran[colpar], columnNr=3+temp, type='float') for temp in range(self.ncoltemp[colpar])]]

            index += 9 + self.ncoltran[colpar]


    def readRadexFile(self, fileName):

        """ Read line data in RADEX format """

        self.name     = readColumn(fileName,  start=0,  nElem=1,  columnNr=0, type='str')[0]
        self.mass     = readDataFlt(fileName, start=1,  stop=2,   regex='\d+\.\d+')[0]
        self.nlev     = readDataInt(fileName, start=2,  stop=3,   regex='\d+')[0]
        self.nrad     = readDataInt(fileName, start=2,  stop=3,   regex='\d+')[1]
        self.energy   = readDataFlt(fileName, start=3,  stop=6,   regex='\d+\.\d{7}')
        self.weight   = readDataFlt(fileName, start=6,  stop=8,   regex='\d+\.\d{1}')
        self.irad     = readDataInt(fileName, start=8,  stop=9,   regex='\d+')
        self.jrad     = readDataInt(fileName, start=9,  stop=10,  regex='\d+')
        self.A_coeff  = readDataFlt(fileName, start=10, stop=14,  regex='\d+\.\d{3}E[+-]?\d{2}')

        self.ncolpar   = 1
        self.partner   = ['H2']
        self.orthoPara = ['n']
        self.ncoltran  = [readDataInt(fileName, start=14, stop=15,  regex='\d+')[0]]
        self.ncoltemp  = [readDataInt(fileName, start=14, stop=15,  regex='\d+')[1]]
        self.coltemp   = [readDataFlt(fileName, start=14, stop=15,  regex='\d+\.\d+')]
        self.icol      = [readDataInt(fileName, start=15, stop=24,  regex='\d+')]
        self.jcol      = [readDataInt(fileName, start=24, stop=33,  regex='\d+')]
        C_temp         = readDataFlt(fileName, start=33, stop=141, regex='\d+\.\d+E[+-]?\d+')
        # Group C_coeff elements (opposite of vectorize)
        self.C_coeff = [zero2(self.ncoltemp[0], self.ncoltran[0])]
        for temp in range(self.ncoltemp[0]):
            for tran in range(self.ncoltran[0]):
                self.C_coeff[0][temp][tran] = C_temp[tran + self.ncoltran[0]*temp]


    def setupMatrices(self):

        """ Calculate derived line data """

        c  = 2.99792458E+10   # speed of light in cgs
        h  = 6.62606896E-27   # Planck's constant in cgs
        kb = 1.38065040E-16   # Boltzmann's constant in cgs units

        # Convert energies from cm^-1 to erg
        for i in range(len(self.energy)):
            self.energy[i] = h*c* self.energy[i]
        self.A         = zero2(self.nlev,self.nlev)
        self.B         = zero2(self.nlev,self.nlev)
        self.frequency = zero2(self.nlev,self.nlev)
        self.C_data    = [ [zero2(self.nlev,self.nlev) for _ in range(self.ncoltemp[colpar])] for colpar in range(self.ncolpar)]

        for k in range(self.nrad):
            self.irad[k] = self.irad[k]-1
            self.jrad[k] = self.jrad[k]-1

        for colpar in range(self.ncolpar):
            for tran in range(self.ncoltran[colpar]):
                self.icol[colpar][tran] = self.icol[colpar][tran]-1
                self.jcol[colpar][tran] = self.jcol[colpar][tran]-1

        # for i in range(self.nlev):
        #     for j in range(self.nlev):
        #         self.frequency[i][j] = (self.energy[i]-self.energy[j]) * c
        #         self.frequency[j][i] = self.frequency[i][j]

        for k in range(self.nrad):
            i = self.irad[k]
            j = self.jrad[k]
            self.frequency[i][j] = (self.energy[i]-self.energy[j]) / h
            self.frequency[j][i] = self.frequency[i][j]
            self.A[i][j] = self.A_coeff[k]
            self.B[i][j] = self.A[i][j] * c**2 / (2.0*h*self.frequency[i][j]**3)
            self.B[j][i] = self.weight[i] / self.weight[j] * self.B[i][j]

        for colpar in range(self.ncolpar):
            for temp in range(self.ncoltemp[colpar]):
                for tran in range(self.ncoltran[colpar]):
                    i = self.icol[colpar][tran]
                    j = self.jcol[colpar][tran]
                    self.C_data[colpar][temp][i][j] = self.C_coeff[colpar][temp][tran]
                for tran in range(self.ncoltran[colpar]):
                    i = self.icol[colpar][tran]
                    j = self.jcol[colpar][tran]
                    if (self.C_data[colpar][temp][j][i] == 0.0) and (self.C_data[colpar][temp][i][j] != 0.0):
                        self.C_data[colpar][temp][j][i] = self.C_data[colpar][temp][i][j] * self.weight[i] / self.weight[j] * np.exp(-(self.energy[i]-self.energy[j])/(kb * self.coltemp[colpar][temp]))
