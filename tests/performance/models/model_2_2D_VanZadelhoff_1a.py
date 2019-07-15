#!/usr/bin/env python
# coding: utf-8

# # van Zadelhoff et al. (2002) benchmark problem 1a
# ---

# In[1]:


import os, inspect
thisFolder     = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
magritteFolder = f'{thisFolder}/../../../'


# ## 0) Setup
# ---

# Add Magritte's `/setup/` and `/bin/` directories to the Python path.

# In[2]:


from sys import path as sysPath
sysPath.insert (0, f'{magritteFolder}/setup/')
sysPath.insert (0, f'{magritteFolder}/bin/')


# In[3]:


print(magritteFolder)


# Import Magritte's Python modules and setup.

# In[4]:


from magritte import Model, Long1, Long2, Double1, Double2, String1

from setup import Setup, linedata_from_LAMDA_file


# ## 1) Define model
# ---

# Problem 1a in the van Zadelhoff et al. (2002) benchmark considers a model with a ficticious 2-level species in a spherically symmetric geometry where only the density varies with the radius and all other quantities are constant.

# \begin{align}
# \rho(r)  \ &= \ \rho_{\text{in}} \ \left(\frac{r_{\text{in}}}{r}\right)^{2} \\
# n_{i}(r) \ &= \ X_{\text{mol}} \ \rho(r)                                    \\
# \end{align}

# All constants are given by

# \begin{align}
# r_{\text{in}}    \ &= \ 1.0 \times 10^{13} \ \text{m}      \\
# r_{\text{out}}   \ &= \ 7.8 \times 10^{16} \ \text{m}      \\
# \rho_{\text{in}} \ &= \ 2.0 \times 10^{13} \ \text{m}^{-3} \\
# X_{\text{mol}}   \ &= \ 10^{-8}                            \\
# T(r)             \ &= \ 20 \ \text{K} \\
# \end{align}

# Define helper quantities for the model.

# In[5]:


r_in   = 1.0E13   # [m]
r_out  = 7.8E16   # [m]

rho_in = 2.0E13    # [m^-3]
X_mol  = 1.0E-8   # [.]

T      = 20.0     # [K]

turb   = 150.0    # [m/s]


# In[6]:


def rho (r):
    if (r >= r_in):
        return rho_in * np.power(r_in/r,     2.0)
    else:
        return rho_in * np.power(r_in/r_out, 2.0)

def abn (r):
    return X_mol * rho(r)


# In[7]:


dimension = 1
ncells    = 25
nrays     = 2
nspecs    = 5
nlspecs   = 1
nquads    = 21


# In[8]:


import numpy as np

base = 10

log_r_in  = np.log(r_in)       / np.log(base)
log_r_out = np.log(r_out)      / np.log(base)

grid = np.logspace (log_r_in, log_r_out, ncells, base=base, endpoint=True)
# grid = np.linspace (r_in, r_out, ncells, endpoint=True)


# In[9]:


setup = Setup (dimension = dimension)


# Create a Magritte model object.

# In[10]:


model1D = Model ()


# Define model parameters.

# In[11]:


model1D.parameters.set_ncells  (ncells)
model1D.parameters.set_nrays   (nrays)
model1D.parameters.set_nspecs  (nspecs)
model1D.parameters.set_nlspecs (nlspecs)
model1D.parameters.set_nquads  (nquads)


# Define geometry. First define cells.

# In[12]:


model1D.geometry.cells.x  = Double1 (grid)
model1D.geometry.cells.y  = Double1 ([0.0 for i in range(ncells)])
model1D.geometry.cells.z  = Double1 ([0.0 for i in range(ncells)])

model1D.geometry.cells.vx = Double1 ([0.0 for i in range(ncells)])
model1D.geometry.cells.vy = Double1 ([0.0 for i in range(ncells)])
model1D.geometry.cells.vz = Double1 ([0.0 for i in range(ncells)])

# Note that the points need to be specified before neighbors can be found
model1D.geometry.cells = setup.neighborLists (model1D.geometry.cells)


# Then define the boundary of the geometry.

# In[13]:


model1D.geometry.boundary.boundary2cell_nr = Long1 ([0, ncells-1])


# Finally, define the rays for the geometry.

# In[14]:


model1D.geometry.rays = setup.rays (nrays=nrays, cells=model1D.geometry.cells)


# Define thermodynamics.

# In[15]:


model1D.thermodynamics.temperature.gas   = Double1 ([T    for i in range(ncells)])
model1D.thermodynamics.turbulence.vturb2 = Double1 ([turb for i in range(ncells)])


# Define the chemical species involved.

# In[16]:


model1D.chemistry.species.abundance = Double2 ([ Double1 ([0.0, abn(r), rho(r), 0.0, 1.0]) for r in grid])
model1D.chemistry.species.sym       = String1 (['dummy0', 'test', 'H2', 'e-', 'dummy1'])


# Define the folder containing the linedata.

# In[17]:


linedataFolder = f'{thisFolder}/data/Linedata/test.txt'


# Define the linedata.

# In[18]:


model1D.lines.lineProducingSpecies.append (linedata_from_LAMDA_file (linedataFolder, model1D.chemistry.species))


# Define the quadrature roots and weights.

# In[19]:


import quadrature

model1D.lines.lineProducingSpecies[0].quadrature.roots   = Double1 (quadrature.H_roots   (nquads))
model1D.lines.lineProducingSpecies[0].quadrature.weights = Double1 (quadrature.H_weights (nquads))


# ## 1.2) Map to 2D
# ---

# In[20]:


dimension = 2
nrays     = 1500


# In[21]:


def number_of_points_in_shell (s):
    return int(6+2.0*np.pi*np.log(s+1))


# In[22]:


cellsInShell = []
index        = 0

for s in range (ncells):
    cellsInShell.append ([])
    for _ in range (number_of_points_in_shell(s)):
        cellsInShell[s].append (index)
        index += 1


# In[23]:


from mapModel import mapToXD

model = mapToXD (model1D=model1D, dimension=dimension, nrays=nrays, cellsInShell=cellsInShell)


# In[24]:


print('ncells =', model.parameters.ncells())


# ## 2) Write model file
# ---

# In[25]:


#from ioMagritte import IoPython
from ioMagritte import IoText
#from os         import remove
from setup      import make_file_structure
from shutil     import rmtree


# In[26]:


#modelName = f'{ProjectFolder}model_problem_1a.hdf5'
modelName = f'{thisFolder}/model_2_2D_VanZadelhoff_1a/'


# Remove old model.

# In[27]:


#remove(modelName)
rmtree(modelName)


# Define an io object to handle input and output. (In this case via Python using HDF5.)

# In[28]:


#io = IoPython ("hdf5", modelName)
io = IoText (modelName)


# In[29]:


make_file_structure (modelName)


# In[30]:


model.write (io)


# In[ ]:




