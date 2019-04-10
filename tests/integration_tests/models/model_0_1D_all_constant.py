
# coding: utf-8

# # 0) Analytical Model: 1D all constant
# ---

# ## 0) Setup
# ---

# Add Magritte's `/setup/` and `/bin/` directories to the Python path.

# In[1]:


from sys import path
path.insert (0, '../../../setup/')
path.insert (0, '../../../bin/')


# Import Magritte's Python modules and setup.

# In[2]:


from magritte import Model, Long1, Long2, Double1, Double2, String1
from setup    import Setup, linedata_from_LAMDA_file


# ## 1) Define model
# ---

# Define helper quantities for the model.

# In[3]:


dimension = 1
ncells    = 20
nrays     = 2
nspecs    = 5
nlspecs   = 1
nquads    = 39


# In[4]:


dens = 1.0E+12   # [m^-3]
abun = 1.0E+4    # [m^-3]
temp = 1.0E+02   # [K]
turb = 0.0E+00   # [m/s]
dx   = 1.0E4     # [m]
bx   = 1.3E0
dT   = 0.0E0     # [K]
dv   = 0.0E0     # [m/s]


# In[5]:


setup = Setup (dimension = dimension)


# Create a Magritte model object.

# In[6]:


model = Model ()


# Define model parameters.

# In[7]:


model.parameters.set_ncells  (ncells)
model.parameters.set_nrays   (nrays)
model.parameters.set_nspecs  (nspecs)
model.parameters.set_nlspecs (nlspecs)
model.parameters.set_nquads  (nquads)


# Define geometry. First define cells.

# In[8]:


model.geometry.cells.x  = Double1 ([i*dx for i in range(ncells)])
model.geometry.cells.y  = Double1 ([0.0  for _ in range(ncells)])
model.geometry.cells.z  = Double1 ([0.0  for _ in range(ncells)])

model.geometry.cells.vx = Double1 ([0.0  for _ in range(ncells)])
model.geometry.cells.vy = Double1 ([0.0  for _ in range(ncells)])
model.geometry.cells.vz = Double1 ([0.0  for _ in range(ncells)])

# Note that the points need to be specified before neighbors can be found
model.geometry.cells = setup.neighborLists (model.geometry.cells)


# Then define the boundary of the geometry.

# In[9]:


model.geometry.boundary.boundary2cell_nr = Long1 ([0, ncells-1])


# Finally, define the rays for the geometry.

# In[10]:


model.geometry.rays = setup.rays (nrays=nrays, cells=model.geometry.cells)


# Define thermodynamics.

# In[11]:


model.thermodynamics.temperature.gas   = Double1 ([temp for _ in range(ncells)])
model.thermodynamics.turbulence.vturb2 = Double1 ([turb for _ in range(ncells)])


# Define the chemical species involved.

# In[12]:


model.chemistry.species.abundance = Double2 ([ Double1 ([0.0, abun, dens, 0.0, 1.0]) for _ in range(ncells)])
model.chemistry.species.sym       = String1 (['dummy0', 'test', 'H2', 'e-', 'dummy1'])


# Define the folder containing the linedata.

# In[15]:


linedataFolder = 'data/Linedata/test.txt'


# Define the linedata.

# In[16]:


model.lines.lineProducingSpecies.append (linedata_from_LAMDA_file (linedataFolder, model.chemistry.species))


# Define the quadrature roots and weights.

# In[17]:


import quadrature

model.lines.lineProducingSpecies[0].quadrature.roots   = Double1 (quadrature.H_roots   (nquads))
model.lines.lineProducingSpecies[0].quadrature.weights = Double1 (quadrature.H_weights (nquads))


# ## 2) Write input file
# ---

# In[18]:


from ioMagritte import IoPython


# In[19]:


modelName = 'model_0_1D_all_constant.hdf5'


# Define an io object to handle input and output. (In this case via Python using HDF5.)

# In[20]:


io = IoPython ("hdf5", modelName)


# In[21]:


model.write (io)

