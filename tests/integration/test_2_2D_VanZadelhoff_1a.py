#!/usr/bin/env python
# coding: utf-8

# # van Zadelhoff et al. (2002) benchmark problem 1a
# ---

# ## 0) Setup
# ---

# Add Magritte's `/setup/` and `/bin/` directories to the Python path.

# In[1]:


from sys import path
path.insert (0, '../../setup/')
path.insert (0, '../../bin/')


# Import Magritte's Python modules and setup.

# In[2]:


from magritte import Model, Long1, Long2, Double1, Double2, String1
from setup    import Setup, linedata_from_LAMDA_file


# ## 1) Read the input file

# In[3]:


#from ioMagritte import IoPython
from ioMagritte import IoText
from   magritte import Simulation 


# In[4]:


#modelName = 'models/model_0_1D_all_constant.hdf5'
modelName = 'models/model_2_2D_VanZadelhoff_1a/'


# Define an io object to handle input and output. (In this case via Python using HDF5.)

# In[5]:


#io = IoPython ("hdf5", modelName)
io = IoText (modelName)


# In[6]:


simulation = Simulation ()


# In[7]:


simulation.read (io)


# ## 3) Run the model
# ---

# ### Option 1: in the notebook
# ---

# In[9]:


import numpy as np

def get_r (p):
    x = simulation.geometry.cells.x[p]
    y = simulation.geometry.cells.y[p]
    return np.sqrt(x**2+y**2)


# In[10]:


simulation.parameters.set_max_iter (100)
simulation.parameters.set_pop_prec (1.0E-6)
simulation.parameters.r = 99999999999999999#0
simulation.parameters.o = 99999999999999999#0
simulation.parameters.f = 99999999999999999#int(simulation.parameters.nfreqs()/4)


# In[11]:


simulation.compute_spectral_discretisation ()


# In[12]:


simulation.compute_boundary_intensities ()


# In[13]:


simulation.compute_LTE_level_populations ()


# In[14]:


simulation.compute_level_populations (io)


# In[15]:


simulation.write (io)


# ### Option 2: in the shell
# ---

# In[8]:


# Number of processes and threads
NUMBER_OF_PROCS = 1
NUMBER_OF_THRDS = 1

# Flag for shared memory systems
FLAGS = '-env I_MPI_SHM_LMT shm'

# Path to Magritte executable
PATH_TO_EXECUTABLE = '../../bin/examples/example_2.exe'


# In[9]:


# Set number of threads
get_ipython().system(' export OMP_NUM_THREADS=$NUMBER_OF_THRDS')


# In[10]:


get_ipython().system(' mpirun -np $NUMBER_OF_PROCS $FLAGS $PATH_TO_EXECUTABLE $modelName')


# ## 4) Plot the output
# ---

# In[11]:


import numpy as np


# In[12]:


from bokeh.resources import INLINE
from bokeh.plotting  import figure, show, save, gridplot
from bokeh.palettes  import cividis
from bokeh.io        import output_notebook
output_notebook (INLINE)


# In[13]:


def color(s):
    ns = int((s_max-s_min) / s_step + 1)
    es = int((s    -s_min) / s_step)
    return cividis(ns)[es]

def legend(s):
    return f'{s}'


# Add Magritte's `/bin/` directories to the Python path.

# In[14]:


from sys import path
path.insert (0, f'{MagritteFolder}bin/')


# In[22]:


from   magritte import Simulation
#from ioMagritte import IoPython
from ioMagritte import IoText


# Define an io object to handle input and output. (In this case via Python using HDF5.)

# In[23]:


#io = IoPython ("hdf5", modelName)
io = IoText (modelName)


# In[24]:


simulation = Simulation ()


# In[25]:


simulation.read (io)


# In[26]:


plot = figure (title='Populations over iterations', width=700, height=500, x_axis_type='log', y_axis_type='log')


r = []
for p in range(simulation.parameters.ncells()):
    cells = simulation.geometry.cells
    r.append (np.sqrt(cells.x[p]**2 + cells.y[p]**2 + cells.z[p]**2))

spec_nr = simulation.lines.lineProducingSpecies[0].linedata.num
    
for it in range(0, 27, 1):
    (p0, p1) = np.loadtxt(f'{modelName}Lines/LineProducingSpecies_0/population_iteration_{it}.txt', unpack=True)
    y = [p1[p]/simulation.chemistry.species.abundance[p][spec_nr] for p in range(simulation.parameters.ncells())]
    plot.line(r, y, color=cividis(27)[it], legend=legend(it))

show(plot)


# In[20]:


simulation.rayPair.ndep


# In[21]:


plot = figure (title='u(r)', width=700, height=500, x_axis_type='log')

for i in range(simulation.rayPair.ndep):
    p = simulation.rayPair.nrs[i]
    r = np.sqrt(simulation.geometry.cells.x[p]**2 + simulation.geometry.cells.y[p]**2)
    x.append(r)

y = simulation.rayPair.Su[:simulation.rayPair.ndep]

plot.line(x, y, color='blue')

show(plot)


# In[22]:


# Store in convenient format

data = np.stack((x, y), axis=1)
np.savetxt ('u_r.txt', data)


# In[23]:


simulation.rayPair.frs[0]


# In[24]:


import numpy as np


# In[25]:


ncells = simulation.parameters.ncells()
nfreqs = simulation.parameters.nfreqs()


# #### Get the results for the benchmark paper.

# In[26]:


data = np.loadtxt('/home/frederik/Codes/Ratran_copy/FrederiksExapmle/output/vanZadelhoff_problem_1a.pop', skiprows=14)

pop = data[:,10]

x_min = data[:,1]
x_max = data[:,2]


# In[27]:


x, p1 = np.loadtxt ('vanZadelhoff_results/case1a_mean_zadelhoff.dat', unpack=True)
ox, op1 = np.loadtxt ('vanZadelhoff_results/case1a_mean_zadelhoff.dat', unpack=True)


# In[28]:


i,ra,rb,nh,tk,nm,vr,db,td,lp0,lp1 = np.loadtxt ('vanZadelhoff_results/origprob1.pop', skiprows=14, unpack=True)


# In[ ]:





# In[29]:


# Load Jeremy's data

(x_Jeremy, p_Jeremy, F_Jeremy) = np.loadtxt ('/home/frederik/Dropbox/Astro/Benchmarks/vanZadelhoff/1_iteration_Jeremy.txt', unpack=True)

# Derive mean intensity from Jeremy's populations

Aul = simulation.lines.lineProducingSpecies[0].linedata.A[0]
Bul = simulation.lines.lineProducingSpecies[0].linedata.Bs[0]
Blu = simulation.lines.lineProducingSpecies[0].linedata.Ba[0]
Clu = simulation.lines.lineProducingSpecies[0].linedata.colpar[0].Ce[0][0]
Cul = simulation.lines.lineProducingSpecies[0].linedata.colpar[0].Cd[0][0]

J_Jeremy = []

for nu in p_Jeremy:
    nl = 1.0 - nu
    J_Jeremy.append( (nu*(Aul+Cul)-nl*Clu) / (nl*Blu-nu*Bul) )
    
F_Jeremy = F_Jeremy * 1.0E-3
    
from scipy import interpolate

J_Jeremy_interpolated = interpolate.interp1d(x_Jeremy, J_Jeremy)


# In[30]:


# Find theoretical values after 1 iteration

import numpy as np
import tests

import scipy.integrate as integrate


linedata = simulation.lines.lineProducingSpecies[0].linedata

r_in   = 1.0E13   # [m]
r_out  = 7.8E16   # [m]

rho_in = 2.0E13   # [m^-3]
X_mol  = 1.0E-8   # [.]

T      = 20.0     # [K]

turb   = 150.0    # [m/s]

R = r_out


# Constants
c     = 2.99792458E+8    # [m/s] speed of light
kb    = 1.38064852E-23   # [J/K] Boltzmann's constant
mp    = 1.6726219E-27    # [kg] proton mass
T_CMB = 2.7254800        # [K] CMB temperature


line  = 0


pops       = tests.LTEpop         (linedata, T)
emissivity = tests.lineEmissivity (linedata, pops)
opacity    = tests.lineOpacity    (linedata, pops)
source     = tests.lineSource     (linedata, pops)

def bcd (nu):
    return tests.planck(T_CMB, nu)

S    =  source[line]
chi  = opacity[line]
nuij = linedata.frequency[line]
dnu  = nuij/c * 150.0 # np.sqrt(2.0*kb*T/mp + vturb**2)


def phi (nu):
    return 1 / (np.sqrt(np.pi) * dnu) * np.exp(-((nu-nuij)/dnu)**2)

def f(x, x_in, theta):
    xsintheta = x*np.sin(theta)
    term1 = np.arccos(xsintheta) + 0.5*np.pi - theta
    if ((xsintheta < x_in) and (theta < 0.5*np.pi)):
        term2 = 2.0 * np.arccos(xsintheta/x_in)
        return (term1 - term2) / xsintheta
    else:
        return term1 / xsintheta

def tau(nu, r, theta):
    return chi * X_mol * rho_in * r_in**2 / R * phi(nu) * f(r/R, r_in/R, theta)


def integrand(nu, r,theta):
    return np.exp(-tau(nu, r, theta))


import quadrature

def J(nu, r):
    B = bcd (nu)
    nquads = 11
    integral = 0.0
    for z in range(nquads):
        root   = quadrature.H_roots(nquads)[z]
        weight = quadrature.H_weights(nquads)[z]
        nu = nuij + dnu * root
        integral += weight * integrate.quad (lambda theta: np.exp(-tau(nu, r, theta)), 0, np.pi)[0]
    return S + (B-S) / np.pi * integral

# def J(nu, r):
#     B = bcd (nu)
#     return S + (B-S) / np.pi * integrate.quad (lambda theta: np.exp(-tau(nuij, r, theta)), 0, np.pi)[0]

base = 100

log_r_in  = np.log(r_in)  / np.log(base)
log_r_out = np.log(r_out) / np.log(base)

RR_theory = np.logspace (log_r_in, log_r_out, 100, base=base, endpoint=True)
JJ_theory = [J(nuij, r) for r in RR_theory]


# In[31]:


S


# In[32]:


c     = 2.99792458E+8    # [m/s] speed of light
kb    = 1.38064852E-23   # [J/K] Boltzmann's constant
mp    = 1.6726219E-27    # [kg] proton mass
T_CMB = 2.7254800        # [K] CMB temperature


# In[33]:


np.sqrt(2.0*kb*20.0/mp + (150.0)**2)


# In[34]:


# Convert [cm] to [m]
x = 1.0E-2 * x
ox = 1.0E-2 * ox


# #### Plot of the level populations

# In[35]:


plot = figure (title='Level populations', width=700, height=500, x_axis_type='log', y_axis_type='log')

r = []
for p in range(ncells):
    cells = simulation.geometry.cells
    r.append (np.sqrt(cells.x[p]**2 + cells.y[p]**2 + cells.z[p]**2))

spec_nr = simulation.lines.lineProducingSpecies[0].linedata.num

def index (p,i):
    return i + p * simulation.lines.lineProducingSpecies[0].linedata.nlev

y0 = [simulation.lines.lineProducingSpecies[0].population[index(s,0)]/simulation.chemistry.species.abundance[s][spec_nr] for s in range(ncells)]
y1 = [simulation.lines.lineProducingSpecies[0].population[index(s,1)]/simulation.chemistry.species.abundance[s][spec_nr] for s in range(ncells)]

# y1_p   = [simulation.lines.population_prev1[s][0][1]/simulation.chemistry.species.abundance[s][spec_nr] for s in range(ncells)]
# y1_pp  = [simulation.lines.population_prev2[s][0][1]/simulation.chemistry.species.abundance[s][spec_nr] for s in range(ncells)]
# y1_ppp = [simulation.lines.population_prev3[s][0][1]/simulation.chemistry.species.abundance[s][spec_nr] for s in range(ncells)]

#plot.circle (r, y0, color='red')

#plot.line (x_Jeremy, p_Jeremy, color='green', legend="Jeremy, 1 iteration")

plot.circle (r, y1, color='blue', legend="Magritte")


# plot.circle (r, y1_p,   color='orange')
# plot.circle (r, y1_pp,  color='brown')
# plot.circle (r, y1_ppp, color='red')

#plot.circle (x, p1, color='green')
#plot.circle (ox, op1, color='red')
plot.line (x_min, pop, color='orange')
plot.line (x_max, pop, color='orange')

plot.xaxis.axis_label = "radius [m]"
plot.yaxis.axis_label = "fractional level populations"
show(plot)
#save(plot, f'plot_nc_{simulation.parameters.ncells()}_nr_{simulation.parameters.nrays()}.html')


# In[36]:


plot = figure (title='error over the iterations', width=700, height=500, y_axis_type='log')

x1 = range(len(simulation.error_max))
y1 = [error for error in simulation.error_max]

x2 = range(len(simulation.error_mean))c
y2 = [error for error in simulation.error_mean]

plot.circle(x1, y1, color='blue',  legend='maximum')
plot.circle(x2, y2, color='black', legend='mean')

plot.xaxis.axis_label = "# iterations"
plot.yaxis.axis_label = "relative change in populations"

show(plot)


# In[70]:


def color(s):
    ns = int((s_max-s_min) / s_step + 1)
    es = int((s    -s_min) / s_step)
    return cividis(ns)[es]

def legend(s):
    return f'{s}'


# In[71]:


s_min  = 0
s_max  = simulation.parameters.ncells()
s_step = 20


# In[72]:


def rindex (p, f):
    return f + p * simulation.parameters.nfreqs_red()


# #### Plot of the spectrum

# In[73]:


plot = figure (title='Spectrum', width=700, height=500, y_axis_type='log')

for s in range(s_min, s_max, s_step):
    x = [simulation.radiation.frequencies.nu[s][f] for f in range(nfreqs)]
    y = [simulation.radiation.u[0][rindex(s, f)]   for f in range(nfreqs)]
    plot.line(x, y, color=color(s), legend=f'{s}')

plot.xaxis.axis_label = "frequencies [Hz]"
plot.yaxis.axis_label = "Mean intensity [W/m^2]"
show(plot)


# In[74]:


id, jbar, pops0, pops1 = np.loadtxt('/home/frederik/Codes/Ratran_copy/FrederiksExapmle/jbar.txt', unpack=True)


# In[75]:


RRR, JJJ = np.loadtxt ('../0_analytical_models/analytic_result.txt', unpack=True)


# In[76]:


unique_id = []
unique_jb = []
unique_po = []

for i in range(len(id)-1,-1,-1):
    if not id[i] in unique_id:
        unique_id.append(  id[i])
        unique_jb.append(jbar[i])
        unique_po.append(pops1[i])

        
# unique_id = []
# unique_jb = []
# unique_po = []
        
        
# for i in range(0,len(id),+1):
#     if not id[i] in unique_id:
#         unique_id.append(  id[i])
#         unique_jb.append(jbar[i])
#         unique_po.append(pops1[i])


# In[77]:


plot = figure (title='Intensity field after 1 iteration', width=700, height=500, x_axis_type='log', y_axis_type='log')

x  = [np.sqrt(simulation.geometry.cells.x[p]**2 + simulation.geometry.cells.y[p]**2) for p in range(simulation.parameters.ncells())]
y1 = [simulation.lines.lineProducingSpecies[0].Jlin[p][0]                            for p in range(simulation.parameters.ncells())]
y2 = [simulation.lines.lineProducingSpecies[0].Jeff[p][0]                            for p in range(simulation.parameters.ncells())]

#plot.circle(x, y1, color='black', legend='J')
plot.circle(x, y2, color='red',   legend='Magritte\'s mean intensity')

# x_min = data[:,1]
# x_max = data[:,2]


# x_min = [data[int(i)-1,1] for i in unique_id]
# x_max = [data[int(i)-1,2] for i in unique_id]
# y = unique_jb

#plot.line (x_min, y, color='orange')
#plot.line (x_max, y, color='orange')

# x_min = [data[int(i)-1,1] for i in id]
# x_max = [data[int(i)-1,2] for i in id]
# y = jbar

#plot.circle (x_min, y, color='brown')
#plot.circle (x_max, y, color='brown')

plot.line (x_Jeremy,  J_Jeremy,  color='green', legend='derived from SMMOL pops')
plot.line (x_Jeremy,  F_Jeremy,  color='black', legend='Jeremy')
plot.line (RR_theory, JJ_theory, color='blue',  legend='analytical result')
plot.line (RRR, JJJ, color='yellow',  legend='A R')

plot.legend.location = "bottom_left"

plot.xaxis.axis_label = "radius [m]"
plot.yaxis.axis_label = "Mean intensity [W/m^2]"

show(plot)


# In[78]:


plot = figure (title='Intensity field after 1 iteration', width=700, height=500, x_axis_type='log', y_axis_type='log')

x  = [np.sqrt(simulation.geometry.cells.x[p]**2 + simulation.geometry.cells.y[p]**2) for p in range(simulation.parameters.ncells())]
y1 = [simulation.lines.lineProducingSpecies[0].Jlin[p][0]                            for p in range(simulation.parameters.ncells())]
y2 = [simulation.lines.lineProducingSpecies[0].Jeff[p][0]                            for p in range(simulation.parameters.ncells())]


XXX = x[70:]
YYY = [y2[p] / J_Jeremy_interpolated(x) for p,x in enumerate(XXX)]

plot.circle(XXX, YYY, color='red',   legend='Magritte\'s mean intensity')


plot.legend.location = "bottom_left"

plot.xaxis.axis_label = "radius [m]"
plot.yaxis.axis_label = "Mean intensity [W/m^2]"

show(plot)


# In[79]:


plot = figure (width=700, height=500, x_axis_type='log', y_axis_type='log')

# x  = [np.sqrt(simulation.geometry.cells.x[p]**2 + simulation.geometry.cells.y[p]**2) for p in range(simulation.parameters.ncells())]
# y1 = [simulation.lines.lineProducingSpecies[0].Jlin[p][0]                            for p in range(simulation.parameters.ncells())]
# y2 = [simulation.lines.lineProducingSpecies[0].Jeff[p][0]                            for p in range(simulation.parameters.ncells())]
# plot.circle(x, y1, color='black', legend='J')
# plot.circle(x, y2, color='red',   legend='Jeff')

x_min = data[:,1]
x_max = data[:,2]


x_min = [data[int(i)-1,1] for i in unique_id]
x_max = [data[int(i)-1,2] for i in unique_id]
y = unique_po

plot.line (x_min, y, color='orange')
plot.line (x_max, y, color='orange')

# x_min = [data[int(i)-1,1] for i in id]
# x_max = [data[int(i)-1,2] for i in id]
# y = jbar

plot.circle (x_min, y, color='brown')
plot.circle (x_max, y, color='brown')

show(plot)


# In[ ]:


unique_po


# In[45]:


simulation.parameters.ncells()


# To Do
# ---
# 
# * Check if angular intergral is computed correctly in analytic result.
# * Double check units.
# * Prepare document for Jeremy (and Ward + Leen?)

# In[130]:


np.mean(np.array(J_Jeremy) / np.array(F_Jeremy))


# In[ ]:




