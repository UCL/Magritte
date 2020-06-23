#!/usr/bin/env python
# coding: utf-8

# # 0) Analytical Model: 1D all constant
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


from magritte.setup import Setup, linedata_from_LAMDA_file


# ## 1) Read the input file

# In[3]:


#from ioMagritte import IoPython
from ioMagritte import IoText
from   magritte import Simulation 


# In[4]:


#modelName = 'models/model_1_1D_velocity_gradient.hdf5'
modelName = 'models/model_1_1D_velocity_gradient/'


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

# Set additional run parameters

# In[8]:


simulation.parameters.set_pop_prec (1.0E-8)


# In[9]:


simulation.compute_spectral_discretisation ()


# In[10]:


simulation.compute_boundary_intensities ()


# In[11]:


simulation.compute_LTE_level_populations ()


# In[12]:


simulation.compute_radiation_field ()


# ## 4) Check the output
# ---

# In[13]:


from bokeh.plotting import figure, show, gridplot
from bokeh.palettes import cividis
from bokeh.io       import output_notebook
output_notebook()


# In[14]:


line = 0


# Define helper quantities for the model.

# In[15]:


ncells = 50


# In[16]:


dens = 1.0E+12   # [m^-3]
abun = 1.0E+06   # [m^-3]
temp = 2.5E+02   # [K]
turb = 2.5E+02   # [m/s]
dx   = 1.0E+04   # [m]
dv   = 2.0E+01   # [m/s]


# In[17]:


def color(s):
    ns = int((s_max-s_min) / s_step + 1)
    es = int((s    -s_min) / s_step)
    return cividis(ns)[es]

def legend(s):
    return f'{s}'


# In[18]:


s_min  = 0
s_max  = ncells
s_step = 4


# In[19]:


def rindex (p, f):
    return f + p * simulation.parameters.nfreqs_red()


# #### **Analytical model**

# Assuming a constant source function $S_{\nu}(x)=S_{\nu}$ along the ray and boundary condition $B_{\nu}$ on both sides of the ray, the mean intensity J and 1D flux G are given by
# 
# \begin{align}
#     J_{\nu}(\tau(x)) \ &= \ S_{\nu} \ + \ \frac{1}{2} \ \left(B_{\nu}-S_{\nu}\right) \ \left[e^{-\tau_{\nu}(x)} + e^{-\tau_{\nu}(L-x)}\right], \\
#     G_{\nu}(\tau(x)) \ &= \ \color{white}S_{\color{white}\nu} \ - \ \frac{1}{2} \ \left(B_{\nu}-S_{\nu}\right) \ \left[e^{-\tau_{\nu}(x)} - e^{-\tau_{\nu}(L-x)}\right],
# \end{align}
# 
# where the optical depth $\tau_{\nu}$ is given by
# 
# \begin{equation}
#     \tau_{\nu}(\ell) \ = \ \int_{0}^{\ell} \text{d} l \ \chi_{\nu}(l) .
# \end{equation}
# 
# The frequency dependence of the opacity only comes from the line profile
# 
# \begin{equation}
#     \chi_{\nu}(x) \ = \ \chi_{ij} \phi_{\nu},
# \end{equation}
# 
# where we assume a Gaussian profile
# 
# \begin{equation}
# 	\phi_{\nu}^{ij}(x) \ = \ \frac{1}{\sqrt{\pi} \ \delta\nu_{ij}} \ \exp \left[-\left(\frac{\nu-\nu_{ij}} {\delta\nu_{ij}(x)}\right)^{2}\right], \hspace{5mm} \text{where} \hspace{5mm} \delta\nu_{ij}(x) \ = \ \frac{\nu_{ij}}{c} \sqrt{ \frac{2 k_{b} T(x)}{m_{\text{spec}}} \ + \ v_{\text{turb}}^{2}(x)}.
# \end{equation}
# 
# To account for the velocity gradient, after a slab of length $\ell$, the frequency shifts as
# 
# \begin{equation}
#     \nu \ \rightarrow \ \left( 1 - \frac{v_{\max} \ell}{c L} \right) \nu.
# \end{equation}
# 
# Solving the integral for the optical depth then yields
# 
# \begin{equation}
#   \tau_{\nu}(\ell) \ = \ \frac{\chi L}{\nu} \ \frac{c}{v_{\max}}  \ \frac{1}{2} \left\{ \text{Erf}\left[\frac{\nu-\nu_{ij}}{\delta\nu_{ij}}\right] \ + \ \text{Erf}\left[\frac{v_{\max}}{c} \frac{\nu}{\delta\nu_{ij}}\frac{\ell}{L} - \frac{\nu-\nu_{ij}}{\delta\nu_{ij}}\right] \right\} .
# \end{equation}

# In[20]:


import numpy as np
import tests

from scipy.special import erf


linedata = simulation.lines.lineProducingSpecies[0].linedata

c     = 2.99792458E+8    # [m/s] speed of light
kb    = 1.38064852E-23   # [J/K] Boltzmann's constant
amu   = 1.66053904E-27   # [kg] atomic mass unit
T_CMB = 2.7254800        # [K] CMB temperature

inverse_mass = linedata.inverse_mass

pops       = tests.LTEpop         (linedata, temp) * abun
emissivity = tests.lineEmissivity (linedata, pops)
opacity    = tests.lineOpacity    (linedata, pops)
source     = tests.lineSource     (linedata, pops)

def bcd (nu):
    return tests.planck(T_CMB, nu)

S    =  source[line]
chi  = opacity[line]
L    = dx * (ncells-1)
vmax = dv * (ncells-1)
nuij = linedata.frequency[line]
dnu  = nuij / c * np.sqrt(2.0*kb*temp*inverse_mass + turb**2)


def phi (nu):
    return 1 / (np.sqrt(np.pi) * dnu) * np.exp(-((nu-nuij)/dnu)**2)

def tau(nu, l):
    arg = (nu - nuij) / dnu
    fct = vmax/c * nu/dnu
    return chi*L / (fct*dnu) * 0.5 * (erf(arg) + erf(fct*l/L-arg))
    
def J (nu, x):
    tau1 = tau(nu, x)
    tau2 = tau(nu, L-x)
    B = bcd (nu)
    return S + 0.5 * (B-S) * (np.exp(-tau1) + np.exp(-tau2))

def G (nu, x):
    tau1 = tau(nu, x)
    tau2 = tau(nu, L-x)
    B = bcd (nu)
    return   - 0.5 * (B-S) * (np.exp(-tau1) - np.exp(-tau2))


# In[21]:


def relativeError (a,b):
    a = np.array(a)
    b = np.array(b)
    return 2.0 * np.abs((a-b)/(a+b))


# In[22]:


nr_center =  simulation.parameters.nquads() // 2
n_wings   = (simulation.parameters.nquads() - 1) // 2       


# #### Compare Magritte against analytic model

# In[23]:


plot_model = figure(title='u analytic and numeric', width=400, height=400, y_axis_type="log")
plot_error = figure(title='Error',                  width=400, height=400, y_axis_type="log")

for s in range(s_min, s_max, s_step):
    M = int(simulation.lines.lineProducingSpecies[0].nr_line[s][line][nr_center] - n_wings    )
    N = int(simulation.lines.lineProducingSpecies[0].nr_line[s][line][nr_center] + n_wings + 1)
    # model
    x1 = nuij + 18 * dnu * np.linspace(-1,1,500)
    y1 = [J(x, simulation.geometry.cells.x[s]) for x in x1]
    plot_model.line(x1, y1, color=color(s), legend=legend(s))
    # data
    x2 = [simulation.radiation.frequencies.nu[s][f] for f in range(M,N)]
    y2 = [simulation.radiation.u[0][rindex(s, f)]   for f in range(M,N)]
    plot_model.circle(x2, y2, color=color(s), legend=legend(s))
    mo = [J(x, simulation.geometry.cells.x[s]) for x in x2]
    er = relativeError (mo, y2)
    plot_error.circle(x2, er, color=color(s), legend=legend(s))

plot = gridplot([[plot_model, plot_error]])

show(plot)


# In[24]:


plot_model = figure(title='v analytic and numeric', width=400, height=400, y_axis_type="log")
plot_error = figure(title='Error',                  width=400, height=400, y_axis_type="log")

for s in range(s_min, s_max, s_step):
    M = int(simulation.lines.lineProducingSpecies[0].nr_line[s][line][20] - 18    )
    N = int(simulation.lines.lineProducingSpecies[0].nr_line[s][line][20] + 18 + 1)
    # model
    x1 = nuij + 18 * dnu * np.linspace(-1,1,500)
    y1 = [G(x, simulation.geometry.cells.x[s]) for x in x1]
    plot_model.line(x1, y1, color=color(s), legend=legend(s))
    # data
    x2 = [simulation.radiation.frequencies.nu[s][f] for f in range(M,N)]
    y2 = [simulation.radiation.v[0][rindex(s, f)]   for f in range(M,N)]
    plot_model.circle(x2, y2, color=color(s), legend=legend(s))
    mo = [G(x, simulation.geometry.cells.x[s]) for x in x2]
    er = relativeError (mo, y2)
    plot_error.circle(x2, er, color=color(s), legend=legend(s))

plot = gridplot([[plot_model, plot_error]])

show(plot)


# In[ ]:




