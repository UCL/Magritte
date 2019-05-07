#!/usr/bin/env python
# coding: utf-8

# # Magritte Performance audit
# ---

# In[1]:


import os, inspect
thisFolder     = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
magritteFolder = f'{thisFolder}/../../'


# Instrument Magritte using Score-P. This requires recompiling (rebuilding). First make sure the build directory is clean.

# In[2]:


get_ipython().system(' bash ../../build.sh clean')


# Now Magritte can be build in `performance_audit` mode.

# In[3]:


SCOREP_FOLDER = f'{magritteFolder}/dependencies/scorep/installed/bin/'


# In[4]:


get_ipython().system(' mkdir build')
get_ipython().system(' cd build')


# In[5]:


get_ipython().system(' SCOREP_WRAPPER=off                                  cmake                                                 -DPERF_ANALYSIS=ON                                  -DCMAKE_C_COMPILER=$SCOREP_FOLDER/scorep-gcc        -DCMAKE_CXX_COMPILER=$SCOREP_FOLDER/scorep-g++      -DOMP_PARALLEL=OFF                                  -DMPI_PARALLEL=OFF                                  -DGRID_SIMD=OFF                                     $magritteFolder')


# In[ ]:


get_ipython().system(' make')


# In[3]:


get_ipython().system(' bash ../../build.sh performance_audit')


# Magritte example executable

# In[15]:


EXECUTABLE = "../../bin/tests/performance/examples/example_2.exe"


# In[18]:


get_ipython().system(' ldd $EXECUTABLE')


# In[8]:


MODEL_FILE = f'{curdir}/models/model_2_2D_VanZadelhoff_1a/'

# Number of processes and threads
N_PROCS = 1
N_THRDS = 1

# Flag for shared memory systems
FLAGS = "-env I_MPI_SHM_LMT shm"

# Path to Magritte executable
EXECUTABLE = "../../bin/tests/performance/examples/example_2.exe"


# Set number of threads

# In[12]:


get_ipython().system(' export OMP_NUM_THREADS=$N_THRDS')


# In[13]:


get_ipython().system(' echo $OMP_NUM_THREADS')


# In[14]:


get_ipython().system(' mpirun -np $N_PROCS $FLAGS $EXECUTABLE $MODEL')


# In[ ]:




