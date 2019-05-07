#!/usr/bin/env python
# coding: utf-8

# # Magritte Performance audit
# 
# Instrument Magritte using Score-P. This requires recompiling (rebuilding). First make sure the build directory is clean.

# In[1]:


get_ipython().system(' bash ../../build.sh clean')


# Now Magritte can be build in `performance_audit` mode.

# In[3]:


get_ipython().system(' bash ../../build.sh performance_audit')


# In[5]:


MODEL_FILE = 

# Number of processes and threads
NUMBER_OF_PROCS = 1
NUMBER_OF_THRDS = 1

# Flag for shared memory systems
FLAGS = "-env I_MPI_SHM_LMT shm"

# Path to Magritte executable
PATH_TO_EXECUTABLE = "../../bin/tests/performance/examples/example_2.exe"


# Set number of threads

# In[4]:


get_ipython().system(' export OMP_NUM_THREADS=$NUMBER_OF_THRDS')


# In[ ]:





# In[ ]:


get_ipython().system(' mpirun -np $NUMBER_OF_PROCS $FLAGS $PATH_TO_EXECUTABLE $MODEL_FILE')

