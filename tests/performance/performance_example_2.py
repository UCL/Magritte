#!/usr/bin/env python
# coding: utf-8

# # Example 2: CPU and GPU performance tests
# ---

# In[1]:


import matplotlib.pyplot as plt
import numpy             as np
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


executable_cpu = '/home/frederik/Dropbox/Astro/Magritte/bin/examples/example_2_cpu.exe'
executable_gpu = '/home/frederik/Dropbox/Astro/Magritte/bin/examples/example_2_gpu.exe'


# In[3]:


test_model_cpu = '/home/frederik/Magritte_all/Models/Benchmarks/1_van_Zadelhoff/model_problem_1a_mini_cpu/'
test_model_gpu = '/home/frederik/Magritte_all/Models/Benchmarks/1_van_Zadelhoff/model_problem_1a_mini_gpu/'


# In[4]:


test_model_cpu = '/home/frederik/Dropbox/GitHub/Magritte-code/Benchmarks/0_analytical_models/models/model_3_1D_density_distribution_test_cpu/'
test_model_gpu = '/home/frederik/Dropbox/GitHub/Magritte-code/Benchmarks/0_analytical_models/models/model_3_1D_density_distribution_test_gpu/'


# ---

# Create model files.

# In[48]:


get_ipython().run_line_magic('run', '/home/frederik/Dropbox/GitHub/Magritte-code/Benchmarks/1_van_Zadelhoff/models/model_1a_mini_cpu.ipynb')
get_ipython().run_line_magic('run', '/home/frederik/Dropbox/GitHub/Magritte-code/Benchmarks/1_van_Zadelhoff/models/model_1a_mini_gpu.ipynb')


# ---

# Quick **memory check** with `valgrind` and `cuda-memcheck`.

# In[33]:


get_ipython().system(' valgrind      $executable_cpu $test_model_cpu')


# In[49]:


get_ipython().system(' cuda-memcheck $executable_gpu $test_model_gpu')


# ---

# Run the models.

# In[23]:


timing_cpu = get_ipython().run_line_magic('timeit', '-o ! $executable_cpu $test_model_cpu')


# In[19]:


timing_gpu = get_ipython().run_line_magic('timeit', '-o ! $executable_gpu $test_model_gpu')


# In[5]:


result_J_cpu = np.loadtxt(f'{test_model_cpu}Radiation/J.txt')
result_J_gpu = np.loadtxt(f'{test_model_gpu}Radiation/J.txt')


# In[21]:


timing.average


# In[6]:


result_J_cpu


# In[7]:


result_J_gpu


# In[8]:


relative_difference = 2.0 * np.abs(result_J_cpu - result_J_gpu) / (result_J_cpu + result_J_gpu)


# In[15]:


plt.loglog(result_J_gpu)
plt.show()
plt.plot(result_J_gpu.transpose())
plt.show()


# In[16]:


plt.loglog(result_J_cpu)
plt.show()
plt.plot(result_J_cpu.transpose())
plt.show()


# In[10]:


plt.plot(relative_difference)
plt.show()
plt.loglog(relative_difference.transpose())
plt.show()


# In[13]:


result_J_gpu


# In[12]:


result_J_cpu


# In[14]:


result_J_gpu


# In[ ]:




