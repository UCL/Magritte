# functions C++ modules for python
----------------------------------

Since we will use the module in the `magritte` conda environment, create makefile like this:
```
cmake -DPYTHON_EXECUTABLE:FILEPATH=/home/frederik/software/anaconda3/envs/magritte/bin/python
```
Afterwards use the usual
```
make
```
to compile the module.
