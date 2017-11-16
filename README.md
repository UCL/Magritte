# <IMG SRC="/docs/Images/Magritte_logo.png" width="70" vertical-align:middle STYLE="vertical-align:middle"> Magritte   [![Build Status](https://travis-ci.com/UCL/Magritte.svg?token=j3NNTbFLxGaJNsSoKgCz&branch=master)](https://travis-ci.com/UCL/Magritte)

#### Multidimensional Accelerated General-purpose Radiative Transfer


How to get started:
-------------------

- Run the cofigure.sh bash script to produce the Makefile.
  ```
  $ bash configure.sh
  ```


How to use:
-----------

- Specify the parameters in the parameters.hpp file.
  (Data files are stored in the /data folder.
   Grids are stored in the /input folder.)

- Run make to setup and compile the source code given the specified parameters.
  This will create the Magritte executable.
  ```
  $ make
  ```

- Run the executable to generate the ouput in the /output/files/YY-MM-DD_hh:mm_output folder,
  where first part of the last folder idicates the date and time when the makefile was executed.
  ```
  $ ./Magritte.exe
  ```

- The output can be visualized using the python scripts in the output/plot_scripts folder.
  The results of these plot scripts will be saved in a seperate file /plots in the output folder.






Current issues and warnings
---------------------------

(!) Look out for devisions by 4PI which should be devisions by the number of rays.

(!) 1D models assume NRAYS=12.

(!) Rounding errors in the angle calculation in ray tracing (temporary fix).

(!) The first and last species are "dummies".





---

Developed by [Frederik De Ceuster](https://github.com/FrederikDeCeuster) at [UCL](https://github.com/ucl) and [KU Leuven](https://github.com/IvS-KULeuven)
