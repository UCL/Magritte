# ![Magritte logo](https://raw.githubusercontent.com/UCL/Magritte/master/docs/Images/Magritte_logo.png?token=AWbw4ahxpYY-sLWuJbKZ95Mhqo9xW3pYks5Z64V6wA%3D%3D)   Magritte   [![Build Status](https://travis-ci.com/UCL/Magritte.svg?token=j3NNTbFLxGaJNsSoKgCz&branch=master)](https://travis-ci.com/UCL/Magritte)

## Multidimensional Accelerated General-purpose RadIaTive TransfEr


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

Developed by @FrederikDeCeuster at @UCL and @IvS-KULeuven
