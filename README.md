# ![Magritte logo](https://raw.githubusercontent.com/UCL/3D-RT/master/docs/Images/Magritte_logo.png?token=AWbw4XMo9p-hKCF2_2jhw1pGbRlDOA32ks5Z4TFWwA%3D%3D) Magritte [![Build Status](https://travis-ci.com/UCL/3D-RT.svg?token=j3NNTbFLxGaJNsSoKgCz&branch=master)](https://travis-ci.com/UCL/3D-RT)

## a multidimensional accelerated general-purpose radiative transfer code


How to get started:

- Run the cofigure.sh script to produce the Makefile.
  ```
  $ ./configure.sh
  ```


How to use:

- Specify the parameters in the parameters.txt file.
  (Data files are stored in the /data folder.
   Grids are stored in the /input folder.)

- Run make to setup and compile the source code given the specified parameters.
  This will create the 3D-RT executable.
  ```
  $ make
  ```

- Run the executable to generate the ouput in the /output folder.
  ```
  $ ./3D-RT.exe
  ```

- The output can be visualized using the python scripts in the output/plots folder.



Current issues and todo
-----------------------

(!) Look out for devisions by 4PI which should be devisions by the number of rays.

(!) The cosmic ray variables ZETA and OMEGA are defined as standard code to be 3.85 (from 3D-PDR).

(!) 1D models assume NRAYS=12.

(!) Rounding errors in the angle calculation in ray tracing (temporary fix).

(!) Note factors of PC in distances.

(!) The first and last species are "dummies".

todo: Write one proper Python script that does the whole setup.
