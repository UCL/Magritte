# ![Magritte logo](https://raw.githubusercontent.com/UCL/Magritte/master/docs/Images/Magritte_logo.png?token=AWbw4U46loMQBSOlQVJh5Hy6DvN_vLicks5Z4TI7wA%3D%3D)   Magritte   [![Build Status](https://travis-ci.com/UCL/Magritte.svg?token=j3NNTbFLxGaJNsSoKgCz&branch=master)](https://travis-ci.com/UCL/Magritte)

## a Multidimensional Accelerated General-purpose RadIaTive TransfEr code


How to get started:
-------------------

- Run the cofigure.sh script to produce the Makefile.
  ```
  $ ./configure.sh
  ```


How to use:
-----------

- Specify the parameters in the parameters.txt file.
  (Data files are stored in the /data folder.
   Grids are stored in the /input folder.)

- Run make to setup and compile the source code given the specified parameters.
  This will create the Magritte executable.
  ```
  $ make
  ```

- Run the executable to generate the ouput in the /output folder.
  ```
  $ ./Magritte.exe
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
