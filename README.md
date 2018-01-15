<img src="/docs/Images/Magritte_logo.png" height="130">

[![Build Status](https://travis-ci.com/UCL/Magritte.svg?token=j3NNTbFLxGaJNsSoKgCz&branch=master)](https://travis-ci.com/UCL/Magritte)

<img src="/docs/Images/Magritte_name.png" height="30">

---

## How to get started:

- Run the `cofigure.sh` bash script to produce the Makefile.
  ```
  $ bash configure.sh
  ```


## How to use:

- Specify the parameters in the `parameters.hpp` file.
  (Data files are stored in the `/data` folder.
   Grids are stored in the `/input` folder.)

- Run make to setup and compile the source code given the specified parameters.
  This will create the Magritte executable.
  ```
  $ make
  ```

- Run the executable to generate the ouput in the `/output/files/YY-MM-DD_hh:mm_output` folder,
  where first part of the last folder idicates the date and time when the makefile was executed.
  ```
  $ ./Magritte.exe
  ```

- The output can be visualized in multiple depending in the input format:
  - For .txt input files one can use the python scripts in the `output/plot_scripts` folder.
    The results of these plot scripts will be saved in a seperate file `/plots` in the output folder.
  - For .vtu input files one can use the same visualization tool that was used for the input.
    The results are just appended to the input file and stored in the output folder.



## A first parallelization (OpenMP)

In a first parallelization round we simply paralellize every loop over all cells.



---



Developed by [Frederik De Ceuster](https://github.com/FrederikDeCeuster) at [UCL](https://github.com/ucl) and [KU Leuven](https://github.com/IvS-KULeuven)
