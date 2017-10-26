#$ /bin/bash

# This script plots all standard output files

python plot_abundances.py $1 $2

python plot_lines.py $1 $2

python plot_temperatures.py $1 $2


# TO DO if $2 is not specified automatically put "final"
