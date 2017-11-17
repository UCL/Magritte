#! /bin/bash

# Run all comparissons

python compare_abundances.py $1 final

python compare_pops.py $1 C final
python compare_pops.py $1 C+ final
python compare_pops.py $1 O final
python compare_pops.py $1 CO final

python compare.py $1 temperature_gas final
