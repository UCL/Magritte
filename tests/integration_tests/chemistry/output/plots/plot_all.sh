#! /bin/bash

for filename in ../files/abundances_*
do
    tag=${filename#"../files/abundances_"}
    tag=${tag%".txt"}
    python plot_abun.py $tag
done
