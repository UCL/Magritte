#! /bin/bash

for filename in ../abundances_*
do
    tag=${filename#"../abundances_"}
    tag=${tag%".txt"}
    python abundances.py $tag
done
