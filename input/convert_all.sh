#! /bin/bash

for filename in 1D*.dat
do
  python input_convertor.py $filename
done
