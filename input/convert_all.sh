#! /bin/bash

for filename in files/1D*.dat
do
  python input_convertor.py $filename
done
