#! /bin/bash

icc -g -qopenmp test_main.o -o test_spline test_spline.cpp

echo "Log for test_spline" > test_spline.log
echo "+++++++++++++++++++" >> test_spline.log
echo "                   " >> test_spline.log

./test_spline >> test_spline.log

python test_spline_plot.py




icc test_main.o -o test_rate_calculations_radfield test_rate_calculations_radfield.cpp 

echo "Log for test_rate_calculations_radfield" > test_rate_calculations_radfield.log
echo "+++++++++++++++++++++++++++++++++++++++" >> test_rate_calculations_radfield.log
echo "                                       " >> test_rate_calculations_radfield.log

python test_rate_calculations_radfield_plot.py
