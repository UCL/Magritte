#!/bin/bash


echo "Run the tests"
echo "_____________"
echo "             "

echo "Unit tests:"

cd unit_tests

echo "  - Test ray-tracing"
cd ray_tracing
cmake . && make
cd t1_1D
./test_ray_tracing.exe
cd ../..

echo "  - Test line data"
cd line_data
cmake . && make
cd t1
./test_line_data.exe
cd ../..

# echo "  - Test column density calculator"
# cd ../calc_column_density
# cmake . && make
# ./test_calc_column_density.exe

cd ..
