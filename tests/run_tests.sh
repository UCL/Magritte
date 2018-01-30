#!/bin/bash


echo "Run the tests"
echo "_____________"
echo "             "

echo "Unit tests:"


echo "  - Test ray-tracing"
cd unit_tests/ray_tracing
cmake . && make
./test_ray_tracing.exe

echo "  - Test column density calculator"
cd ../calc_column_density
cmake . && make
./test_calc_column_density.exe
