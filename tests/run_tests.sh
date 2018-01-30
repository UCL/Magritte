#!/bin/bash


echo "Run the tests"
echo "_____________"
echo "             "

echo "Unit tests:"

echo "  - Test ray-tracing"

cd unit_tests/ray_tracing

make

./test_ray_tracing.exe
