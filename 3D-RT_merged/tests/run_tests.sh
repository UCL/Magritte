#! /bin/bash

icc -g -qopenmp test_main.o -o test_ray_tracing test_ray_tracing.cpp
icc -g -qopenmp test_main.o -o test_level_population_solver test_level_population_solver.cpp


echo "Log for test_ray_tracing" > test_ray_tracing.log
echo "++++++++++++++++++++++++" >> test_ray_tracing.log
echo "                        " >> test_ray_tracing.log

./test_ray_tracing >> test_ray_tracing.log



echo "Log for test_level_population_solver" > test_level_population_solver.log
echo "++++++++++++++++++++++++++++++++++++" >> test_level_population_solver.log
echo "                                    " >> test_level_population_solver.log

./test_ray_tracing >> test_level_population_solver.log
