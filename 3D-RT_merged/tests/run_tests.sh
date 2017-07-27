#! /bin/bash



echo "Log for test_ray_tracing" > test_ray_tracing.log
echo "++++++++++++++++++++++++" >> test_ray_tracing.log
echo "                        " >> test_ray_tracing.log

cd ../input

python grid_1D_regular.py 250

cd ..

./setup >> tests/test_ray_tracing.log

cd tests

icc -g -qopenmp test_main.o -o test_ray_tracing test_ray_tracing.cpp

./test_ray_tracing >> test_ray_tracing.log



echo "Log for test_level_population_solver" > test_level_population_solver.log
echo "++++++++++++++++++++++++++++++++++++" >> test_level_population_solver.log
echo "                                    " >> test_level_population_solver.log

icc -g -qopenmp test_main.o -o test_level_population_solver test_level_population_solver.cpp

./test_ray_tracing >> test_level_population_solver.log



echo "Log for test_exact_feautrier" > test_exact_feautrier.log
echo "++++++++++++++++++++++++++++" >> test_exact_feautrier.log
echo "                            " >> test_exact_feautrier.log

cd ../input

python grid_1D_regular.py 101

cd ..

./setup >> tests/test_exact_feautrier.log

cd tests

icc -g -qopenmp test_main.o -o test_exact_feautrier test_exact_feautrier.cpp

./test_exact_feautrier >> test_exact_feautrier.log
