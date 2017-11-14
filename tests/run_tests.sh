#!/bin/bash



echo "Log for test_ray_tracing" > test_logs/test_ray_tracing.log
echo "++++++++++++++++++++++++" >> test_logs/test_ray_tracing.log
echo "                        " >> test_logs/test_ray_tracing.log

cd ../input

python grid_1D_regular.py 250

cd ../tests

./test_ray_tracing >> test_logs/test_ray_tracing.log



echo "Log for test_level_population_solver" > test_logs/test_level_population_solver.log
echo "++++++++++++++++++++++++++++++++++++" >> test_logs/test_level_population_solver.log
echo "                                    " >> test_logs/test_level_population_solver.log

./test_ray_tracing >> test_logs/test_level_population_solver.log



echo "Log for test_feautrier" > test_logs/test_feautrier.log
echo "++++++++++++++++++++++++++++" >> test_logs/test_feautrier.log
echo "                            " >> test_logs/test_feautrier.log

cd ../input

python grid_1D_regular.py 101

cd ../tests

./test_feautrier >> test_logs/test_feautrier.log





#   CHEMISTRY
#--------------------------------------------------------------------------------------------------



echo "Log for test_spline" > test_logs/test_spline.log
echo "+++++++++++++++++++" >> test_logs/test_spline.log
echo "                   " >> test_logs/test_spline.log

./test_spline >> test_logs/test_spline.log


cd test_output
python test_spline_plot.py
cd ..


echo "Log for test_calc_reac_rates_rad" > test_logs/test_calc_reac_rates_rad.log
echo "+++++++++++++++++++++++++++++++++++++++" >> test_logs/test_calc_reac_rates_rad.log
echo "                                       " >> test_logs/test_calc_reac_rates_rad.log

cd test_output
python test_calc_reac_rates_rad_plot.py
cd ..

