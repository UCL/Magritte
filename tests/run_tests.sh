#! /bin/bash


# Run all tests

cd unit
bash run_unit_tests.sh
cd ..

cd integration
bash run_integration_tests.sh
cd ..
