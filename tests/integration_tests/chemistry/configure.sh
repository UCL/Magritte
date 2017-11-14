#! /bin/bash



# Make sure Magritte is configured

cd ../../../
./configure.sh
cd tests/integration_tests/chemistry


# Create the Makefile

cmake .
