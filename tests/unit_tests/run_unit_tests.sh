#! /bin/bash


# Run all tests in the ../../bin/tests/ directory

# Go to the directory
cd ../../bin/tests

# Make a directory for the test data
mkdir testdata

# Run all executables
for test_program in test_*.exe; do
    ./"$test_program" || break
done

# Remove the test data
rm -rf testdata
