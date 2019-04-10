#! /bin/bash


# Run all tests in this directory

# Run all executables
for test_program in test_*.py; do
  python "$test_program" || break
done
