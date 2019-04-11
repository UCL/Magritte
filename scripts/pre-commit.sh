#! /bin/bash

echo "Running pre-commit hook"

# Convert ipynb to executable scripts
bash scripts/convert_all_ipynb_to_py.sh


# Use when tests are included

## $? stores exit value of the last command
# if [ $? -ne 0 ]; then
#  echo "Tests must pass before commit!"
#  exit 1
# fi
