#! /bin/bash

# Convert all ipynb (Ipython or Jupyter notebooks) to executable python scripts.

# Extract all IPython/Jupyter notebooks
IPYNB_FILES=$(find . -type f -name "*.ipynb")

for IPYNB_FILE in $IPYNB_FILES; do
  # If it is not a checkpoint file, convert to a python script
  if [[ $IPYNB_FILE != *".ipynb_checkpoints"* ]]; then
      jupyter nbconvert --to script "$IPYNB_FILE" || break
  fi
done
