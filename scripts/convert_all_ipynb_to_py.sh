#! /bin/bash

# Convert all ipynb (Ipython or Jupyter notebooks) to executable python scripts.

IPYNB_FILES=$(find ../ -type f -name "*.ipynb")

for IPYNB_FILE in $IPYNB_FILES; do
    jupyter nbconvert --to script "$IPYNB_FILE" || break
done
