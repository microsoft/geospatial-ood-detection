#!/bin/bash
black .
isort .
pydocstyle .
nbqa isort *.ipynb  
autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive ./
flake8 .