#!/usr/bin/bash

conda deactivate
conda create --name petscenv3 --file conda_spec_file.txt
conda activate petscenv3

pip install . --process-dependency-links
