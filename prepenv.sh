#!/usr/bin/bash

conda deactivate
conda create --name petscenv --file conda_spec_file.txt
source activate petscenv

pip install . --process-dependency-links