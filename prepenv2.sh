#!/usr/bin/bash

conda deactivate
conda create --name petscenv2 --file conda_spec_file.txt
conda activate petscenv2

pip install . --process-dependency-links
