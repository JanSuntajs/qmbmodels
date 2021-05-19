#!/usr/bin/bash

conda deactivate
conda create --name petscenv --file conda_spec_file_with_correct_h5py.txt
conda activate petscenv

pip install . --process-dependency-links
