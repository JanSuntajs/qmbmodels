#!/usr/bin/bash

conda deactivate
conda create --name petscenv --file ./conda_spec_files/conda_spec_file_with_correct_h5py.txt
source activate petscenv

pip install . --process-dependency-links
