#!/usr/bin/bash

conda deactivate
conda create --name petsc_cluster3 --file petsccluster_spec_file.txt
conda activate petsc_cluster3

pip install . --process-dependency-links
