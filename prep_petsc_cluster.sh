#!/usr/bin/bash

conda deactivate
conda create --name petsc_cluster --file petsccluster_spec_file.txt
conda activate petsc_cluster

pip install . --process-dependency-links
