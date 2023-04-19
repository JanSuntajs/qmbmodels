#!/usr/bin/bash

conda deactivate
conda create --name petscenv_stripped --file ./conda_spec_files/conda_spec_file_petscenv_stripped.txt
conda activate petscenv_stripped

pip install .  --upgrade --force-reinstall  #--process-dependency-links
