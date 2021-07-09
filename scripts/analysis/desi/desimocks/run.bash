#!/bin/bash

#
# Mehdi Rezaie, mr095415@ohio.edu
# 
#
# activate environment


eval "$(/home/mehdi/miniconda3/bin/conda shell.bash hook)"
conda activate py3p6

#
#   CODES


# DD(s,mu)

# run step 2
survey2pc=/home/mehdi/github/LSSutils/scripts/analysis/survey2pc.py

# run step 3
survey2pc2=/home/mehdi/github/LSSutils/scripts/analysis/survey2pc_step3.py

# 
# real red
for space in red
do
    echo ${space}
    mpirun -np 16 python $survey2pc2 $space 0.7 1.0
    mpirun -np 16 python $survey2pc2 $space 1.0 1.3 
    mpirun -np 16 python $survey2pc2 $space 1.3 1.7 
done
