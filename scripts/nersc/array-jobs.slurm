#!/bin/bash
#SBATCH -q debug
#SBATCH -o job_array_test%j.out
#SBATCH -n 1
#SBATCH --time 00:02:00
#SBATCH -C haswell
#SBATCH --mail-type=BEGIN,ENd,FAIl
#SBATCH --mail-user=gm240915@ohio.edu
# load packages
source /global/common/software/m3035/conda-activate.sh 3.7
printf -v mockid "%04d" $SLURM_ARRAY_TASK_ID
export pks=/global/project/projectdirs/eboss/czhao/EZmock/QSO_v5/clustering/PK/
export input=${pks}PK_EZmock_eBOSS_QSO_NGC_v5_z0.8z2.2_${mockid}.dat
export output=${SCRATCH}/baofits/ezmocks/baofit_${mockid}.dat

# run
srun -n 1 python jobarray_test.py --input $input --output $output
