#!/bin/bash
#SBATCH --job-name=nn
#SBATCH --account=PHS0336 
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mr095415@ohio.edu

# run with
# manually add the path, later we will install the pipeline with `pip`
source ${HOME}/.bashrc

export PYTHONPATH=${HOME}/github/sysnetdev:${HOME}/github/LSSutils:${PYTHONPATH}
export NUMEXPR_MAX_THREADS=1
export OMP_NUM_THREADS=1
source activate sysnet

cd ${HOME}/github/LSSutils/analysis/desi/scripts/

ell=$1
echo $ell
python get_window_mat.py $ell
