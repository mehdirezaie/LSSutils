#!/bin/bash
#SBATCH --job-name=dr9lin
#SBATCH --account=PHS0336 
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mr095415@ohio.edu

# manually add the path, later we will install the pipeline with `pip`
source ${HOME}/.bashrc

export PYTHONPATH=${HOME}/github/sysnetdev:${HOME}/github/LSSutils:${PYTHONPATH}
export NUMEXPR_MAX_THREADS=2
source activate sysnet

cd ${HOME}/github/LSSutils/scripts/analysis/lssxcmb/


do_linfit=false
do_wlinfit=true
do_cl=false

target=elg
region=bmzls
nside=1024
mversion=v2

linfit=${HOME}/github/LSSutils/scripts/analysis/lssxcmb/run_linear_mcmc.py
wlinfit=${HOME}/github/LSSutils/scripts/analysis/lssxcmb/run_wlinear_mcmc.py
cl=${HOME}/github/LSSutils/scripts/analysis/lssxcmb/run_cell_sv3.py

root_dir=/fs/ess/PHS0336/data/rongpu/imaging_sys


if [ "${do_linfit}" = true ]
then
    input_path=${root_dir}/tables/n${target}_features_${region}_${nside}.fits
    output_path=/fs/ess/PHS0336/data/tanveer/dr9/elg_linear/mcmc_${region}_${nside}.npz
    du -h $input_path
    echo $output_path
    srun -n 1 python $linfit $input_path $output_path
fi

if [ "${do_wlinfit}" = true ]
then
    input_path=${root_dir}/tables/n${target}_features_${region}_${nside}.fits
    #output_path=/fs/ess/PHS0336/data/tanveer/dr9/elg_linear/mcmc_${region}_${nside}wfrac.npz
    output_path=/fs/ess/PHS0336/data/tanveer/dr9/elg_linear/mcmc_${region}_${nside}wfracsq.npz
    du -h $input_path
    echo $output_path
    srun -n 1 python $wlinfit $input_path $output_path
fi


if [ "${do_cl}" = true ]
then
    input_path=${root_dir}/tables/n${target}_features_${region}_${nside}.fits
    output_path=${root_dir}/clustering/${mversion}/cl_${target}_${region}_${nside}_noweight.npy
    du -h $input_path
    echo $output_path
    mpirun -np 4 python $cl -d ${input_path} -o ${output_path}
    
    output_path=${root_dir}/clustering/${mversion}/cl_${target}_${region}_${nside}_nn.npy
    selection=/home/mehdi/data/tanveer/dr9/elg_mse_snapshots/${region}/windows_mean.hp1024.fits
    echo $output_path
    du -h $selection
    mpirun -np 4 python $cl -d ${input_path} -o ${output_path} -s ${selection}            
fi
