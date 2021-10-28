#!/bin/bash
#SBATCH --job-name=dr9nn$1
#SBATCH --account=PHS0336 
#SBATCH --time=75:00:00
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


do_linfit=false   # 10 h
do_nnfit=true    # 10 m lr finder, 75 h fit 
do_cl=false

target=elg
region=$1   # options are bmzls, ndecals, sdecals
nside=1024
version=v3

# nn parameters
#axes=({0..12})
axes=(0 1 2 3 4 5 6 7 10 11 12)    # ELG's do not need 8 and 9, which are W1 and W1 bands
nchain=5
nepoch=200
nns=(4 20)
bsize=4098
lr=0.3
model=dnn
loss=mse
etamin=0.0001


linfit=${HOME}/github/LSSutils/scripts/analysis/lssxcmb/run_wlinear_mcmc.py
nnfit=${HOME}/github/sysnetdev/scripts/app.py
nnfite=${HOME}/github/sysnetdev/scripts/appensemble.py
pullens=${HOME}/github/LSSutils/scripts/analysis/lssxcmb/pull_sysnet_snapshot_mpidr9.py
cl=${HOME}/github/LSSutils/scripts/analysis/lssxcmb/run_cell_sv3.py

root_dir=/fs/ess/PHS0336/data/rongpu/imaging_sys


if [ "${do_linfit}" = true ]
then
    input_path=${root_dir}/tables/${version}/n${target}_features_${region}_${nside}.fits
    output_path=/fs/ess/PHS0336/data/tanveer/dr9/${version}/${target}_linear/mcmc_${region}_${nside}.npz
    du -h $input_path
    echo $output_path
    srun -n 1 python $linfit $input_path $output_path
fi


if [ "${do_nnfit}" = true ]
then
    input_path=${root_dir}/tables/${version}/n${target}_features_${region}_${nside}.fits
    output_path=/fs/ess/PHS0336/data/tanveer/dr9/${version}/${target}_dnn/${region}_${nside}/
    du -h $input_path
    echo $output_path
    
    # find lr
    #srun -n 1 python $nnfit -i ${input_path} -o ${output_path}hp/ -ax ${axes[@]} -bs ${bsize} \
    #                  --model $model --loss $loss --nn_structure ${nns[@]} -fl

    srun -n 1 python $nnfite -i ${input_path} -o ${output_path} -ax ${axes[@]} -bs ${bsize} \
                      --model $model --loss $loss --nn_structure ${nns[@]} -lr $lr --eta_min $etamin -ne $nepoch -nc $nchain \
                      --snapshot_ensemble -k --no_eval  
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
