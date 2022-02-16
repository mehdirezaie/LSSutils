#!/bin/bash
#SBATCH --job-name=linclx
#SBATCH --account=PHS0336 
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mr095415@ohio.edu

# manually add the path, later we will install the pipeline with `pip`
source ${HOME}/.bashrc

export PYTHONPATH=${HOME}/github/sysnetdev:${HOME}/github/LSSutils:${PYTHONPATH}
export NUMEXPR_MAX_THREADS=1
export OMP_NUM_THREADS=1

source activate sysnet

cd ${HOME}/github/LSSutils/analysis/lssxcmb/scripts/


do_linfit=false    # 10 h x 14
do_nnfit=false     # 10 m lr finder, 120 h fit 
do_linsamp=false    # 1 h x 1
do_nnsamp=false    # 3h x 10tpn
do_nnpull=false    # 1 h
do_lincell=false   # 5hx14tpn
do_nncell=false    # 5hx14tpn
do_cl=false
do_clx=true        # 20min x 4tpn


target=elg
region=$1   # options are bmzls, ndecals, sdecals
nside=1024
version=v3
printf -v mockid "%d" $SLURM_ARRAY_TASK_ID
#mockid=$2
echo $mockid

# nn parameters
#axes=({0..12})
axes=(0 1 2 3 4 5 6 7 10 11 12)    # ELG's do not need 8 and 9, which are W1 and W1 bands
nchain=5
nepoch=200
nns=(4 20)
bsize=4098
lr=0.008
model=dnnp
loss=pnll
etamin=0.00001


linfit=${HOME}/github/LSSutils/analysis/lssxcmb/scripts/run_wlinear_mcmc_p.py
nnfit=${HOME}/github/sysnetdev/scripts/app.py
nnfite=${HOME}/github/sysnetdev/scripts/appensemble.py
nnsamp=${HOME}/github/LSSutils/analysis/lssxcmb/scripts/pull_sysnet_snapshot_mpidr9.py
nnpull=${HOME}/github/LSSutils/analysis/lssxcmb/scripts/combine_nn_windows.py
linsamp=${HOME}/github/LSSutils/analysis/lssxcmb/scripts/sample_linear_windows.py
cl=${HOME}/github/LSSutils/analysis/lssxcmb/scripts/run_cell_sv3.py

root_dir=/fs/ess/PHS0336/data/rongpu/imaging_sys


if [ "${do_linfit}" = true ]
then
    input_path=${root_dir}/tables/${version}/n${target}_features_${region}_${nside}.fits
    #output_path=/fs/ess/PHS0336/data/tanveer/dr9/${version}/${target}_linear/mcmc_${region}_${nside}.npz
    output_path=/fs/ess/PHS0336/data/tanveer/dr9/${version}/${target}_linearp/mcmc_${region}_${nside}.npz
    du -h $input_path
    echo $output_path
    python $linfit -d $input_path -o $output_path -ax ${axes[@]}
fi


if [ "${do_nnfit}" = true ]
then
    input_path=${root_dir}/tables/${version}/n${target}_features_${region}_${nside}.fits
    output_path=/fs/ess/PHS0336/data/tanveer/dr9/${version}/${target}_dnnp/${region}_${nside}/
    du -h $input_path
    echo $output_path
    # find lr
    #srun -n 1 python $nnfite -i ${input_path} -o ${output_path}hp/ -ax ${axes[@]} -bs ${bsize} \
    #                  --model $model --loss $loss --nn_structure ${nns[@]} -fl
    srun -n 1 python $nnfite -i ${input_path} -o ${output_path} -ax ${axes[@]} -bs ${bsize} \
                      --model $model --loss $loss --nn_structure ${nns[@]} -lr $lr --eta_min $etamin -ne $nepoch -nc $nchain \
                      --snapshot_ensemble -k --no_eval  
fi

if [ "${do_linsamp}" = true ]
then
    # will do all regions at once
    srun -n 1 python $linsamp
fi

if [ "${do_nnsamp}" = true ]
then
    srun -n 10 python $nnsamp $region
fi


if [ "${do_nnpull}" = true ]
then
    srun -n 1 python $nnpull
fi


if [ "${do_nncell}" = true ]
then
    input_path=/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/v3/nelg_features_${region}_1024.fits
    output_path=/fs/ess/PHS0336/data/tanveer/dr9/v3/elg_dnnp/windows/cell_${region}.npy
    wind_dir=/fs/ess/PHS0336/data/tanveer/dr9/v3/elg_dnnp/windows/

    du -h $input_path 
    echo $output_path
    #ls $wind_dir
    srun -n 14 python run_cell_windows.py -d $input_path  -m $wind_dir -o $output_path
fi

if [ "${do_lincell}" = true ] 
then
    input_path=/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/v3/nelg_features_${region}_1024.fits
    output_path=/fs/ess/PHS0336/data/tanveer/dr9/v3/elg_linear/windows/cell_${region}.npy
    wind_dir=/fs/ess/PHS0336/data/tanveer/dr9/v3/elg_linear/windows/

    srun -n 14 python run_cell_windows.py -d $input_path  -m $wind_dir -o $output_path
fi


if [ "${do_clx}" = true ]
then
    input_path=${root_dir}/tables/v3/n${target}_features_${region}_${nside}.fits
    #output_path=/fs/ess/PHS0336/data/tanveer/dr9/${version}/clustering/cl_${target}_${region}_${nside}_noweight.npy
    du -h $input_path
    #echo $output_path
    #mpirun -np 4 python $cl -d ${input_path} -o ${output_path}

    output_path=/fs/ess/PHS0336/data/tanveer/dr9/${version}/clustering/cl_${target}_${region}_${nside}_${mockid}_linp.npy
    selection=/fs/ess/PHS0336/data/tanveer/dr9/v3/elg_linearp/windows_clean/linwindow_${mockid}.hp1024.fits
    echo $output_path
    du -h $selection
    mpirun -np 4 python $cl -d ${input_path} -o ${output_path} -s ${selection}            
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
