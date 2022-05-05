#!/bin/bash
#SBATCH --job-name=nncont
#SBATCH --account=????
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=????

## run with
# sbatch --array=1-1000 run_mocks.bash


source ${HOME}/.bashrc
source activate sysnet

export PYTHONPATH=${HOME}/github/sysnetdev:${PYTHONPATH}
export NUMEXPR_MAX_THREADS=1
export OMP_NUM_THREADS=1

do_nn=true     # 20 h


printf -v mockid "%d" $SLURM_ARRAY_TASK_ID
echo ${mockid}
bsize=4098
region="bmzls" # bmzls, ndecals, sdecals
iscont=1
maps="known6"
target="lrg"
fnltag="zero" #zero, po100
ver=v2 # 
root_dir="/fs/ess/PHS0336/data/lognormal/${ver}"           # path to the lognormal mocks
root_dir2="/fs/ess/PHS0336/data/rongpu/imaging_sys/tables" # path to the hpix, features, ... tabulated data
model=dnn
loss=mse
nns=(4 20)
nepoch=70 # or 150
nchain=20
etamin=0.001


nnfit=${HOME}/github/sysnetdev/scripts/app.py

function get_lr(){
    if [ $1 = "lrg" ]
    then
        lr=0.1
    elif [ $1 = "elg" ]
    then
        lr=0.2
    fi
    echo $lr
}


function get_axes(){
    if [ $1 = "known1" ]
    then
        axes=(0)   # ebv
    elif [ $1 = "known2" ]
    then
        axes=(0 11)   # ebv, psfsize-g 
    elif [ $1 = "known3" ]
    then
        axes=(0 3 11) # ebv, gdepth-g, psfs-g
    elif [ $1 = "known4" ]
    then
        axes=(0 3) # ebv, gdepth-g
    elif [ $1 = "known5" ]
    then
        axes=(0 1 3) # ebv, nstar, gdepth-g
    elif [ $1 = "known6" ]
    then
        axes=(0 1 3 11) # ebv, nstar, gdepth-g, psf-g

    elif [ $1 = "all" ]
    then
        axes=({0..12}) # all maps
        #axes=(0 1 2 3 4 5 6 7 10 11 12) # ELG's do not need 8 and 9, which are W1 and W1 bands
    fi
    echo ${axes[@]}
}


if [ "${do_nn}" = true ]
then
    lr=$(get_lr ${target})
    axes=$(get_axes ${maps})

    echo ${target} ${region} $lr ${axes[@]} $maps

    input_path=${root_dir2}/0.57.0/n${target}_features_${region}_${nside}.fits
    input_map=${root_dir}/hpmaps/${target}hp-${fnltag}-${mockid}-f1z1-contaminated.fits
    output_path=${root_dir}/regression/fnl_${fnltag}/cont/${mockid}/${region}/nn_${maps}/

    du -h $input_map $input_path
    echo $output_path	
    
    # --- to test the learning rate finder. it's ok to run this on single mock
    #python $nnfit -i ${input_path} ${input_map} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model $model --loss $loss --nn_structure ${nns[@]}  -fl
    
    # --- after finding the learning rate, uncomment this line and run
    srun -n 1 python $nnfit -i ${input_path} ${input_map} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model $model --loss $loss --nn_structure ${nns[@]}  -lr $lr --eta_min $etamin -ne $nepoch -k -nc $nchain --do_tar -k
fi
