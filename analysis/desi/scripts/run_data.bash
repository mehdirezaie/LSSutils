#!/bin/bash
#SBATCH --job-name=cl
#SBATCH --account=PHS0336 
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mr095415@ohio.edu

# run with
# sbatch run_sv3.bash lrg "bmzls ndecals sdecals" 256
# or
# sbatch run_sv3.bash elg "bmzls ndecals sdecals" 1024


# manually add the path, later we will install the pipeline with `pip`
source ${HOME}/.bashrc

export PYTHONPATH=${HOME}/github/sysnetdev:${HOME}/github/LSSutils:${PYTHONPATH}
export NUMEXPR_MAX_THREADS=1
export OMP_NUM_THREADS=1
source activate sysnet

cd ${HOME}/github/LSSutils/analysis/desi/scripts/

do_prep=false     # 20 min x 1 tpn
do_lr=false       # 20 min x 1 tpn
do_fit=false       # 20 h x 1 tpn
do_rfe=false       
do_assign=false
do_nbar=false     # 10 min x 4 tpn
do_cl=true       # 20 min x 4 tpn
do_mcmc=false     # 10 h x 14 tpn

bsize=5000    # 
targets='lrg' # lrg
regions=$1    # bmzls, ndecals, sdecals
maps=$2       # known, all
tag_d=0.57.0  # 0.57.0 (sv3) or 1.0.0 (main)
nside=256     # lrg=256, elg=1024
method=""     # lin, nn, or noweight
model=dnn
loss=mse
nns=(4 20)
nepoch=150  # v0 with 71
nchain=20
etamin=0.001

root_dir=/fs/ess/PHS0336/data/rongpu/imaging_sys

prep=${HOME}/github/LSSutils/analysis/desi/scripts/prep_desi.py
nnfit=${HOME}/github/sysnetdev/scripts/app.py
cl=${HOME}/github/LSSutils/analysis/desi/scripts/run_cell_sv3.py
nbar=${HOME}/github/LSSutils/analysis/desi/scripts/run_nnbar_sv3.py
assign=${HOME}/github/LSSutils/scripts/analysis/desi/fetch_weights.py
mcmc=${HOME}/github/LSSutils/scripts/analysis/desi/run_mcmc_fast.py

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
    if [ $1 = "known" ]
    then
        axes=(0 1 2)   # EBV, Nstar, galdepth-r (pcc selected)
    elif [ $1 = "all" ]
    then
        axes=({0..12}) # all maps
        #axes=(0 1 2 3 4 5 6 7 10 11 12) # ELG's do not need 8 and 9, which are W1 and W1 bands
    fi
    echo ${axes[@]}
}



function get_reg(){
    if [ $1 = 'NBMZLS' ]
    then
        reg='bmzls'
    elif [ $1 = 'NDECALS' ]
    then
        reg='ndecals'
    elif [ $1 = 'SDECALS' ]
    then
        reg='sdecals'
    elif [ $1 = 'SDECALS_noDES' ]
    then
        reg='sdecals'
    elif [ $1 = 'DES_noLMC' ]
    then
        reg='sdecals'
    elif [ $1 = 'DES' ]
    then
        reg='sdecals'
    fi
    echo $reg
}

# e.g., sbatch run_data.bash "lrg elg" "bmzls ndecals sdecals" 0.57.0 256
if [ "${do_prep}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            echo ${target} ${region} ${nside} ${root_dir} $tag_d
            srun -n 1 python $prep $target $region $nside $root_dir $tag_d
        done
    done
fi


if [ "${do_lr}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            echo ${target} ${region}
            input_path=${root_dir}/tables/${tag_d}/n${target}_features_${region}_${nside}.fits
            output_path=${root_dir}/regression/${tag_d}/${model}_${target}_${region}_${nside}/hp/
            
            axes=$(get_axes ${target})           
            du -h $input_path
            echo $output_path $axes
            srun -n 1 python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model $model --loss $loss --nn_structure ${nns[@]} -fl
        done
    done
fi

if [ "${do_rfe}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            lr=$(get_lr ${target})
            axes=$(get_axes ${target})

            echo ${target} ${region} $lr ${axes[@]}

            input_path=${root_dir}/tables/${tag_d}/n${target}_features_${region}_${nside}.fits
            output_path=${root_dir}/rfe/${tag_d}/${model}_${target}_${region}_${nside}/
           
            echo $output_path 
            du -h $input_path
            
            python $nnfit -i ${input_path} -o ${output_path} -k --do_rfe --axes $axes 
            echo

        done
    done
fi

if [ "${do_fit}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            maps=$2
            lr=$(get_lr ${target})
            axes=$(get_axes ${maps})

            echo ${target} ${region} $lr ${axes[@]} $maps

            input_path=${root_dir}/tables/${tag_d}/n${target}_features_${region}_${nside}.fits
            output_path=${root_dir}/regression/${tag_d}/${model}_${target}_${region}_${nside}_${maps}/
           
            echo $output_path 
            du -h $input_path
            srun -n 1 python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model $model --loss $loss --nn_structure ${nns[@]} -lr $lr --eta_min $etamin -ne $nepoch -k -nc $nchain

            echo

        done
    done
fi


# bash run_sv3.bash 'LRG ELG BGS_ANY' 'NBMZLS NDECALS SDECALS SDECALS_noDES DES'
if [ "${do_assign}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            echo ${target} ${region} ${tag_d}
            input=/home/mehdi/data/dr9v0.57.0/sv3_v1/sv3target_${target}_${region}.fits
            output=${input}_MrWsys/wsys_${tag_d}.fits
            wreg=$(get_reg ${region})
            nnwsys=/home/mehdi/data/rongpu/imaging_sys/regression/v2/sv3nn_${target,,}_${wreg}_256/nn-weights.fits
            
            du -h $input $nnwsys
            echo ${output}
            python $assign ${input} ${nnwsys} ${output}
        done
    done
fi


if [ "${do_nbar}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            input_path=${root_dir}/tables/${tag_d}/n${target}_features_${region}_${nside}.fits
            output_path=${root_dir}/clustering/${tag_d}/nbar_${target}_${region}_${nside}_noweight.npy
            du -h $input_path
            
            if [ ! -f $output_path ] 
            then
                echo $output_path
                srun -n 4 python $nbar -d ${input_path} -o ${output_path}
            fi

            output_path=${root_dir}/clustering/${tag_d}/nbar_${target}_${region}_${nside}_nn_${maps}.npy
            selection=${root_dir}/regression/${tag_d}/${model}_${target}_${region}_${nside}_${maps}/nn-weights.fits
            du -h $selection
            echo $output_path
            srun -n 4 python $nbar -d ${input_path} -o ${output_path} -s ${selection}            
            
        done
    done
fi

if [ "${do_cl}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            input_path=${root_dir}/tables/${tag_d}/n${target}_features_${region}_${nside}.fits
            output_path=${root_dir}/clustering/${tag_d}/cl_${target}_${region}_${nside}_noweight.npy
            du -h $input_path
            
            if [ ! -f $output_path ]
            then
                echo $output_path
                srun -n 4 python $cl -d ${input_path} -o ${output_path}
            fi

            output_path=${root_dir}/clustering/${tag_d}/cl_${target}_${region}_${nside}_nn_${maps}.npy
            selection=${root_dir}/regression/${tag_d}/${model}_${target}_${region}_${nside}_${maps}/nn-weights.fits
            du -h $selection
            echo $output_path
            srun -n 4 python $cl -d ${input_path} -o ${output_path} -s ${selection}            
        done
    done
fi


if [ "${do_mcmc}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do       
            output_mcmc=${root_dir}/clustering/${tag_d}/mcmc_${target}_${region}_${method}.npy
            echo $target $region $method $output_mcmc
            python $mcmc $region $method $output_mcmc
        done
    done
fi
