#!/bin/bash
#SBATCH --job-name=dr9mcmc
#SBATCH --account=PHS0336 
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --mail-type=FAIL
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

cd ${HOME}/github/LSSutils/scripts/analysis/desi

do_prep=false     # 20 min x 1 tpn
do_lr=false       # 20 min x 1 tpn
do_fit=false      # 24 h x 1 tpn
do_assign=false
do_nbar=false
do_cl=false       # 20 min x 4 tpn
do_mcmc=false      # 10 h x 14 tpn

cversion=v1
mversion=v3
nside=256   # lrg=256, elg=1024
bsize=4098  # v1 500
targets=lrg #'QSO'
regions=$1  # BMZLS
method=$2   # lin, nn, or noweight
axes=({0..12})
model=dnn
loss=mse
nns=(4 20)
nepoch=150 # v0 with 71
nchain=20
etamin=0.001

#root_dir=/home/mehdi/data/dr9v0.57.0/sv3nn_${cversion}
root_dir=/fs/ess/PHS0336/data/rongpu/imaging_sys

prep=${HOME}/github/LSSutils/scripts/analysis/desi/prep_desi.py
nnfit=${HOME}/github/sysnetdev/scripts/app.py
cl=${HOME}/github/LSSutils/scripts/analysis/desi/run_cell_sv3.py
nbar=${HOME}/github/LSSutils/scripts/analysis/desi/run_nnbar_sv3.py
assign=${HOME}/github/LSSutils/scripts/analysis/desi/fetch_weights.py
mcmc=${HOME}/github/LSSutils/scripts/analysis/desi/run_mcmc_fast.py

function get_lr(){
    if [ $1 = "lrg" ]
    then
        lr=0.2
    elif [ $1 = "elg" ]
    then
        lr=0.3
    elif [ $1 = "qso" ]
    then
        lr=0.2
    elif [ $1 = 'bgs_any' ]
    then
        lr=0.3
    elif [ $1 = 'bgs_bright' ]
    then
        lr=0.3
    fi
    echo $lr
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



if [ "${do_prep}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            echo ${target} ${region} ${nside} ${root_dir} $mversion
            srun -n 1 python $prep $target $region $nside $root_dir $mversion
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
            input_path=${root_dir}/tables/${mversion}/n${target}_features_${region}_${nside}.fits
            output_path=${root_dir}/regression/${mversion}/sv3nn_${target}_${region}_${nside}/hp/
            
	    du -h $input_path
            echo $output_path
	    srun -n 1 python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model $model --loss $loss --nn_structure ${nns[@]} -fl
        done
    done
fi


if [ "${do_fit}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            lr=$(get_lr ${target})
            echo ${target} ${region} $lr
            input_path=${root_dir}/tables/${mversion}/n${target}_features_${region}_${nside}.fits
            #output_path=${root_dir}/regression/${mversion}/sv3nn_${target}_${region}_${nside}/
            output_path=${root_dir}/regression/${mversion}/sv3nn_${target}_${region}_${nside}wfpix/
            du -h $input_path
            srun -n 1 python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model $model --loss $loss --nn_structure ${nns[@]} -lr $lr --eta_min $etamin -ne $nepoch -k -nc $nchain 
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
            echo ${target} ${region} ${mversion}
            input=/home/mehdi/data/dr9v0.57.0/sv3_v1/sv3target_${target}_${region}.fits
            output=${input}_MrWsys/wsys_${mversion}.fits
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
            input_path=${root_dir}/tables/${mversion}/n${target}_features_${region}_${nside}.fits
            output_path=${root_dir}/clustering/${mversion}/nbar_${target}_${region}_${nside}_noweight.npy
            srun -n 4 python $nbar -d ${input_path} -o ${output_path}
            
            output_path=${root_dir}/clustering/${mversion}/nbar_${target}_${region}_${nside}_nn.npy
            selection=${root_dir}/regression/${mversion}/sv3nn_${target}_${region}_${nside}/nn-weights.fits
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
            input_path=${root_dir}/tables/${mversion}/n${target}_features_${region}_${nside}.fits
            output_path=${root_dir}/clustering/${mversion}/cl_${target}_${region}_${nside}_noweight.npy
            srun -n 4 python $cl -d ${input_path} -o ${output_path}
            
            output_path=${root_dir}/clustering/${mversion}/cl_${target}_${region}_${nside}_nn.npy
            selection=${root_dir}/regression/${mversion}/sv3nn_${target}_${region}_${nside}/nn-weights.fits
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
            output_mcmc=${root_dir}/clustering/${mversion}/mcmc_${target}_${region}_${method}.npy
            echo $target $region $method $output_mcmc
            python $mcmc $region $method $output_mcmc
        done
    done
fi