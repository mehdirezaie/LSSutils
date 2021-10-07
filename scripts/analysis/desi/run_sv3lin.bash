#!/bin/bash
#SBATCH --job-name=dr9lin
#SBATCH --account=PHS0336 
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mr095415@ohio.edu

# manually add the path, later we will install the pipeline with `pip`
source ${HOME}/.bashrc

export PYTHONPATH=${HOME}/github/sysnetdev:${HOME}/github/LSSutils:${PYTHONPATH}
export NUMEXPR_MAX_THREADS=2
source activate sysnet

cd ${HOME}/github/LSSutils/scripts/analysis/desi

do_prep=false
do_lr=false
do_fit=false
do_assign=false
do_nbar=false
do_cl=true

cversion=v1
mversion=v2
nside=256
bsize=4098 # v1 500
targets='lrg' #'QSO'
regions=$1 #BMZLS
axes=({0..12})
model=lin
loss=mse
nns=(4 20)
nepoch=150 # 150
nchain=20 # 20
etamin=0.001

#root_dir=/home/mehdi/data/dr9v0.57.0/sv3nn_${cversion}
root_dir=/fs/ess/PHS0336/data/rongpu/imaging_sys

prep=${HOME}/github/LSSutils/scripts/analysis/desi/prep_desi.py
nnfit=${HOME}/github/sysnetdev/scripts/app.py
cl=${HOME}/github/LSSutils/scripts/analysis/desi/run_cell_sv3.py
nbar=${HOME}/github/LSSutils/scripts/analysis/desi/run_nnbar_sv3.py
assign=${HOME}/github/LSSutils/scripts/analysis/desi/fetch_weights.py

function get_lr(){
    if [ $1 = "lrg" ]
    then
        lr=0.6
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



if [ "${do_lr}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            echo ${target} ${region}
            input_path=${root_dir}/tables/n${target}_features_${region}_${nside}.fits
            output_path=${root_dir}/regression/${mversion}/sv3nn_${target}_${region}_${nside}_lin/hp/
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
            input_path=${root_dir}/tables/n${target}_features_${region}_${nside}.fits
            output_path=${root_dir}/regression/${mversion}/sv3nn_${target}_${region}_${nside}_lin/
            du -h $input_path
            srun -n 1 python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model $model --loss $loss --nn_structure ${nns[@]} -lr $lr --eta_min $etamin -ne $nepoch -k -nc $nchain 
        done
    done
fi


if [ "${do_nbar}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            input_path=${root_dir}/tables/n${target}_features_${region}_${nside}.fits
            output_path=${root_dir}/clustering/${mversion}/nbar_${target}_${region}_${nside}_noweight.npy
            #srun -n 4 python $nbar -d ${input_path} -o ${output_path}
            
            output_path=${root_dir}/clustering/${mversion}/nbar_${target}_${region}_${nside}_lin.npy
            selection=${root_dir}/regression/${mversion}/sv3nn_${target}_${region}_${nside}_lin/nn-weights.fits
            du -h $selection $input_path
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
            input_path=${root_dir}/tables/n${target}_features_${region}_${nside}.fits
            output_path=${root_dir}/clustering/${mversion}/cl_${target}_${region}_${nside}_noweight.npy
            du -h $input_path
	    #srun -n 4 python $cl -d ${input_path} -o ${output_path}
            
            output_path=${root_dir}/clustering/${mversion}/cl_${target}_${region}_${nside}_lin.npy
            selection=${root_dir}/regression/${mversion}/sv3nn_${target}_${region}_${nside}_lin/nn-weights.fits
            echo $output_path
	    du -h $selection
	    srun -n 4 python $cl -d ${input_path} -o ${output_path} -s ${selection}            
        done
    done
fi
