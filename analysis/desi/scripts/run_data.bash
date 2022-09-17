#!/bin/bash
#SBATCH --job-name=mcmc
#SBATCH --account=PHS0336 
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=14
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

do_prep=false     # 20 min x 1 tpn
do_lr=false       # 20 min x 1 tpn
do_fit=false       # linmcmc:20m x 14, nn:20 h x 1 tpn
do_linsam=false   # 10 min x 1
do_rfe=false       
do_assign=false
do_nbar=false     # 10 min x 4 tpn
do_cl=false       # 20 min x 4 tpn
do_mcmc=true     # 3 h x 14 tpn
do_mcmc_joint=false # 3x14
do_mcmc_joint3=false # 5x14

bsize=5000    # 
target='lrg'  # lrg
region="desi"     # bmzls, ndecals, sdecals
maps=$1 # known, all
tag_d=0.57.0  # 0.57.0 (sv3) or 1.0.0 (main)
nside=256     # lrg=256, elg=1024
method=$1     # lin, nn, or noweight
fnltag="zero"
model=linp
loss=pnll
nns=(4 20)
nepoch=70  # v0 with 71
nchain=20
etamin=0.001

root_dir=/fs/ess/PHS0336/data/rongpu/imaging_sys
mock_dir=/fs/ess/PHS0336/data/lognormal/v3

prep=${HOME}/github/LSSutils/analysis/desi/scripts/prep_desi.py
nnfit=${HOME}/github/sysnetdev/scripts/app.py
linfit=${HOME}/github/LSSutils/analysis/desi/scripts/run_wlinear_mcmc_p.py
linsam=${HOME}/github/LSSutils/analysis/desi/scripts/sample_linear_windows.py
cl=${HOME}/github/LSSutils/analysis/desi/scripts/run_cell_sv3.py
nbar=${HOME}/github/LSSutils/analysis/desi/scripts/run_nnbar_sv3.py
assign=${HOME}/github/LSSutils/scripts/analysis/desi/fetch_weights.py
mcmc=${HOME}/github/LSSutils/analysis/desi/scripts/run_mcmc_fast.py
mcmc_joint=${HOME}/github/LSSutils/analysis/desi/scripts/run_mcmc_joint.py
mcmc_joint3=${HOME}/github/LSSutils/analysis/desi/scripts/run_mcmc_joint3.py

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
        axes=(0 4)   # EBV
    elif [ $1 = "known1" ]
    then
        axes=(0 4 7)
    elif [ $1 = "all" ]
    then
        axes=(0 2 3 4 5 6 7 8) # all maps
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
        echo ${target} ${region} ${nside} ${root_dir} $tag_d
        srun -n 1 python $prep $target $region $nside $root_dir $tag_d
fi


if [ "${do_lr}" = true ]
then
        echo ${target} ${region}
        input_path=${root_dir}/tables/${tag_d}/n${target}_features_${region}_${nside}.fits
        output_path=${root_dir}/regression/${tag_d}/${model}_${target}_${region}_${nside}/hp/
        
        axes=$(get_axes ${target})           
        du -h $input_path
        echo $output_path $axes
        srun -n 1 python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model $model --loss $loss --nn_structure ${nns[@]} -fl
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
        lr=$(get_lr ${target})
        axes=$(get_axes ${maps})

        echo ${target} ${region} $lr ${axes[@]} $maps

        input_path=${root_dir}/tables/${tag_d}/n${target}_features_${region}_${nside}.fits
        output_path=${root_dir}/regression/${tag_d}/${model}_${target}_${region}_${nside}_${maps}/mcmc_${region}_${maps}.npz
       
        echo $output_path 
        du -h $input_path
        #srun -n 1 python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model $model --loss $loss --nn_structure ${nns[@]} -lr $lr --eta_min $etamin -ne $nepoch -k -nc $nchain
        python $linfit -d $input_path -o $output_path -ax ${axes[@]}
fi


if [ "${do_linsam}" = true ]
then
    axes=$(get_axes ${maps})
    output_path=${root_dir}/regression/${tag_d}/${model}_${target}_${maps}.hp${nside}.fits
    echo $output_path
    python $linsam -o $output_path -m $maps -ax $axes
fi

if [ "${do_nbar}" = true ]
then
        input_path=${root_dir}/tables/${tag_d}/n${target}_features_${region}_${nside}.fits
        output_path=${root_dir}/clustering/${tag_d}/nbar_${target}_${region}_${nside}_noweight.npy
        du -h $input_path
        
        if [ ! -f $output_path ] 
        then
            echo $output_path
            srun -n 1 python $nbar -d ${input_path} -o ${output_path}
        fi

        output_path=${root_dir}/clustering/${tag_d}/nbar_${target}_${region}_${nside}_nn_${maps}.npy
        selection=${root_dir}/regression/${tag_d}/${model}_${target}_${region}_${nside}_${maps}/nn-weights.fits
        du -h $selection
        echo $output_path
        srun -n 1 python $nbar -d ${input_path} -o ${output_path} -s ${selection}            
fi

if [ "${do_cl}" = true ]
then
    input_path=${root_dir}/tables/${tag_d}/n${target}_features_${region}_${nside}.fits
    output_path=${root_dir}/clustering/${tag_d}/cl_${target}_${region}_${nside}_noweight.npy
    du -h $input_path
    
    if [ ! -f $output_path ]
    then
        echo $output_path
        srun -n 4 python $cl -d ${input_path} -o ${output_path}
    fi

    output_path=${root_dir}/clustering/${tag_d}/cl_${target}_${region}_${nside}_${model}_${maps}.npy
    selection=${root_dir}/regression/${tag_d}/${model}_${target}_${maps}.hp${nside}.fits
    du -h $selection
    echo $output_path
    srun -n 4 python $cl -d ${input_path} -o ${output_path} -s ${selection}           
fi


if [ "${do_mcmc}" = true ]
then
    path_cl=${root_dir}/clustering/${tag_d}/cl_${target}_${region}_${nside}_${method}.npy
    path_cov=${mock_dir}/clustering/clmock_0_${target}_${fnltag}_${region}_256_noweight_cov.npz
    output_mcmc=${root_dir}/mcmc/${tag_d}/mcmc_${target}_${fnltag}_${region}_${maps}_steps10k_walkers50.npz
        
    du -h $path_cl $path_cov
    echo $target $region $maps $output_mcmc
    python $mcmc $path_cl $path_cov $region $output_mcmc 1.0
fi

if [ "${do_mcmc_joint}" = true ]
then
    region=$1
    region1=$2
    fnltag=zero
    maps=$3
    path_cl=${root_dir}/clustering/${tag_d}/clgg_lrg_${region}_256_${maps}.npz
    path_cov=${mock_dir}/clustering/clmock_${fnltag}_${region}_cov.npz
    path_cl1=${root_dir}/clustering/${tag_d}/clgg_lrg_${region1}_256_${maps}.npz
    path_cov1=${mock_dir}/clustering/clmock_${fnltag}_${region1}_cov.npz

    regionj=${region}${region1}
    output_mcmc=${root_dir}/mcmc/${tag_d}/mcmc_${target}_${fnltag}_${regionj}_${maps}_steps10k_walkers50.npz
        
    du -h $path_cl $path_cov
    du -h $path_cl1 $path_cov1
    echo $target $region $reion1 $maps $output_mcmc
    
    python $mcmc_joint $path_cl $path_cl1 $path_cov $path_cov1 $region $region1 $output_mcmc
fi

if [ "${do_mcmc_joint3}" = true ]
then
    region=$1
    region1=$2
    region2=$3
    fnltag=zero
    maps=$4

    path_cl=${root_dir}/clustering/${tag_d}/clgg_lrg_${region}_256_${maps}.npz
    path_cov=${mock_dir}/clustering/clmock_${fnltag}_${region}_cov.npz
    path_cl1=${root_dir}/clustering/${tag_d}/clgg_lrg_${region1}_256_${maps}.npz
    path_cov1=${mock_dir}/clustering/clmock_${fnltag}_${region1}_cov.npz
    path_cl2=${root_dir}/clustering/${tag_d}/clgg_lrg_${region2}_256_${maps}.npz
    path_cov2=${mock_dir}/clustering/clmock_${fnltag}_${region2}_cov.npz

    regionj=${region}${region1}${region2}
    output_mcmc=${root_dir}/mcmc/${tag_d}/mcmc_${target}_${fnltag}_${regionj}_${maps}_steps10k_walkers50.npz
        
    du -h $path_cl $path_cov
    du -h $path_cl1 $path_cov1
    du -h $path_cl2 $path_cov2
    echo $target $region $reion1 $region2 $maps $output_mcmc
    
    python $mcmc_joint3 $path_cl $path_cl1 $path_cl2 $path_cov $path_cov1 $path_cov2 $region $region1 $region2 $output_mcmc
fi


