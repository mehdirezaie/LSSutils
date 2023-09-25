#!/bin/bash
#SBATCH --job-name=nn
#SBATCH --account=PHS0336 
#SBATCH --time=05:00:00
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
do_rfe=false      # 
do_assign=false   #
do_nbar=false     # 10 min x 4 tpn
do_cl=false       # 20 min x 4 tpn
do_mcmc=true     # 3 h x 14 tpn
do_mcmc_joint3=false # 5x14

bsize=5000    # 
target="lrg"  # lrg
region=$1     # bmzls, ndecalsc, sdecalsc, or desic
maps=$2       # known, all, known1, known2
tag_d=0.57.0  # 0.57.0 (sv3) or 1.0.0 (main)
nside=256     # lrg=256, elg=1024
fnltag="zero"
model=$3    # dnnp, linp
method=${model}_${maps}       # dnnp_known1, linp_known, or noweight
lmin=$4
p=$5
s=$6
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
mcmclog=${HOME}/github/LSSutils/analysis/desi/scripts/run_mcmc_logfast.py
mcmcf=${HOME}/github/LSSutils/analysis/desi/scripts/run_mcmc_frac.py
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
        axes=(0 4)   # EBV, galdepth-z
    elif [ $1 = "known1" ]
    then
        axes=(0 4 7) # EBV, galdepth-z, psfsize-r
    elif [ $1 = "knownpp" ]
    then
        axes=(0 4 7 9) # EBV, galdepth-z, psfsize-r, nstar**2

    elif [ $1 = "knownp" ]
    then
        axes=(0 1 4 7) # EBV, nstar, galdepth-z, psfsize-r
    elif [ $1 = "known1ext" ]
    then
        axes=(0 4 7 9 10) # EBV, galdepth-z, psfsize-r, calibz, logHI

    elif [ $1 = "known2" ]
    then
        axes=(0 2 3 4) # EBV,galdepth-grz

    elif [ $1 = "all" ]
    then
        axes=(0 2 3 4 5 6 7 8) # all maps
    elif [ $1 = "allp" ]
    then
        axes=(0 1 2 3 4 5 6 7 8) # all maps p nstar
    fi

    echo ${axes[@]}
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


if [ "${do_fit}" = true ]
then
        lr=$(get_lr ${target})
        axes=$(get_axes ${maps})
        echo ${target} ${region} $lr ${axes[@]} $maps
        input_path=${root_dir}/tables/${tag_d}/n${target}_features_${region}_${nside}.fits
        ## uncomment for linp
        #output_path=${root_dir}/regression/${tag_d}/${model}_${target}_${region}_${nside}_${maps}/mcmc_${region}_${maps}.npz
        output_path=${root_dir}/regression/${tag_d}/${model}_${target}_${region}_${nside}_${maps}/
        echo $output_path 
        du -h $input_path
        srun -n 1 python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model $model --loss $loss --nn_structure ${nns[@]} -lr $lr --eta_min $etamin -ne $nepoch -k -nc $nchain
        #python $linfit -d $input_path -o $output_path -ax ${axes[@]}
fi


if [ "${do_linsam}" = true ]
then
    axes=$(get_axes ${maps})
    output_path=${root_dir}/regression/${tag_d}/${model}_${target}_${region}_${maps}.hp${nside}.fits
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
        srun -n 4 python $nbar -d ${input_path} -o ${output_path}
    fi
    
    output_path=${root_dir}/clustering/${tag_d}/nbar_${target}_${region}_${nside}_${model}_${maps}.npy
    selection=${root_dir}/regression/${tag_d}/${model}_${target}_desic_${maps}.hp${nside}.fits
    du -h $selection
    echo $output_path
    srun -n 4 python $nbar -d ${input_path} -o ${output_path} -s ${selection}            
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
    selection=${root_dir}/regression/${tag_d}/${model}_${target}_desic_${maps}.hp${nside}.fits
    du -h $selection
    echo $output_path
    srun -n 4 python $cl -d ${input_path} -o ${output_path} -s ${selection}           
fi


if [ "${do_mcmc}" = true ]
then
    path_cl=${root_dir}/clustering/${tag_d}/cl_${target}_${region}_${nside}_${method}.npy
    path_cov=${mock_dir}/clustering/logclmock_0_${target}_${fnltag}_${region}_256_noweight_cov.npz
    output_mcmc=${root_dir}/mcmc/${tag_d}/logmcmc_${target}_${fnltag}_${region}_${method}_steps10k_walkers50_elmin${lmin}_p${p}_s${s}.npz  
    
    du -h $path_cl $path_cov
    echo $target $region $maps $output_mcmc
    python $mcmclog --path_cl $path_cl --path_cov $path_cov --region $region --output $output_mcmc --scale --elmin $lmin --p $p --s $s
fi


if [ "${do_mcmc_joint3}" = true ]
then
    region=$1
    region1=$2
    region2=$3

    path_cl=${root_dir}/clustering/${tag_d}/cl_${target}_${region}_${nside}_${method}.npy
    path_cov=${mock_dir}/clustering/logclmock_0_${target}_${fnltag}_${region}_256_noweight_cov.npz
    path_cl1=${root_dir}/clustering/${tag_d}/cl_${target}_${region1}_${nside}_${method}.npy
    path_cov1=${mock_dir}/clustering/logclmock_0_${target}_${fnltag}_${region1}_256_noweight_cov.npz
    path_cl2=${root_dir}/clustering/${tag_d}/cl_${target}_${region2}_${nside}_${method}.npy
    path_cov2=${mock_dir}/clustering/logclmock_0_${target}_${fnltag}_${region2}_256_noweight_cov.npz
     
    regionj=${region}${region1}${region2}
    output_mcmc=${root_dir}/mcmc/${tag_d}/logmcmc_${target}_${fnltag}_${regionj}_${method}_steps10k_walkers50_elmin${lmin}.npz   
    
    du -h $path_cl $path_cov
    du -h $path_cl1 $path_cov1
    du -h $path_cl2 $path_cov2
    echo $target $region $reion1 $region2 $maps $output_mcmc

    python $mcmc_joint3 $path_cl $path_cl1 $path_cl2 $path_cov $path_cov1 $path_cov2 $region $region1 $region2 $output_mcmc 1.0
fi


