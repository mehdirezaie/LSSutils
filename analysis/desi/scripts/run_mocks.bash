#!/bin/bash
#SBATCH --job-name=mcmc
#SBATCH --account=PHS0336 
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=14
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mr095415@ohio.edu

## run with
# sbatch --array=1-100 run_mocks.bash ndecals
# manually add the path, later we will install the pipeline with `pip`
source ${HOME}/.bashrc

export PYTHONPATH=${HOME}/github/sysnetdev:${HOME}/github/LSSutils:${PYTHONPATH}
export NUMEXPR_MAX_THREADS=1
export OMP_NUM_THREADS=1

source activate sysnet
cd ${HOME}/github/LSSutils/analysis/desi/scripts/

do_prep=false        #
do_nn=false          # 10 h
do_regrs=false       # 25 min
do_nbar=false        # 10 m x 4
do_cl=false          # 10 m x 4
do_clfull=false      # 10 m x 14
do_mcmc=false        # 3 h x 14
do_mcmc_scale=false  #
do_mcmc_log=false
do_mcmc_logscale=false
do_bfit=true        # 3 h x 14
do_mcmc_cont=false   # 
do_mcmc_joint=false  # 3hx14
do_mcmc_joint3=false # 5x14

#mockid=1 # for debugging
printf -v mockid "%d" $SLURM_ARRAY_TASK_ID
echo ${mockid}
bsize=5000
region=$1 # desi, bmzls, ndecals, sdecals
iscont=0
maps="noweight" #e.g., "known5" or "all"
method="noweight" # noweight, nn_all
target="lrg"
fnltag="zero" #"zero" #zero, po100
ver=v3 # 
root_dir=/fs/ess/PHS0336/data/lognormal/${ver}
root_dir2=/fs/ess/PHS0336/data/rongpu/imaging_sys/tables
nside=256
model=dnnp
loss=pnll
nns=(4 20)
nepoch=70 # or 150
nchain=20
etamin=0.001
lr=0.2


prep=${HOME}/github/LSSutils/analysis/desi/scripts/prep_mocks.py
regrs=${HOME}/github/LSSutils/analysis/desi/scripts/run_regressis.py
cl=${HOME}/github/LSSutils/analysis/desi/scripts/run_cell_mocks.py
clfull=${HOME}/github/LSSutils/analysis/desi/scripts/run_cell_mocks_mpi.py
nbar=${HOME}/github/LSSutils/analysis/desi/scripts/run_nnbar_mocks.py
nnfit=${HOME}/github/sysnetdev/scripts/app.py
mcmc=${HOME}/github/LSSutils/analysis/desi/scripts/run_mcmc_fast.py
mcmclog=${HOME}/github/LSSutils/analysis/desi/scripts/run_mcmc_logfast.py
mcmc_joint=${HOME}/github/LSSutils/analysis/desi/scripts/run_mcmc_joint.py
mcmc_joint3=${HOME}/github/LSSutils/analysis/desi/scripts/run_mcmc_joint3.py
bfit=${HOME}/github/LSSutils/analysis/desi/scripts/run_best_fit.py
bfitlog=${HOME}/github/LSSutils/analysis/desi/scripts/run_best_logfit.py

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
        axes=(0 6)   # ebv, psfdepth-g 
    elif [ $1 = "known3" ]
    then
        axes=(0 1 6) # ebv, nstar, psfdepth-g
    elif [ $1 = "known4" ]
    then
        axes=(0 1 6 11) # ebv, nstar, psfdepth-g, psfsize-g
    elif [ $1 = "known5" ]
    then
        axes=(0 1 4 6 11) # ebv, nstar, galdepth-z, psfdepth-g, psfsize-g
    elif [ $1 = "all2" ]
    then
        axes=({0..12})
        axes=($(echo ${axes[*]}) $(echo ${axes[*]}))
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
    #python $nnfit -i ${input_path} ${input_map} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model $model --loss $loss --nn_structure ${nns[@]}  -fl
    srun -n 1 python $nnfit -i ${input_path} ${input_map} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model $model --loss $loss --nn_structure ${nns[@]}  -lr $lr --eta_min $etamin -ne $nepoch -k -nc $nchain --do_tar -k
fi


if [ "${do_regrs}" = true ]
then
    if [ $iscont = 1 ]
    then
        input_map=${root_dir}/hpmaps/${target}hp-${fnltag}-${mockid}-f1z1-contaminated.fits
    else
        input_map=${root_dir}/hpmaps/${target}hp-${fnltag}-${mockid}-f1z1.fits
    fi
    echo $target $region $iscont $maps
    
    du -h $input_map

    if [ $iscont = 1 ]
    then
        output_path=${root_dir}/clustering/clmock_${iscont}_${mockid}_${target}_${fnltag}_${region}_${nside}_nn_${maps}_regrslin.npz 
        echo $output_path
        srun -n 1 python $regrs ${input_map} ${output_path} Linear
    fi
fi


if [ "${do_nbar}" = true ]
then
    if [ $iscont = 1 ]
    then
        input_map=${root_dir}/hpmaps/${target}hp-${fnltag}-${mockid}-f1z1-contaminated.fits
        input_wsys=${root_dir}/regression/fnl_${fnltag}/cont/${mockid}/${region}/nn_${maps}/nn-weights.fits
    else
        input_map=${root_dir}/hpmaps/${target}hp-${fnltag}-${mockid}-f1z1.fits
        input_wsys=${root_dir}/regression/fnl_${fnltag}/null/${mockid}/${region}/nn_${maps}/nn-weights.fits
    fi
    echo $target $region $iscont $maps
    input_path=${root_dir2}/0.57.0/n${target}_features_${region}_${nside}.fits
    
    du -h $input_map $input_path $input_wsys

    # no weight
    output_path=${root_dir}/clustering/nbarmock_${iscont}_${mockid}_${target}_${fnltag}_${region}_${nside}_noweight.npy                
    echo $output_path
    if [ ! -f $output_path ]
    then
        echo "running w/o weights"
        srun -n 4 python $nbar -d ${input_path} -m ${input_map} -o ${output_path}
    fi

    if [ $iscont = 1 ]
    then
        # nn weight
        output_path=${root_dir}/clustering/nbarmock_${iscont}_${mockid}_${target}_${fnltag}_${region}_${nside}_nn_${maps}.npy                
        echo $output_path
        srun -n 4 python $nbar -d ${input_path} -m ${input_map} -o ${output_path} -s ${input_wsys}
    fi
fi


if [ "${do_cl}" = true ]
then
    if [ $iscont = 1 ]
    then
        input_map=${root_dir}/hpmaps/${target}hp-${fnltag}-${mockid}-f1z1-contaminated.fits
        input_wsys=${root_dir}/regression/fnl_${fnltag}/cont/${mockid}/${region}/nn_${maps}/nn-weights.fits
        #input_wsys=/fs/ess/PHS0336/data/lognormal/v2/regression/fnl_zero_cont_bmzls_nn_known5.fits
    else
        input_map=${root_dir}/hpmaps/${target}hp-${fnltag}-${mockid}-f1z1.fits
        input_wsys=${root_dir}/regression/fnl_${fnltag}/null/${mockid}/${region}/nn_${maps}/nn-weights.fits
    fi
    echo $target $region $iscont $maps
    input_path=${root_dir2}/0.57.0/n${target}_features_${region}_${nside}.fits
    
    du -h $input_map $input_path $input_wsys

    # no weight
    output_path=${root_dir}/clustering/clmock_${iscont}_${mockid}_${target}_${fnltag}_${region}_${nside}_noweight.npy                
    if [ ! -f $output_path ]
    then
        echo $output_path
        srun -n 4 python $cl -d ${input_path} -m ${input_map} -o ${output_path}
    fi

    if [ $iscont = 1 ]
    then
        # nn weight
        output_path=${root_dir}/clustering/clmock_${iscont}_${mockid}_${target}_${fnltag}_${region}_${nside}_nn_${maps}.npy                
        #output_path=${root_dir}/clustering/clmock_${iscont}_${mockid}_${target}_${fnltag}_${region}_${nside}_nn_${maps}_avgwsys0.npy                
        echo $output_path
        srun -n 4 python $cl -d ${input_path} -m ${input_map} -o ${output_path} -s ${input_wsys}
    fi
fi




if [ "${do_mcmc_log}" = true ]
then
    path_cl=${root_dir}/clustering/logclmock_${iscont}_${target}_${fnltag}_${region}_256_${method}_mean.npz
    path_cov=${root_dir}/clustering/logclmock_${iscont}_${target}_${fnltag}_${region}_256_${method}_cov.npz
    output_mcmc=${root_dir}/mcmc/logmcmc_${iscont}_${target}_${fnltag}_${region}_256_${method}_steps10k_walkers50.npz
        
    du -h $path_cl $path_cov
    echo $target $region $method $output_mcmc
    python $mcmclog $path_cl $path_cov $region $output_mcmc -1 
fi


if [ "${do_mcmc_logscale}" = true ]
then
    path_cl=${root_dir}/clustering/logclmock_${iscont}_${target}_po100_${region}_256_${method}_mean.npz
    path_cov=${root_dir}/clustering/logclmock_${iscont}_${target}_zero_${region}_256_${method}_cov.npz
    output_mcmc=${root_dir}/mcmc/logmcmc_${iscont}_${target}_pozero_${region}_256_${method}_steps10k_walkers50.npz
        
    du -h $path_cl $path_cov
    echo $target $region $method $output_mcmc
    python $mcmclog $path_cl $path_cov $region $output_mcmc -1 
fi


if [ "${do_bfit}" = true ]
then
    # options are fullsky, bmzls, ndecals, sdecals
    path_cov=${root_dir}/clustering/logclmock_${iscont}_${target}_${fnltag}_${region}_256_${method}_cov.npz
    output_bestfit=${root_dir}/mcmc/logbestfit_${iscont}_${target}_${fnltag}_${region}_256_${method}.npz

    du -h $path_cov 
    echo $output_bestfit $region
    srun -n 14 python $bfitlog $region $path_cov $output_bestfit
fi

#--- fit C-ell other than log C-ell (above)
if [ "${do_mcmc}" = true ]
then
    path_cl=${root_dir}/clustering/clmock_${iscont}_${target}_${fnltag}_${region}_256_${method}_mean.npz
    path_cov=${root_dir}/clustering/clmock_${iscont}_${target}_${fnltag}_${region}_256_${method}_cov.npz
    output_mcmc=${root_dir}/mcmc/mcmc_${iscont}_${target}_${fnltag}_${region}_256_${method}_steps10k_walkers50.npz
        
    du -h $path_cl $path_cov
    echo $target $region $method $output_mcmc
    python $mcmc $path_cl $path_cov $region $output_mcmc -1 
fi


if [ "${do_mcmc_scale}" = true ]
then
    path_cl=${root_dir}/clustering/clmock_${iscont}_${target}_po100_${region}_256_${method}_mean.npz
    path_cov=${root_dir}/clustering/clmock_${iscont}_${target}_zero_${region}_256_${method}_cov.npz
    output_mcmc=${root_dir}/mcmc/mcmc_${iscont}_${target}_pozero_${region}_256_${method}_steps10k_walkers50.npz
        
    du -h $path_cl $path_cov
    echo $target $region $method $output_mcmc
    python $mcmc $path_cl $path_cov $region $output_mcmc -1 
fi




if [ "${do_mcmc_joint}" = true ]
then
    region=$1
    region1=$2
    fnltag=zero
    method=noweight
    path_cl=${root_dir}/clustering/clmock_${iscont}_${target}_${fnltag}_${region}_256_${method}_mean.npz
    path_cov=${root_dir}/clustering/clmock_${iscont}_${target}_${fnltag}_${region}_256_${method}_cov.npz
    path_cl=${root_dir}/clustering/clmock_${iscont}_${target}_${fnltag}_${region1}_256_${method}_mean.npz
    path_cov=${root_dir}/clustering/clmock_${iscont}_${target}_${fnltag}_${region1}_256_${method}_cov.npz

    regionj=${region}${region1}
    output_mcmc=${root_dir}/mcmc/mcmc_${iscont}_${target}_${fnltag}_${regionj}_256_${method}_steps10k_walkers50.npz
   
    du -h $path_cl $path_cov
    du -h $path_cl1 $path_cov1
    echo $target $region $reion1 $method $output_mcmc
    
    #python $mcmc_joint $path_cl $path_cl1 $path_cov $path_cov1 $region $region1 $output_mcmc
fi

if [ "${do_mcmc_joint3}" = true ]
then
    region=$1
    region1=$2
    region2=$3
    fnltag=zero
    method=noweight
    path_cl=${root_dir}/clustering/clmock_${iscont}_${target}_${fnltag}_${region}_256_${method}_mean.npz
    path_cov=${root_dir}/clustering/clmock_${iscont}_${target}_${fnltag}_${region}_256_${method}_cov.npz
    path_cl1=${root_dir}/clustering/clmock_${iscont}_${target}_${fnltag}_${region1}_256_${method}_mean.npz
    path_cov1=${root_dir}/clustering/clmock_${iscont}_${target}_${fnltag}_${region1}_256_${method}_cov.npz
    path_cl2=${root_dir}/clustering/clmock_${iscont}_${target}_${fnltag}_${region2}_256_${method}_mean.npz
    path_cov2=${root_dir}/clustering/clmock_${iscont}_${target}_${fnltag}_${region2}_256_${method}_cov.npz

    regionj=${region}${region1}${region2}
    output_mcmc=${root_dir}/mcmc/mcmc_${iscont}_${target}_${fnltag}_${regionj}_256_${method}_steps10k_walkers50.npz
       
    du -h $path_cl $path_cov
    du -h $path_cl1 $path_cov1
    du -h $path_cl2 $path_cov2
    echo $target $region $reion1 $region2 $method $output_mcmc
    
    python $mcmc_joint3 $path_cl $path_cl1 $path_cl2 $path_cov $path_cov1 $path_cov2 $region $region1 $region2 $output_mcmc
fi

