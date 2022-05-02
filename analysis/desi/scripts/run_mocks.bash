#!/bin/bash
#SBATCH --job-name=nncont
#SBATCH --account=PHS0336 
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
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

do_prep=false   #
do_nn=true     # 20 h
do_nbar=false   # 10 m x 4
do_cl=false     # 10 m x 4
do_clfull=false # 10 m x 14
do_mcmc=false    #  3 h x 14
do_mcmc_joint=false # 3hx14
do_mcmc_joint3=false # 5x14
do_mcmc_scale=false
do_bfit=false   #  3 h x 14

mockid=1 # for debugging
#printf -v mockid "%d" $SLURM_ARRAY_TASK_ID
#echo ${mockid}
bsize=4098
region="bmzls" # bmzls, ndecals, sdecals
iscont=1
maps="known2"
target="lrg"
fnltag="zero" #zero, po100
ver=v2 # 
root_dir=/fs/ess/PHS0336/data/lognormal/${ver}
root_dir2=/fs/ess/PHS0336/data/rongpu/imaging_sys/tables
nside=256
model=dnn
loss=mse
nns=(4 20)
nepoch=70 # or 150
nchain=20
etamin=0.001
lr=0.2


prep=${HOME}/github/LSSutils/analysis/desi/scripts/prep_mocks.py
cl=${HOME}/github/LSSutils/analysis/desi/scripts/run_cell_mocks.py
clfull=${HOME}/github/LSSutils/analysis/desi/scripts/run_cell_mocks_mpi.py
nbar=${HOME}/github/LSSutils/analysis/desi/scripts/run_nnbar_mocks.py
nnfit=${HOME}/github/sysnetdev/scripts/app.py
mcmc=${HOME}/github/LSSutils/analysis/desi/scripts/run_mcmc_fast.py
mcmc_joint=${HOME}/github/LSSutils/analysis/desi/scripts/run_mcmc_joint.py
mcmc_joint3=${HOME}/github/LSSutils/analysis/desi/scripts/run_mcmc_joint3.py
bfit=${HOME}/github/LSSutils/analysis/desi/scripts/run_best_fit.py

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


if [ "${do_prep}" = true ]
then
    for region in ${regions}
    do
        echo $region
        input_path=${root_dir}/lrg-${mockid}-f1z1.fits
        output_path=${root_dir}/tables/${region}/nlrg-${mockid}-${region}.fits

        du -h $input_path
        echo $output_path
        srun -n 1 python $prep $input_path $output_path ${region}
    done
fi


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
    #srun -n 4 python $nbar -d ${input_path} -m ${input_map} -o ${output_path}

    # nn weight
    output_path=${root_dir}/clustering/nbarmock_${iscont}_${mockid}_${target}_${fnltag}_${region}_${nside}_nn_${maps}.npy                
    echo $output_path
    srun -n 4 python $nbar -d ${input_path} -m ${input_map} -o ${output_path} -s ${input_wsys}
fi

if [ "${do_cl}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            #for mockid in {1..1000}
            #do
                input_path=${root_dir2}/n${target}_features_${region}_${nside}.fits
                input_map=${root_dir}/lrg-${mockid}-f1z1.fits
                output_path=${root_dir}/clustering/clmock_${mockid}_${target}_${region}_${nside}_noweight.npy
                
                du -h $input_map $input_path
                #echo $output_path
                #srun -np 4 python $cl -d ${input_path} -m ${input_map} -o ${output_path}

                # nn weight
                input_wsys=${root_dir}/regression/${mockid}/${region}/nn/nn-weights.fits
                output_path=${root_dir}/clustering/clmock_${mockid}_${target}_${region}_${nside}_nn.npy
                du -h $input_wsys
                echo $output_path
                srun -n 4 python $cl -d ${input_path} -m ${input_map} -o ${output_path} -s ${input_wsys}
            #done
        done
    done
fi


if [ "${do_clfull}" = true ]
then
    region=$1
    indir=${root_dir}/hpmaps/
    oudir=${root_dir}/clustering/clmock_${fnltag}_${region}.npy
    echo $region
    echo $indir
    echo $oudir

    srun -n 14 python $clfull -m ${indir} -o ${oudir} -t ${fnltag} -r $region
fi

if [ "${do_mcmc}" = true ]
then
    region=$1
    method=noweight
    path_cl=${root_dir}/clustering/clmock_${fnltag}_${region}_mean.npz
    path_cov=${root_dir}/clustering/clmock_${fnltag}_${region}_cov.npz

    
    output_mcmc=${root_dir}/mcmc/mcmc_${target}_${fnltag}_${region}_${method}_steps10k_walkers50.npz
        
    du -h $path_cl $path_cov
    echo $target $region $method $output_mcmc
    python $mcmc $path_cl $path_cov $region $output_mcmc
fi

if [ "${do_mcmc_scale}" = true ]
then
    region=fullskyscaled
    fnltag=zero
    method=noweight
    path_cl=${root_dir}/clustering/clmock_${fnltag}_fullsky_mean.npz
    path_cov=${root_dir}/clustering/clmock_${fnltag}_bmzls_cov.npz
    
    output_mcmc=${root_dir}/mcmc/mcmc_${target}_${fnltag}_${region}_${method}_steps10k_walkers50.npz
        
    du -h $path_cl $path_cov
    echo $target $region $method $output_mcmc
    python $mcmc $path_cl $path_cov $region $output_mcmc
fi


if [ "${do_mcmc_joint}" = true ]
then
    region=$1
    region1=$2
    fnltag=zero
    method=noweight
    path_cl=${root_dir}/clustering/clmock_${fnltag}_${region}_mean.npz
    path_cov=${root_dir}/clustering/clmock_${fnltag}_${region}_cov.npz
    path_cl1=${root_dir}/clustering/clmock_${fnltag}_${region1}_mean.npz
    path_cov1=${root_dir}/clustering/clmock_${fnltag}_${region1}_cov.npz

    regionj=${region}${region1}
    output_mcmc=${root_dir}/mcmc/mcmc_${target}_${fnltag}_${regionj}_${method}_steps10k_walkers50.npz
        
    du -h $path_cl $path_cov
    du -h $path_cl1 $path_cov1
    echo $target $region $reion1 $method $output_mcmc
    
    python $mcmc_joint $path_cl $path_cl1 $path_cov $path_cov1 $region $region1 $output_mcmc
fi

if [ "${do_mcmc_joint3}" = true ]
then
    region=$1
    region1=$2
    region2=$3
    fnltag=zero
    method=noweight
    path_cl=${root_dir}/clustering/clmock_${fnltag}_${region}_mean.npz
    path_cov=${root_dir}/clustering/clmock_${fnltag}_${region}_cov.npz
    path_cl1=${root_dir}/clustering/clmock_${fnltag}_${region1}_mean.npz
    path_cov1=${root_dir}/clustering/clmock_${fnltag}_${region1}_cov.npz
    path_cl2=${root_dir}/clustering/clmock_${fnltag}_${region2}_mean.npz
    path_cov2=${root_dir}/clustering/clmock_${fnltag}_${region2}_cov.npz

    regionj=${region}${region1}${region2}
    output_mcmc=${root_dir}/mcmc/mcmc_${target}_${fnltag}_${regionj}_${method}_steps10k_walkers50.npz
        
    du -h $path_cl $path_cov
    du -h $path_cl1 $path_cov1
    du -h $path_cl2 $path_cov2
    echo $target $region $reion1 $region2 $method $output_mcmc
    
    python $mcmc_joint3 $path_cl $path_cl1 $path_cl2 $path_cov $path_cov1 $path_cov2 $region $region1 $region2 $output_mcmc
fi



if [ "${do_bfit}" = true ]
then
    # options are fullsky, bmzls, ndecals, sdecals
    region=$1
    method=noweight
    path_cl=${root_dir}/clustering/clmock_${fnltag}_${region}.npy
    path_cov=${root_dir}/clustering/clmock_${fnltag}_${region}_cov.npz
    output_bestfit=${root_dir}/mcmc/bestfit_${target}_${fnltag}_${region}_${method}.npz

    srun -n 14 python $bfit $region $path_cl $path_cov $output_bestfit
fi
