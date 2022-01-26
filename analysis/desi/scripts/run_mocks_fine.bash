#!/bin/bash
#SBATCH --job-name=fitfnl
#SBATCH --account=PHS0336 
#SBATCH --time=03:00:00
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

do_prep=false
do_nn=false       # 20 h
do_assign=false
do_nbar=false     # 10 m
do_cl=false       # 10 m
do_clfull=false   # 10 m x 14
do_mcmc=false      # 3 h x 14
do_mcmc_full=true # 3 h x 14

#mockid=1
printf -v mockid "%d" $SLURM_ARRAY_TASK_ID
echo ${mockid}
bsize=4098
regions=$1
targets="lrg"
ver=v0
root_dir=/fs/ess/PHS0336/data/lognormal/${ver}
root_dir2=/fs/ess/PHS0336/data/rongpu/imaging_sys/tables
nside=256
axes=({0..12})
model=dnn
loss=mse
nns=(4 20)
nepoch=70 # or 150
nchain=20
etamin=0.001
lr=0.2


prep=${HOME}/github/LSSutils/scripts/analysis/desi/prep_mocks.py
cl=${HOME}/github/LSSutils/scripts/analysis/desi/run_cell_mocks.py
clfull=${HOME}/github/LSSutils/analysis/desi/scripts/run_cell_mocks_mpi.py
nbar=${HOME}/github/LSSutils/scripts/analysis/desi/run_nnbar_mocks.py
nnfit=${HOME}/github/sysnetdev/scripts/app.py
mcmc=${HOME}/github/LSSutils/analysis/desi/scripts/run_mcmc_fast.py
mcmc2=${HOME}/github/LSSutils/analysis/desi/scripts/run_mcmc_fast2.py



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
    for region in ${regions}
    do
        input_path=${root_dir}/tables/${region}/nlrg-${mockid}-${region}.fits
        output_path=${root_dir}/regression/${mockid}/${region}/nn/
        du -h $input_path
        echo $output_path	
        srun -n 1 python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model $model --loss $loss --nn_structure ${nns[@]} -lr $lr --eta_min $etamin -ne $nepoch -k -nc $nchain --do_tar -k
    done
fi



if [ "${do_nbar}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            #for mockid in {1..1000} ##--- no need to do this with slurm array
            #do
                echo $target $region
                input_path=${root_dir2}/n${target}_features_${region}_${nside}.fits
                input_map=${root_dir}/${target}-${mockid}-f1z1.fits
                du -h $input_map $input_path

                # no weight
                output_path=${root_dir}/clustering/nbarmock_${mockid}_${target}_${region}_${nside}_noweight.npy                
                echo $output_path
                #srun -n 4 python $nbar -d ${input_path} -m ${input_map} -o ${output_path}
                #python ./run_nmocks.py -d ${input_path} -m ${input_map} -o ${output_path} # find num. of mock gal in each catalog

                # nn weight
                input_wsys=${root_dir}/regression/${mockid}/${region}/nn/nn-weights.fits
                output_path=${root_dir}/clustering/nbarmock_${mockid}_${target}_${region}_${nside}_nn.npy
                echo $output_path
                srun -n 4 python $nbar -d ${input_path} -m ${input_map} -o ${output_path} -s ${input_wsys}
            #done
        done
    done
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
    srun -n 14 python $clfull -m ${root_dir} -o ${root_dir}/clustering/clmock_fullsky.npy
fi

if [ "${do_mcmc}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do       
            method=$2
            path_cl=/fs/ess/PHS0336/data/lognormal/v0/clustering/clmock_${region}_${method}_mean_fine.npz
            path_cov=/fs/ess/PHS0336/data/lognormal/v0/clustering/clmock_${region}_noweight_cov_fine.npz   # fix covariance
            output_mcmc=${root_dir}/mcmc/mcmc_${target}_${region}_${method}_steps10000_walkers50_fine.npz
            
            du -h $path_cl $path_cov
            echo $target $region $method $output_mcmc
            python $mcmc $path_cl $path_cov $region $output_mcmc
        done
    done
fi


if [ "${do_mcmc_full}" = true ]
then
    for target in ${targets}
    do
        # full sky
        region=fullsky
        method=noweight
        path_cl=/fs/ess/PHS0336/data/lognormal/v0/clustering/clmock_fullsky_mean_fine.npz
        path_cov=/fs/ess/PHS0336/data/lognormal/v0/clustering/clmock_fullsky_cov_fine.npz   # fix covariance
        output_mcmc=${root_dir}/mcmc/mcmc_${target}_${region}_${method}_steps10000_walkers50_fine.npz

        du -h $path_cl $path_cov
        echo $target $region $method $output_mcmc
        python $mcmc $path_cl $path_cov $region $output_mcmc        
    done
fi
