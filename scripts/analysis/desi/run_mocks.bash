
. "/home/mehdi/miniconda3/etc/profile.d/conda.sh"
export NUMEXPR_MAX_THREADS=2
export PYTHONPATH=${HOME}/github/LSSutils:${HOME}/github/sysnetdev

do_nbar=true
do_cl=false

regions=$1 #BMZLS
targets="lrg"
ver=v0
root_dir=/home/mehdi/data/lognormal/${ver}
root_dir2=/home/mehdi/data/rongpu/imaging_sys/tables
nside=256



cl=${HOME}/github/LSSutils/scripts/analysis/desi/run_cell_mocks.py
nbar=${HOME}/github/LSSutils/scripts/analysis/desi/run_nnbar_mocks.py


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



conda activate py3p6

# bash run_sv3.bash 'LRG ELG BGS_ANY' 'NBMZLS NDECALS SDECALS SDECALS_noDES DES'

if [ "${do_nbar}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            for mockid in {1..1000}
            do
                input_path=${root_dir2}/n${target}_features_${region}_${nside}.fits
                input_map=${root_dir}/lrg-${mockid}-f1z1.fits
                output_path=${root_dir}/clustering/nbarmock_${mockid}_${target}_${region}_${nside}_noweight.npy
                
                du -h $input_map $input_path
                echo $output_path
                mpirun -np 4 python $nbar -d ${input_path} -m ${input_map} -o ${output_path}
            done
        done
    done
fi

if [ "${do_cl}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            for mockid in {1..1000}
            do
                input_path=${root_dir2}/n${target}_features_${region}_${nside}.fits
                input_map=${root_dir}/lrg-${mockid}-f1z1.fits
                output_path=${root_dir}/clustering/clmock_${mockid}_${target}_${region}_${nside}_noweight.npy
                
                du -h $input_map $input_path
                echo $output_path
                mpirun -np 4 python $cl -d ${input_path} -m ${input_map} -o ${output_path}
            done
        done
    done
fi
