
. "/home/mehdi/miniconda3/etc/profile.d/conda.sh"
export NUMEXPR_MAX_THREADS=2
export PYTHONPATH=${HOME}/github/LSSutils:${HOME}/github/sysnetdev

do_nbar=false
do_cl=true

target='elg'
regions=$1 #BMZLS
root_dir=/home/mehdi/data/tanveer/mocks/
root_dir2=/home/mehdi/data/rongpu/imaging_sys/tables
nside=1024



cl=${HOME}/github/LSSutils/scripts/analysis/lssxcmb/run_cell_mocks.py


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

if [ "${do_cl}" = true ]
then
        for region in ${regions}
        do
            for mockid in {0..999}
            do
                input_path=${root_dir2}/n${target}_features_${region}_${nside}.fits
                input_map=${root_dir}/delta_${mockid}.hp1024.fits
                output_path=${root_dir}/clustering/clmock_${mockid}_${target}_${region}_${nside}_noweight.npy
                
                du -h $input_map $input_path
                echo $output_path
                mpirun -np 8 python $cl -d ${input_path} -m ${input_map} -o ${output_path}
            done
        done
fi
