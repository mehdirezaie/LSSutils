. "/home/mehdi/miniconda3/etc/profile.d/conda.sh"
export NUMEXPR_MAX_THREADS=2
export PYTHONPATH=${HOME}/github/LSSutils:${HOME}/github/sysnetdev
conda activate sysnet

do_cl=true

target=elg
region=bmzls
nside=1024
mversion=v2

cl=${HOME}/github/LSSutils/scripts/analysis/lssxcmb/run_cell_sv3.py
root_dir=/home/mehdi/data/rongpu/imaging_sys



if [ "${do_cl}" = true ]
then
    input_path=${root_dir}/tables/n${target}_features_${region}_${nside}.fits
    output_path=${root_dir}/clustering/${mversion}/cl_${target}_${region}_${nside}_noweight.npy
    du -h $input_path
    echo $output_path
    mpirun -np 4 python $cl -d ${input_path} -o ${output_path}
    
    output_path=${root_dir}/clustering/${mversion}/cl_${target}_${region}_${nside}_nn.npy
    selection=/home/mehdi/data/tanveer/dr9/elg_mse_snapshots/${region}/windows_mean.hp1024.fits
    echo $output_path
    du -h $selection
    mpirun -np 4 python $cl -d ${input_path} -o ${output_path} -s ${selection}            
fi



