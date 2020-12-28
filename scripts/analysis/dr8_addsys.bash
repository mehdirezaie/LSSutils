#--- bash script to add systematics to a mock dataset

#---- environment variables and activation
. "/home/mehdi/miniconda3/etc/profile.d/conda.sh"
export NUMEXPR_MAX_THREADS=2
export PYTHONPATH=${HOME}/github/LSSutils:${HOME}/github/sysnetdev
conda activate sysnet

# parameters
axes=({0..26})
# 'ebv', 'loghi', 'nstar',
# 'depth_r_total', 'depth_g_total', 'depth_z_total',
#  'fwhm_r_mean', 'fwhm_g_mean', 'fwhm_z_mean',
# 'airmass_r_mean', 'airmass_g_mean', 'airmass_z_mean',
# 'ccdskymag_r_mean', 'ccdskymag_g_mean', 'ccdskymag_z_mean',
#  'exptime_r_total', 'exptime_g_total', 'exptime_z_total',
#  'mjd_r_min', 'mjd_g_min', 'mjd_z_min', 
# 'galdepth_g','galdepth_r', 'galdepth_z', 
# 'psfsize_g', 'psfsize_r', 'psfsize_z'
nchains=1  # 25 for NN-MSE
nepochs=300 # 70 for NN-MSE
lr=0.2  # NN-MSE
#lr=0.7   # Lin-MSE
etamin=0.001
input_path=/home/mehdi/data/tanveer/dr8_elg_0.32.0_256.fits
#output_path=/home/mehdi/data/tanveer/elg_mse/
#output_path=/home/mehdi/data/tanveer/elg_lin/
output_path=/home/mehdi/data/tanveer/elg_mse_snapshots/


find_lr=false
find_st=false
find_ne=false
do_nnfit=true


#---- path to the codes
prep=${HOME}/github/LSSutils/scripts/analysis/prepare_data_eboss.py
nnfit=${HOME}/github/sysnetdev/scripts/app.py
nnfite=${HOME}/github/sysnetdev/scripts/appensemble.py
swap=${HOME}/github/LSSutils/scripts/analysis/swap_data_eboss.py
pk=${HOME}/github/LSSutils/scripts/analysis/run_pk.py
nnbar=${HOME}/github/LSSutils/scripts/analysis/run_nnbar_eboss.py
cl=${HOME}/github/LSSutils/scripts/analysis/run_cell_eboss.py
xi=${HOME}/github/LSSutils/scripts/analysis/run_xi.py


#--- hyper-parameter search
if [ "${find_lr}" = true ]
then
    du -h ${input_path}
    output_path=${output_path}hp/
    echo ${output_path}
    #python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} --model dnn --loss mse -fl 
    python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} --model lin --loss mse -fl 
fi

if [ "${find_st}" = true ]
then
    du -h ${input_path}
    output_path=${output_path}hp/
    echo ${output_path}
    python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} --model dnn --loss mse -lr ${lr} --eta_min ${etamin} -fs 
fi

if [ "${find_ne}" = true ]
then
    du -h ${input_path}
    output_path=${output_path}hp/
    echo ${output_path}
    python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} --model dnn --loss mse -lr ${lr} --eta_min ${etamin} -ne 300

fi


#--- run neural network
if [ "${do_nnfit}" = true ]
then
    du -h ${input_path}
    echo ${output_path}
    #python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} --model dnn --loss mse -lr ${lr} --eta_min ${etamin} -ne $nepochs -nc $nchains -k
    #python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} --model lin --loss mse -lr ${lr} --eta_min ${etamin} -ne $nepochs -nc $nchains -k
    python $nnfite -i ${input_path} -o ${output_path} -ax ${axes[@]} --model dnn --loss mse -lr ${lr} --eta_min ${etamin} -ne $nepochs --snapshot_ensemble -k -bs 4096
fi

