
. "/home/mehdi/miniconda3/etc/profile.d/conda.sh"
export NUMEXPR_MAX_THREADS=2
export PYTHONPATH=${HOME}/github/LSSutils:${HOME}/github/sysnetdev
conda activate sysnet




nside=512
bsize=2000
regions=$1 #NBMZLS
targets=$2 #'QSO'
axes=({0..20})
# # 'nstar', 'ebv', 'loghi', 'ccdskymag_g_mean', 
# 'fwhm_g_mean', 'depth_g_total', 'mjd_g_min', airmass, exptime
nns=(5 10)
lr=0.1
etmin=0.0001
nepoch=71
nchain=20


do_prep=false
do_lr=false
do_fit=true


prep=${HOME}/github/LSSutils/scripts/analysis/prep_desi.py
nnfit=${HOME}/github/sysnetdev/scripts/app.py


if [ "${do_prep}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            echo ${target} ${region}
            python $prep $region $target
        done
    done
fi


if [ "${do_lr}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            echo ${target} ${region}
            input_path=/home/mehdi/data/sv3nn/tables/sv3tab_${target}_${region}.fits
            output_path=/home/mehdi/data/sv3nn/regression/sv3nn_${target}_${region}/hp/
            python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model dnn --loss mse --nn_structure ${nns[@]} -fl
        done
    done
fi


if [ "${do_fit}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            echo ${target} ${region}
            input_path=/home/mehdi/data/sv3nn/tables/sv3tab_${target}_${region}.fits
            output_path=/home/mehdi/data/sv3nn/regression/sv3nn_${target}_${region}/
            python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model dnn --loss mse --nn_structure ${nns[@]} -lr $lr -ne $nepoch -k -nc $nchain
        done
    done
fi
