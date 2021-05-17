
. "/home/mehdi/miniconda3/etc/profile.d/conda.sh"
export NUMEXPR_MAX_THREADS=2
export PYTHONPATH=${HOME}/github/LSSutils:${HOME}/github/sysnetdev
conda activate sysnet

do_prep=false
do_lr=false
do_fit=true
do_cl=false


nside=256
bsize=500
regions=$1 #NBMZLS
targets=$2 #'QSO'
axes=({0..13})
#'lognstar', 'ebv', 'loghi',
#'psfdepth_g', 'psfdepth_r', 'psfdepth_z',
#'galdepth_g', 'galdepth_r', 'galdepth_z', 
#'psfsize_g', 'psfsize_r', 'psfsize_z', 
#'psfdepth_w1', 'psfdepth_w2'
#
model=dnn
loss=mse
nns=(4 20)
nepoch=71
nchain=20
version=v1

root_dir=/home/mehdi/data/dr9v0.57.0/sv3nn_${version}

prep=${HOME}/github/LSSutils/scripts/analysis/prep_desi.py
nnfit=${HOME}/github/sysnetdev/scripts/app.py
svplot=${HOME}/github/LSSutils/scripts/analysis/plot_sv3.py

function get_lr(){
    if [ $1 = "LRG" ]
    then
        lr=0.2
    elif [ $1 = "ELG" ]
    then
        lr=0.3
    elif [ $1 = "QSO" ]
    then
        lr=0.2
    elif [ $1 = 'BGS_ANY' ]
    then
        lr=0.3
    fi
    echo $lr
}



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
            input_path=${root_dir}/tables/sv3tab_${target}_${region}_${nside}.fits
            output_path=${root_dir}/regression/sv3nn_${target}_${region}_${nside}/hp/
            python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model $model --loss $loss --nn_structure ${nns[@]} -fl
        done
    done
fi


if [ "${do_fit}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            lr=$(get_lr ${target})
            echo ${target} ${region} $lr
            input_path=${root_dir}/tables/sv3tab_${target}_${region}_${nside}.fits
            output_path=${root_dir}/regression/sv3nn_${target}_${region}_${nside}/
            du -h $input_path
            python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model $model --loss $loss --nn_structure ${nns[@]} -lr $lr -ne $nepoch -k -nc $nchain 
        done
    done
fi


conda deactivate
conda activate py3p6
if [ "${do_cl}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            python $svplot $region $target
        done
    done
fi
