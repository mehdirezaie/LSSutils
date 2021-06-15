
. "/home/mehdi/miniconda3/etc/profile.d/conda.sh"
export NUMEXPR_MAX_THREADS=2
export PYTHONPATH=${HOME}/github/LSSutils:${HOME}/github/sysnetdev
conda activate sysnet

do_prep=false
do_lr=false
do_nl=false
do_fit=true
do_assign=false
do_nbar=false
do_cl=false

cversion=v1
mversion=v2
nside=256
bsize=4098 # v1 500
targets=$1 #'QSO'
regions=$2 #BMZLS
axes=({0..12})
model=dnn
loss=mse
nns=(4 20)
nepoch=150 # v0 with 71
nchain=20
etamin=0.001

#root_dir=/home/mehdi/data/dr9v0.57.0/sv3nn_${cversion}
root_dir=/home/mehdi/data/rongpu/imaging_sys

prep=${HOME}/github/LSSutils/scripts/analysis/desi/prep_desi.py
nnfit=${HOME}/github/sysnetdev/scripts/app.py
svplot=${HOME}/github/LSSutils/scripts/analysis/desi/plot_sv3.py
nbar=${HOME}/github/LSSutils/scripts/analysis/desi/run_nnbar_sv3.py
assign=${HOME}/github/LSSutils/scripts/analysis/desi/fetch_weights.py

function get_lr(){
    if [ $1 = "lrg" ]
    then
        lr=0.2
    elif [ $1 = "elg" ]
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
            python $prep $target $region
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
            input_path=${root_dir}/tables/n${target}_features_${region}_${nside}.fits
            output_path=${root_dir}/regression/${mversion}/sv3nn_${target}_${region}_${nside}/hp/
            python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model $model --loss $loss --nn_structure ${nns[@]} -fl
        done
    done
fi

if [ "${do_nl}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            lr=$(get_lr ${target})
            echo ${target} ${region} $lr
            input_path=${root_dir}/tables/n${target}_features_${region}_${nside}.fits
            output_path=${root_dir}/regression/${mversion}/sv3nn_${target}_${region}_${nside}/hp/
            python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model $model --loss $loss -lr $lr -fs
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
            input_path=${root_dir}/tables/n${target}_features_${region}_${nside}.fits
            output_path=${root_dir}/regression/${mversion}/sv3nn_${target}_${region}_${nside}/
            du -h $input_path
            python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model $model --loss $loss --nn_structure ${nns[@]} -lr $lr --eta_min $etamin -ne $nepoch -k -nc $nchain 
        done
    done
fi


conda deactivate
conda activate py3p6


if [ "${do_assign}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            echo ${target} ${region} ${mversion}
            python $assign ${target} ${region} ${mversion}
        done
    done
fi


if [ "${do_nbar}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            input_path=${root_dir}/tables/n${target}_features_${region}_${nside}.fits
            output_path=${root_dir}/clustering/${mversion}/nbar_${target}_${region}_${nside}_noweight.npy

            mpirun -np 4 python $nbar -d ${input_path} -o ${output_path}
            
            output_path=${root_dir}/clustering/${mversion}/nbar_${target}_${region}_${nside}_nn.npy
            selection=${root_dir}/regression/${mversion}/sv3nn_${target}_${region}_${nside}/nn-weights.fits
            mpirun -np 4 python $nbar -d ${input_path} -o ${output_path} -s ${selection}            
            
        done
    done
fi

if [ "${do_cl}" = true ]
then
    for target in ${targets}
    do
        for region in ${regions}
        do
            python $svplot $region $target ${mversion}
        done
    done
fi
