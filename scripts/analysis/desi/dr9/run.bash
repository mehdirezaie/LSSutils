
#--- set up env.
#---- environment variables and activation
. "/home/mehdi/miniconda3/etc/profile.d/conda.sh"
export NUMEXPR_MAX_THREADS=2
export PYTHONPATH=${HOME}/github/LSSutils:${HOME}/github/sysnetdev
conda activate sysnet



prep=false
find_lr=false
find_st=false
find_ne=false
run_nn=true
run_nbar=false

cap=$1
target=$2

nepoch=300
nchains=20
nn_structure=(4 20)
model=dnn
loss=mse
batchsize=4098
axes=({0..12})
input=${PWD}/results/dr9m_${target}_${cap}.fits
output=${PWD}/results/regression/${target}/${cap}/nn_all_256


# path to software
prep=prepare_data_4nn.py
nnfit=${HOME}/github/sysnetdev/scripts/app.py
nnbar=./get_meandensity.py



# --- functions
function get_lr() {
    if [ $1 = "elg" ]
    then
        lr=0.3
    elif [ $1 = "lrg" ]
    then
        lr=0.2
    elif [ $1 = "qso" ]
    then
        lr=0.2
    fi
    echo $lr
}



if [ $prep = true ]
then
    echo ${target} ${cap}
    python $prep ${target} ${cap}
fi

if [ $find_lr = true ]
then
    du -h $input
    echo $output
    python $nnfit -i ${input} -o ${output}/hp/ -ax ${axes[@]} -bs $batchsize \
                  --loss $loss --model ${model} -fl
fi

if [ $find_st = true ]
then
    du -h $input
    echo $output
    lr=$(get_lr ${target})
    python $nnfit -i ${input} -o ${output}/hp/ -ax ${axes[@]} -bs $batchsize \
             --loss $loss --model $model -lr ${lr} -fs
fi

if [ $find_ne = true ]
then
    du -h $input
    echo $output
    lr=$(get_lr ${target})
    python $nnfit -i ${input} -o ${output}/hp/ -ax ${axes[@]} -bs $batchsize \
                  --loss $loss --model $model -lr ${lr} --nn_structure ${nn_structure[@]} \
                  -ne 300
fi

if [ $run_nn = true ]
then
    du -h $input
    echo $output
    lr=$(get_lr ${target})
    python $nnfit -i ${input} -o ${output}/ -ax ${axes[@]} -bs $batchsize \
                  --loss $loss --model $model -lr ${lr} --nn_structure ${nn_structure[@]} \
                  -ne $nepoch -nc $nchains -k
fi

if [ $run_nbar = true ]
then
    python $nnbar $target $cap
fi

