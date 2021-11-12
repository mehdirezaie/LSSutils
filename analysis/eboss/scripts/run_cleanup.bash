
source ${HOME}/.bashrc

export PYTHONPATH=${HOME}/github/sysnetdev:${HOME}/github/LSSutils:${PYTHONPATH}
export NUMEXPR_MAX_THREADS=2
source activate sysnet



for mock in {2..100};
do 
    printf -v mockid "%04d" $mock
    for iscont in 0 1;
    do
        for nside in 256 512;
        do
            for temp in known all;
            do
		echo $mock $iscont $nside $temp
		model_path=/fs/ess/PHS0336/data/v7/1.0/${mockid}/${iscont}/${nside}/main/nn_pnnl_${temp}
		python cleanup_nnfit.py ${model_path} 
            done
        done
    done
done


