

# from command line ??
printf -v mockid "%04d" $1

# parameters
nmesh=1024
boxsize=5000.
dk=0.01
kmax=0.4
cosmology='ezmock'

# --- path to input catalogs and output pk
dat_path=/home/mehdi/data/eboss/mocks/1.0/catalogs_raw/null/
out_path=/home/mehdi/data/eboss/mocks/test/pk_null/

dat_fn=${dat_path}EZmock_eBOSS_QSO_NGC_v7_noweight_${mockid}.dat.fits
ran_fn=${dat_fn/.dat./.ran.}
out_pk_fn=${out_path}pk_null_${mockid}.json

echo "data: "${dat_fn}
echo "random: "${ran_fn}
echo "output pk: "${out_pk_fn}


# path to software
export PYTHONPATH=${HOME}/github/sysnetdev:${HOME}/github/LSSutils:${PYTHONPATH}
export NUMEXPR_MAX_THREADS=2
do_pk=${HOME}/github/LSSutils/scripts/analysis/run_pk.py


. "/home/mehdi/miniconda3/etc/profile.d/conda.sh"
conda activate py3p6

mpirun -np 8 python ${do_pk} -g ${dat_fn} -r ${ran_fn} -o ${out_pk_fn} \
                             -n ${nmesh} --dk ${dk} --kmax ${kmax} -b ${boxsize} -z 0.8 2.2 --cosmo ${cosmology} \
                             -p 0 2 4 --use_systot

