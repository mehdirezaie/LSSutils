#---- environment variables and activation
. "/home/mehdi/miniconda3/etc/profile.d/conda.sh"

export PYTHONPATH=${HOME}/github/LSSutils:${HOME}/github/sysnetdev
conda activate sysnet

#---- path to the codes
prep=/home/mehdi/github/LSSutils/scripts/analysis/prepare_data_eboss.py
nnfit=/home/mehdi/github/sysnetdev/scripts/app.py
swap=/home/mehdi/github/LSSutils/scripts/analysis/swap_data_eboss.py
pk=/home/mehdi/github/LSSutils/scripts/analysis/run_pk.py

#---- path to the data
nside=256
l1=-1.0 # l1 regularization deactivated with <0
nn_structure=(4 20) 
# 'star_density', 'ebv', 'loghi', 'sky_g', 'sky_r', 'sky_i', 'sky_z', 
# 'depth_g_minus_ebv','depth_r_minus_ebv', 'depth_i_minus_ebv', 'depth_z_minus_ebv', 
# 'psf_g', 'psf_r', 'psf_i', 'psf_z',
#  'run', 'airmass'
axes_all=({0..16})
axes_known=(1 5 7 13) # ebv, depth-g, psf-i sky-i
nepoch=150
nchains=20
version="v7_2"
release="1.0"
caps="NGC SGC"
slices="main highz low mid z1 z2 z3"
maps="all known"
table_name="ngal_eboss"
templates="/home/mehdi/data/templates/SDSS_WISE_HI_imageprop_nside${nside}.h5"
eboss_dir="/home/mehdi/data/eboss/data/${version}/"

do_prep=false
find_lr=false
find_st=false
find_ne=false
do_nnfit=true
do_swap=false
do_pk=false

#---- functions
function lrsetter() {
    if [ $1 = "main" ]
    then
        lr=0.01
    elif [ $1 = "highz" ]
    then
        lr=0.005
    fi
    echo $lr
}

#---- run
if [ "${do_prep}" = true ] # ~ 1 min
then
    for cap in ${caps}
    do
        dat=${eboss_dir}eBOSS_QSO_full_${cap}_${version}.dat.fits
        ran=${eboss_dir}eBOSS_QSO_full_${cap}_${version}.ran.fits    
        #du -h $dat $ran $templates

        output_path=${eboss_dir}${release}/${cap}/${nside}/

        python $prep -d ${dat} -r ${ran} -s ${templates} -o ${output_path} -n ${nside} -sl ${slices}        
    done
fi

if [ "${find_lr}" = true ]
then
    #---- neural net modeling
    for cap in ${caps}
    do
        for slice in ${slices}
        do
        
            input_dir=${eboss_dir}${release}/${cap}/${nside}/${slice}/      # output of 'do_prep'
            input_path=${input_dir}${table_name}_${slice}_${nside}.fits
            du -h ${input_path}

            for map in ${maps}
            do
                if [ ${map} = "all" ]
                then
                    axes=${axes_all[@]}
                elif [ ${map} = "known" ]
                then
                    axes=${axes_known[@]}
                else
                    exit 
                fi
                output_path=${input_dir}nn_pnnl_${map}/hp/
                echo ${output_path}
                python $nnfit -i ${input_path} -o ${output_path} -ax ${axes} -fl
            done
        done
    done
fi

if [ "${find_st}" = true ]
then
    #---- neural net modeling
    for cap in ${caps}
    do
        for slice in ${slices}
        do
            lr=$(lrsetter ${slice})
       
            input_dir=${eboss_dir}${release}/${cap}/${nside}/${slice}/      # output of 'do_prep'
            input_path=${input_dir}${table_name}_${slice}_${nside}.fits
            du -h ${input_path}


            for map in "known"
            do
                if [ ${map} = "all" ]
                then
                    axes=${axes_all[@]}
                elif [ ${map} = "known" ]
                then
                    axes=${axes_known[@]}
                else
                    exit 
                fi
                output_path=${input_dir}nn_pnnl_${map}/hp/
                echo ${output_path}
                python $nnfit -i ${input_path} -o ${output_path} -ax ${axes} \
                      -lr  ${lr} -fs 
            done
        done
    done
fi

if [ "${find_ne}" = true ]
then
    cap=NGC
    slice=highz

    input_dir=${eboss_dir}${release}/${cap}/${nside}/${slice}/      # output of 'do_prep'
    input_path=${input_dir}${table_name}_${slice}_${nside}.fits
    du -h ${input_path}

    lr=$(lrsetter ${slice})

    for map in "known"
    do
        if [ ${map} = "all" ]
        then
            axes=${axes_all[@]}
        elif [ ${map} = "known" ]
        then
            axes=${axes_known[@]}
        else
            exit 
        fi
        output_path=${input_dir}nn_pnnl_${map}/hp/
        echo ${output_path}
        python $nnfit -i ${input_path} -o ${output_path} -ax ${axes} \
        -lr ${lr} -ne 300
    done
fi

if [ "${do_nnfit}" = true ]
then
    #---- neural net modeling
    for cap in NGC #${caps}
    do
        for slice in highz #highz #${slices}
        do
            lr=$(lrsetter ${slice})
   
            input_dir=${eboss_dir}${release}/${cap}/${nside}/${slice}/      # output of 'do_prep'
            input_path=${input_dir}${table_name}_${slice}_${nside}.fits
            du -h ${input_path}


            for map in ${maps}
            do
                if [ ${map} = "all" ]
                then
                    axes=${axes_all[@]}
                elif [ ${map} = "known" ]
                then
                    axes=${axes_known[@]}
                else
                    exit 
                fi
                output_path=${input_dir}nn_pnnl_${map}
                echo ${output_path}
                python $nnfit -i ${input_path} -o ${output_path} \
                     -ax ${axes}  -lr ${lr} --nn_structure ${nn_structure[@]} \
                     -ne $nepoch -nc $nchains -k
            done
        done
    done
fi

conda deactivate
conda activate py3p6

if [ "${do_swap}" = true ]
then
    for cap in ${caps}
    do
        for map in ${maps}
        do
            python $swap -m ${map} -n ${nside} -s main highz -c ${cap}
        done
    done
fi


if [ "${do_pk}" = true ]
then
    for cap in ${caps}
    do
        # default
        input_dir=${eboss_dir}
        output_dir=${eboss_dir}${release}/measurements/spectra/
        
        dat=${input_dir}eBOSS_QSO_full_${cap}_v7_2.dat.fits
        ran=${dat/.dat./.ran.}
        
        for zrange in highz # main done
        do
            if [ ${zrange} = main ]
            then
                zlim='0.8 2.2'
            elif [ ${zrange} = highz ]
            then
                zlim='2.2 3.5'
            fi

            out=${output_dir}spectra_${cap}_knownsystot_mainhighz_512_v7_2_${zrange}.json
            du -h $dat $ran
            echo ${out} ${zlim}
            mpirun -np 8 python ${pk} -g $dat -r $ran -o $out --use_systot \
            --zlim ${zlim}
            
            for map in ${maps}
            do
                input_dir=${eboss_dir}${release}/catalogs/
                output_dir=${eboss_dir}${release}/measurements/spectra/

                for sample in mainhighz
                do
                    dat=${input_dir}eBOSS_QSO_full_${cap}_${map}_${sample}_${nside}_v7_2.dat.fits
                    ran=${dat/.dat./.ran.}
                    out=${output_dir}spectra_${cap}_${map}_${sample}_${nside}_v7_2_${zrange}.json
                    du -h $dat $ran
                    echo ${out}
                    mpirun -np 8 python ${pk} -g $dat -r $ran -o $out \
                    --use_systot --zlim ${zlim}
                done
            done
        done
    done
fi
