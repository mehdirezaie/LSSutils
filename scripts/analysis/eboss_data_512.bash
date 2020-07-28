#---- environment variables and activation
. "/home/mehdi/miniconda3/etc/profile.d/conda.sh"

export PYTHONPATH=${HOME}/github/LSSutils:${HOME}/github/sysnetdev
conda activate sysnet

#---- path to the codes
prep=/home/mehdi/github/LSSutils/scripts/analysis/prepare_data_eboss.py
nnfit=/home/mehdi/github/sysnetdev/scripts/app.py

#---- path to the data
nside=512
l1=-1.0 # l1 regularization deactivated with <0
nn_structure=(4 20) 
# 'star_density', 'ebv', 'loghi', 'sky_g', 'sky_r', 'sky_i', 'sky_z', 
# 'depth_g_minus_ebv','depth_r_minus_ebv', 'depth_i_minus_ebv', 'depth_z_minus_ebv', 
# 'psf_g', 'psf_r', 'psf_i', 'psf_z',
#  'run', 'airmass'
axes_all=({0..16})
axes_known=(1 5 7 13) # ebv, depth-g, psf-i sky-i
nepoch=50
nchains=15
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
find_ne=true
do_nnfit=false

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

lr=0.01

if [ "${find_st}" = true ]
then
    #---- neural net modeling
    for cap in ${caps}
    do
        for slice in ${slices}
        do
        
            input_dir=${eboss_dir}${release}/${cap}/${nside}/${slice}/      # output of 'do_prep'
            input_path=${input_dir}${table_name}_${slice}_${nside}.fits
            du -h ${input_path}


            for map in "all"
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
    slice=main

    input_dir=${eboss_dir}${release}/${cap}/${nside}/${slice}/      # output of 'do_prep'
    input_path=${input_dir}${table_name}_${slice}_${nside}.fits
    du -h ${input_path}


    for map in "all"
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
                output_path=${input_dir}nn_pnnl_${map}
                echo ${output_path}
                python $nnfit -i ${input_path} -o ${output_path} \
                     -ax ${axes}  -lr ${lr} --nn_structure ${nn_structure[@]} \
                     -ne $nepoch -nc $nchains -k
            done
        done
    done
fi
