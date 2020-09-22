#---- environment variables and activation
. "/home/mehdi/miniconda3/etc/profile.d/conda.sh"
export NUMEXPR_MAX_THREADS=2
export PYTHONPATH=${HOME}/github/LSSutils:${HOME}/github/sysnetdev
conda activate sysnet

#---- path to the codes
prep=/home/mehdi/github/LSSutils/scripts/analysis/prepare_data_eboss.py
nnfit=/home/mehdi/github/sysnetdev/scripts/app.py
swap=/home/mehdi/github/LSSutils/scripts/analysis/swap_data_eboss.py
pk=/home/mehdi/github/LSSutils/scripts/analysis/run_pk.py
nnbar=/home/mehdi/github/LSSutils/scripts/analysis/run_nnbar_eboss.py
cl=/home/mehdi/github/LSSutils/scripts/analysis/run_cell_eboss.py

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
nepoch=150
nchains=20
version="v7_2"
release="1.0"
caps="NGC SGC"  # options are "NGC SGC"
slices="mid" # options are "main highz low mid z1 z2 z3"
maps="all known" # options are "all known"
samples="lowmidhighz" # options are lowmidhighz mainhighz
table_name="ngal_eboss"
templates="/home/mehdi/data/templates/SDSS_WISE_HI_imageprop_nside${nside}.h5"
templates2="/home/mehdi/data/templates/SDSS_WISE_HI_imageprop_nside256.h5"
eboss_dir="/home/mehdi/data/eboss/data/${version}/"

do_prep=false
find_lr=false
find_st=false
find_ne=false
do_nnfit=false
do_swap=false
do_pk=false
do_nnbar=false
do_cl=false
do_default=false

#---- functions
function get_lr() {
    if [ $1 = "main" ]
    then
        lr=0.01
    elif [ $1 = "highz" ]
    then
        lr=0.05
    elif [ $1 = "low" ]
    then
        lr=0.01
    elif [ $1 = "mid" ]
    then
        lr=0.01        
    fi
    echo $lr
}

function get_zlim(){
    
    if [ $1 = main ]
    then
        zlim='0.8 2.2'
    elif [ $1 = highz ]
    then
        zlim='2.2 3.5'
    fi
    echo $zlim
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
    for cap in NGC #${caps}
    do
        for slice in low #${slices}
        do
            lr=$(get_lr ${slice})
       
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
                python $nnfit -i ${input_path} -o ${output_path} -ax ${axes} -lr  ${lr} -fs 
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

    lr=$(get_lr ${slice})

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
        python $nnfit -i ${input_path} -o ${output_path} -ax ${axes} -lr ${lr} -ne 300
    done
fi

if [ "${do_nnfit}" = true ]
then
    #---- neural net modeling
    for cap in ${caps}
    do
        for slice in ${slices}
        do
            lr=$(get_lr ${slice})

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
                python $nnfit -i ${input_path} -o ${output_path} -ax ${axes}  -lr ${lr} --nn_structure ${nn_structure[@]} -ne $nepoch -nc $nchains -k
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
            #python $swap -m ${map} -n ${nside} -s main highz -c ${cap} # 1+1 z
            python $swap -m ${map} -n ${nside} -s low mid highz -c ${cap}
        done
    done
fi


if [ "${do_pk}" = true ]
then
    for cap in ${caps}
    do
        for zrange in main highz
        do
            zlim=$(get_zlim ${zrange})
           
            if [ "${do_default}" = true ]
            then
                # default
                input_dir=${eboss_dir}
                output_dir=${eboss_dir}${release}/measurements/spectra/

                dat=${input_dir}eBOSS_QSO_full_${cap}_v7_2.dat.fits
                ran=${dat/.dat./.ran.}

                out=${output_dir}spectra_${cap}_knownsystot_mainhighz_512_v7_2_${zrange}.json
                du -h $dat $ran
                echo ${out} ${zlim}

                mpirun -np 8 python ${pk} -g $dat -r $ran -o $out --use_systot --zlim ${zlim}


                out=${output_dir}spectra_${cap}_noweight_mainhighz_512_v7_2_${zrange}.json
                du -h $dat $ran
                echo ${out} ${zlim}
                mpirun -np 8 python ${pk} -g $dat -r $ran -o $out --zlim ${zlim}            
            fi
            
            for map in ${maps}
            do
                input_dir=${eboss_dir}${release}/catalogs/
                output_dir=${eboss_dir}${release}/measurements/spectra/

                for sample in ${samples}
                do
                    dat=${input_dir}eBOSS_QSO_full_${cap}_${map}_${sample}_${nside}_v7_2.dat.fits
                    ran=${dat/.dat./.ran.}
                    out=${output_dir}spectra_${cap}_${map}_${sample}_${nside}_v7_2_${zrange}.json
                    du -h $dat $ran
                    echo ${out}
                    mpirun -np 8 python ${pk} -g $dat -r $ran -o $out --use_systot --zlim ${zlim}
                done
            done
        done
    done
fi

if [ "${do_nnbar}" = true ]
then
    for cap in ${caps}
    do
        for zrange in main highz
        do
            zlim=$(get_zlim ${zrange})
           
            if [ "${do_default}" = true ]
            then
                # default
                input_dir=${eboss_dir}
                output_dir=${eboss_dir}${release}/measurements/nnbar/

                dat=${input_dir}eBOSS_QSO_full_${cap}_v7_2.dat.fits
                ran=${dat/.dat./.ran.}

                out=${output_dir}nnbar_${cap}_noweight_mainhighz_512_v7_2_${zrange}.npy
                du -h $dat $ran
                echo ${out} ${zlim}


                mpirun -np 8 python ${nnbar} -d $dat -r $ran -o $out -t ${templates2} --zlim ${zlim}


                out=${output_dir}nnbar_${cap}_knownsystot_mainhighz_512_v7_2_${zrange}.npy
                du -h $dat $ran
                echo ${out} ${zlim}


                mpirun -np 8 python ${nnbar} -d $dat -r $ran -o $out -t ${templates} --use_systot --zlim ${zlim}
            fi

            
            for map in ${maps}
            do
                input_dir=${eboss_dir}${release}/catalogs/
                output_dir=${eboss_dir}${release}/measurements/nnbar/

                for sample in ${samples}
                do
                    dat=${input_dir}eBOSS_QSO_full_${cap}_${map}_${sample}_${nside}_v7_2.dat.fits
                    ran=${dat/.dat./.ran.}
                    out=${output_dir}nnbar_${cap}_${map}_${sample}_${nside}_v7_2_${zrange}.npy
                    du -h $dat $ran
                    echo ${out}
                    mpirun -np 8 python ${nnbar} -d $dat -r $ran -o $out -t ${templates} --use_systot --zlim ${zlim}
                done
            done
        done
    done
fi

if [ "${do_cl}" = true ]
then
    for cap in ${caps}
    do
        for zrange in main highz
        do
            zlim=$(get_zlim ${zrange})
           
            if [ "${do_default}" = true ]
            then
                # default
                input_dir=${eboss_dir}
                output_dir=${eboss_dir}${release}/measurements/cl/

                dat=${input_dir}eBOSS_QSO_full_${cap}_v7_2.dat.fits
                ran=${dat/.dat./.ran.}

                out=${output_dir}cl_${cap}_noweight_mainhighz_512_v7_2_${zrange}.npy
                du -h $dat $ran
                echo ${out} ${zlim}
                #mpirun -np 8 python ${cl} -d $dat -r $ran -o $out -t ${templates} \
                #          --zlim ${zlim}
                python ${cl} -d $dat -r $ran -o $out -t ${templates} --zlim ${zlim} --auto_only --nside 1024


                out=${output_dir}cl_${cap}_knownsystot_mainhighz_512_v7_2_${zrange}.npy
                du -h $dat $ran
                echo ${out} ${zlim}
                #mpirun -np 8 python ${cl} -d $dat -r $ran -o $out -t ${templates} \
                #        --use_systot --zlim ${zlim}
                python ${cl} -d $dat -r $ran -o $out -t ${templates} --use_systot --zlim ${zlim} --auto_only --nside 1024
            fi
            
            for map in ${maps}
            do
                input_dir=${eboss_dir}${release}/catalogs/
                output_dir=${eboss_dir}${release}/measurements/cl/

                for sample in ${samples}
                do
                    dat=${input_dir}eBOSS_QSO_full_${cap}_${map}_${sample}_${nside}_v7_2.dat.fits
                    ran=${dat/.dat./.ran.}
                    out=${output_dir}cl_${cap}_${map}_${sample}_${nside}_v7_2_${zrange}.npy
                    du -h $dat $ran
                    echo ${out}
                    #mpirun -np 8 python ${cl} -d $dat -r $ran -o $out -t ${templates} \
                    #--use_systot --zlim ${zlim}
                    python ${cl} -d $dat -r $ran -o $out -t ${templates} --use_systot --zlim ${zlim} --auto_only --nside 1024
                    
                done
            done
        done
    done
fi
