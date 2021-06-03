#---- environment variables and activation
. "/home/mehdi/miniconda3/etc/profile.d/conda.sh"
export NUMEXPR_MAX_THREADS=2
export PYTHONPATH=${HOME}/github/LSSutils:${HOME}/github/sysnetdev
conda activate sysnet


#---- path to the data
nside=512 # 512 or 256, use cmd line instead
l1=-1.0 # l1 regularization deactivated with <0
nn_structure=(4 20) 
# 'star_density', 'ebv', 'loghi', 'sky_g', 'sky_r', 'sky_i', 'sky_z', 
# 'depth_g_minus_ebv','depth_r_minus_ebv', 'depth_i_minus_ebv', 'depth_z_minus_ebv', 
# 'psf_g', 'psf_r', 'psf_i', 'psf_z',
#  'run', 'airmass'
axes_all=({0..16})
axes_known=(0 1 5 7 13) # star_density, ebv, depth-g, psf-i sky-i
batchsize=4098
nepoch=150
nchains=20
version="v7_2"
release="3.0" # or 3.0 (w/ Gaia)
caps="NGC SGC" # "NGC SGC"  # options are "NGC SGC"
slices="main highz" #"low mid" #"main highz" # options are "main highz low mid z1 z2 z3"
maps="known all" # options are "all known" but known is enough
samples="mainhighz" # options are lowmidhighz mainhighz / only 1: mainlinmse mainlinp mainmse mainhighz mainwocos mainstar mainstarg
table_name="ngal_eboss"
templates="/home/mehdi/data/templates/SDSS_WISE_HI_Gaia_imageprop_nside${nside}.h5"
templates2="/home/mehdi/data/templates/SDSS_WISE_HI_Gaia_imageprop_nside512.h5"
eboss_dir="/home/mehdi/data/eboss/data/${version}/"

do_prep=false
find_lr=false
find_st=false
find_ne=false
do_nnfit=false
do_swap=false
do_pk=false
do_nnbar=true
do_cl=true
do_xi=false
do_default=true


#---- path to the codes
prep=${HOME}/github/LSSutils/scripts/analysis/prepare_data_eboss.py
nnfit=${HOME}/github/sysnetdev/scripts/app.py
swap=${HOME}/github/LSSutils/scripts/analysis/swap_data_eboss.py
pk=${HOME}/github/LSSutils/scripts/analysis/run_pk.py
nnbar=${HOME}/github/LSSutils/scripts/analysis/run_nnbar_eboss.py
cl=${HOME}/github/LSSutils/scripts/analysis/run_cell_eboss.py
xi=${HOME}/github/LSSutils/scripts/analysis/run_xi.py


#---- functions
function get_datran(){
    if [ $1 = "main" ]
    then
        dat=${eboss_dir}eBOSS_QSO_${3}_${2}_v7_2.dat.fits
        ran=${dat/.dat./.ran.}
    elif [ $1 = "highz" ]
    then
        if [ $3 = "full" ]
        then
            dat=${eboss_dir}eBOSS_QSO_${3}_${2}_v7_2.dat.fits
            ran=${dat/.dat./.ran.}
        elif [ $3 = "clustering" ]
        then
            dat=${eboss_dir}eBOSS_QSOhiz_${3}_${2}_v7_2.dat.fits
            ran=${dat/.dat./.ran.}
        fi
    fi
    echo ${dat} ${ran}
}


function get_lr() {
    if [ $1 = "main" ]
    then
        if [ $nside = 512 ]
        then
            lr=0.01
        elif [ $nside = 256 ]
        then
            lr=0.01
        fi
    elif [ $1 = "highz" ]
    then
        if [ $nside = 512 ]
        then
            lr=0.05
        elif [ $nside = 256 ]
        then
            lr=0.01
        fi    
    elif [ $1 = "low" ]
    then
        if [ $nside = 512 ]
        then
            lr=0.01
        elif [ $nside = 256 ]
        then
            lr=0.02
        fi
    elif [ $1 = "mid" ]
    then
        if [ $nside = 512 ]
        then
            lr=0.01
        elif [ $nside = 256 ]
        then
            lr=0.02
        fi
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

function get_axes(){
    if [ $1 = "all" ]
    then
        axes=${axes_all[@]}
    elif [ $1 = "known" ]
    then
        axes=${axes_known[@]}
    else
        exit 
    fi   
    echo $axes
}

#---- run
if [ "${do_prep}" = true ] # ~ 1 min
then
    for cap in ${caps}
    do
        dat=${eboss_dir}eBOSS_QSO_full_${cap}_${version}.dat.fits
        ran=${eboss_dir}eBOSS_QSO_full_${cap}_${version}.ran.fits    
        du -h $dat $ran $templates

        output_path=${eboss_dir}${release}/${cap}/${nside}/
        echo ${output_path} ${nside} ${templates} ${slices}

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
                axes=$(get_axes ${map})

                output_path=${input_dir}nn_pnll_${map}/hp/
                echo ${output_path} ${axes}
                python $nnfit -i ${input_path} -o ${output_path} -ax ${axes} -bs ${batchsize} -fl
            done
        done
    done
fi

if [ "${find_st}" = true ]
then
    #---- neural net modeling
    for cap in NGC #${caps}
    do
        for slice in main #${slices}
        do
            lr=$(get_lr ${slice})
       
            input_dir=${eboss_dir}${release}/${cap}/${nside}/${slice}/      # output of 'do_prep'
            input_path=${input_dir}${table_name}_${slice}_${nside}.fits
            du -h ${input_path}


            for map in "known"
            do
                axes=$(get_axes ${map})
                output_path=${input_dir}nn_pnll_${map}/hp/
                echo ${output_path} $axes
                python $nnfit -i ${input_path} -o ${output_path} -ax ${axes} -lr  ${lr} -bs $batchsize -fs 
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

    lr=$(get_lr ${slice})

    for map in "known"
    do
        axes=$(get_axes ${map}) 
        output_path=${input_dir}nn_pnll_${map}/hp/
        echo ${output_path}
        python $nnfit -i ${input_path} -o ${output_path} -ax ${axes} -lr ${lr} -bs $batchsize -ne 300
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
                
                axes=$(get_axes ${map})
                output_path=${input_dir}nn_pnll_${map}
                echo ${output_path} ${lr} ${axes} ${slice}
                python $nnfit -i ${input_path} -o ${output_path} -ax ${axes} -lr ${lr} -bs $batchsize --nn_structure ${nn_structure[@]} -ne $nepoch -nc $nchains -k
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
            echo ${cap} ${map} ${release} ${nside} ${slices}
            python $swap -m ${map} -n ${nside} -s ${slices} -c ${cap} -v ${release} --method nn_pnll # 1+1 z
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

                read dat ran < <(get_datran $zrange $cap "clustering")

                out=${output_dir}spectra_${cap}_knownsystot_mainhighz_512_v7_2_${zrange}.json
                du -h $dat $ran
                echo ${out} ${zlim}
                mpirun -np 8 python ${pk} -g $dat -r $ran -o $out --use_systot --zlim ${zlim}


                out=${output_dir}spectra_${cap}_noweight_mainhighz_512_v7_2_${zrange}.json
                du -h $dat $ran
                echo ${out} ${zlim}
                mpirun -np 8 python ${pk} -g $dat -r $ran -o $out --zlim ${zlim}            
            fi
            continue
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
        for zrange in ${slices}
        do
            zlim=$(get_zlim ${zrange})
           
            if [ "${do_default}" = true ]
            then
                # default
                input_dir=${eboss_dir}
                output_dir=${eboss_dir}${release}/measurements/nnbar/

                read dat ran < <(get_datran $zrange $cap "full")

                out=${output_dir}nnbar_${cap}_noweight_mainhighz_512_v7_2_${zrange}_512.npy
                du -h $dat $ran
                echo ${out} ${zlim}
                mpirun -np 8 python ${nnbar} -d $dat -r $ran -o $out -t ${templates2} --zlim ${zlim}


                out=${output_dir}nnbar_${cap}_knownsystot_mainhighz_512_v7_2_${zrange}_512.npy
                du -h $dat $ran
                echo ${out} ${zlim}
                mpirun -np 8 python ${nnbar} -d $dat -r $ran -o $out -t ${templates2} --use_systot --zlim ${zlim}
            fi

            continue
            for map in ${maps}
            do
                input_dir=${eboss_dir}${release}/catalogs/
                output_dir=${eboss_dir}${release}/measurements/nnbar/

                for sample in ${samples}
                do
                    dat=${input_dir}eBOSS_QSO_full_${cap}_${map}_${sample}_${nside}_v7_2.dat.fits
                    ran=${dat/.dat./.ran.}
                    out=${output_dir}nnbar_${cap}_${map}_${sample}_${nside}_v7_2_${zrange}_512.npy
                    du -h $dat $ran
                    echo ${out}
                    mpirun -np 8 python ${nnbar} -d $dat -r $ran -o $out -t ${templates2} --use_systot --zlim ${zlim}
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

                read dat ran < <(get_datran $zrange $cap "full")

                out=${output_dir}cl_${cap}_noweight_mainhighz_512_v7_2_${zrange}_512.npy
                du -h $dat $ran
                echo ${out} ${zlim}
                mpirun -np 8 python ${cl} -d $dat -r $ran -o $out -t ${templates2} --zlim ${zlim}


                out=${output_dir}cl_${cap}_knownsystot_mainhighz_512_v7_2_${zrange}_512.npy
                du -h $dat $ran
                echo ${out} ${zlim}
                mpirun -np 8 python ${cl} -d $dat -r $ran -o $out -t ${templates2} --use_systot --zlim ${zlim}
            fi
            continue 
            for map in ${maps}
            do
                input_dir=${eboss_dir}${release}/catalogs/
                output_dir=${eboss_dir}${release}/measurements/cl/

                for sample in ${samples}
                do
                    dat=${input_dir}eBOSS_QSO_full_${cap}_${map}_${sample}_${nside}_v7_2.dat.fits
                    ran=${dat/.dat./.ran.}
                    out=${output_dir}cl_${cap}_${map}_${sample}_${nside}_v7_2_${zrange}_512.npy
                    du -h $dat $ran
                    echo ${out}
                    mpirun -np 8 python ${cl} -d $dat -r $ran -o $out -t ${templates2} --use_systot --zlim ${zlim}
                done
            done
        done
    done
fi

if [ "${do_xi}" = true ]
then
    for cap in ${caps}
    do
        for zrange in highz main
        do
            zlim=$(get_zlim ${zrange})
           
            if [ "${do_default}" = true ]
            then
                # default
                input_dir=${eboss_dir}
                output_dir=${eboss_dir}${release}/measurements/spectra/
                read dat ran < <(get_datran $zrange $cap "clustering")

                out=${output_dir}xi_${cap}_knownsystot_mainhighz_512_v7_2_${zrange}.json
                du -h $dat $ran
                echo ${out} ${zlim}

                mpirun -np 8 python ${xi} -g $dat -r $ran -o $out --use_systot --zlim ${zlim}


                out=${output_dir}xi_${cap}_noweight_mainhighz_512_v7_2_${zrange}.json
                du -h $dat $ran
                echo ${out} ${zlim}
                mpirun -np 8 python ${xi} -g $dat -r $ran -o $out --zlim ${zlim}            
            fi
            
            for map in ${maps}
            do
                input_dir=${eboss_dir}${release}/catalogs/
                output_dir=${eboss_dir}${release}/measurements/spectra/

                for sample in ${samples}
                do
                    dat=${input_dir}eBOSS_QSO_full_${cap}_${map}_${sample}_${nside}_v7_2.dat.fits
                    ran=${dat/.dat./.ran.}
                    out=${output_dir}xi_${cap}_${map}_${sample}_${nside}_v7_2_${zrange}.json
                    du -h $dat $ran
                    echo ${out} ${zlim}
                    mpirun -np 8 python ${xi} -g $dat -r $ran -o $out --use_systot --zlim ${zlim}
                done
            done
        done
    done
fi

