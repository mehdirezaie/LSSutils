#!/bin/bash -l



. "/home/mehdi/miniconda3/etc/profile.d/conda.sh"
export NUMEXPR_MAX_THREADS=2
export PYTHONPATH=${HOME}/github/LSSutils:${HOME}/github/sysnetdev
conda activate sysnet

make_maps=${HOME}/github/LSSutils/scripts/systematic_maps/make_sysmaps.py

dr=$1

if [ $dr = "dr8" ]
then
    # enter the address to write the maps
    address=/Volumes/TimeMachine/data/DR8/
    
    #
    # First two loops took 1143m55.753s 
    # Aug 7, 2019: give up oversampling, combine new ccd with new cols
    # dr8_combined is probably enough
    # we create separate maps for debugging purposes
    # took 150 min

    for DR in dr8_combined decam-dr8
      do 
         #du -h /Volumes/TimeMachine/data/DR8/ccds/ccds-annotated-${DR}.fits
         time python $make_maps --survey DECaLS --dr ${DR} --localdir $address --nside 128
         time python $make_maps --survey DECaLS --dr ${DR} --localdir $address --nside 256
         time python $make_maps --survey DECaLS --dr ${DR} --localdir $address --nside 512
    done

    for DR in 90prime-new
      do 
         time python $make_maps --survey DECaLS --dr ${DR} --localdir $address --nside 128 --bands r g
         time python $make_maps --survey DECaLS --dr ${DR} --localdir $address --nside 256 --bands r g
         time python $make_maps --survey DECaLS --dr ${DR} --localdir $address --nside 512 --bands r g
    done

    for DR in mosaic-dr8
      do 
         time python $make_maps --survey DECaLS --dr ${DR} --localdir $address --nside 128 --bands z
         time python $make_maps --survey DECaLS --dr ${DR} --localdir $address --nside 256 --bands z
         time python $make_maps --survey DECaLS --dr ${DR} --localdir $address --nside 512 --bands z
    done
elif [ $dr = "dr9" ]
then
    address=/home/mehdi/data/templates/dr9/
    bands='r g z'
    for DR in dr9-combined
    do 
        for nside in 256
        do
            echo ${DR} ${dr} ${nside} ${bands}
            time python $make_maps --survey DECaLS --dr ${DR} --localdir $address --nside $nside --bands ${bands}
        done
    done
fi