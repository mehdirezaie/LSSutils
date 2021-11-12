#!/bin/bash -l


#. "/Users/rezaie/anaconda3/etc/profile.d/conda.sh"
. "/home/mehdi/miniconda3/etc/profile.d/conda.sh"
export NUMEXPR_MAX_THREADS=2
export PYTHONPATH=${HOME}/github/LSSutils:${HOME}/github/sysnetdev
conda activate sysnet

makemaps=${HOME}/github/LSSutils/scripts/systematic_maps/make_sysmaps.py
combine=${HOME}/github/LSSutils/scripts/systematic_maps/combine_fits.py

dr=dr9
nside=256
name=dr9m

do_maps=false
do_comb=true


if [ $do_maps = true ]
then
    if [ $dr = "dr8" ]
    then
        input=/home/mehdi/data/templates/ccds/dr8/ccds-annotated-dr8_combined.fits
        output=/home/mehdi/data/templates/dr8/
        time python $makemaps -i $input -o $output -n $nside --name dr8m
    elif [ $dr = "dr9" ]
    then
        input=/home/mehdi/data/templates/ccds/dr9/ccds-annotated-dr9-combined.fits
        output=/home/mehdi/data/templates/dr9/
        time python $makemaps -i $input -o $output -n $nside --name $name
    fi
fi


if [ $do_comb = true ]
then
    input=/home/mehdi/data/templates/dr9/
    output=/home/mehdi/data/templates/dr9/
    time python $combine -n $name --nside $nside -i $input -o $output --add_galactic
fi
