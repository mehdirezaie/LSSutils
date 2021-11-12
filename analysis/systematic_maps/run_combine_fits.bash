#!/bin/bash -l

source ~/.bash_profile
conda activate py3p6

# enter the address to write the maps
address=/Volumes/TimeMachine/data/DR8/
combine_fits=/Users/rezaie/github/LSSutils/scripts/systematic_maps/combine_fits.py


for nside in 128 256 512
do 
    time python $combine_fits --paths ${address}DECaLS_dr8_combined/nside${nside}_oversamp1/DECaLS_dr8_combined_band_* --nside $nside --tohdf ${address}dr8_combined${nside}.h5 --figs ${address}dr8_combined${nside}.png --mkwy
done 
