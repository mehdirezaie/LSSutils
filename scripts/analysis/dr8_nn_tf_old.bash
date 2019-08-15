#!/bin/bash
source ~/.bash_profile
conda activate py3p6

# codes
ablation=/Users/rezaie/github/LSSutils/scripts/analysis/ablation_tf_old.py
multfit=/Users/rezaie/github/LSSutils/scripts/analysis/mult_fit.py
nnfit=/Users/rezaie/github/LSSutils/scripts/analysis/nn_fit_tf_old.py
docl=/Users/rezaie/github/LSSutils/scripts/analysis/run_pipeline.py
# docont=/Users/rezaie/github/SYSNet/src/contaminate.py

# ablation and regression 221 min
# clustering took 700 min
for gal in elg lrg
do 
   nside=256
   lmax=512
   glmp=/Volumes/TimeMachine/data/DR8/alternative/${gal}_gal${nside}.fits
   glmp5=/Volumes/TimeMachine/data/DR8/alternative/dr8_${gal}_${nside}_5r.npy
   drfeat=/Volumes/TimeMachine/data/DR8/alternative/dr8_${gal}_${nside}.fits
   rnmp=/Volumes/TimeMachine/data/DR8/alternative/frac_${gal}_${nside}.fits  # frac_elg_256.fits
   oudr_ab=/Volumes/TimeMachine/data/DR8/alternative/results_${gal}/ablation/
   oudr_r=/Volumes/TimeMachine/data/DR8/alternative/results_${gal}/regression/
   oudr_c=/Volumes/TimeMachine/data/DR8/alternative/results_${gal}/clustering/
   maskc=/Volumes/TimeMachine/data/DR8/alternative/mask_${gal}_${nside}.cut.fits    # remove pixels with extreme weights
   maskclog=/Volumes/TimeMachine/data/DR8/alternative/mask_${gal}_${nside}.cut.log
   mult1=mult_all
   mult2=mult_depz
   mult3=mult_ab
   mult4=mult_f
   log_ab=dr8.log
   nn1=nn_ab
   nn3=nn_p
   ## axfit MUST change depending on the imaging maps
   axfit='0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20'
   
    ## REGRESSION
    #echo 'ablation on ' $gal 'with ' $axfit 
    #for rk in 0 1 2 3 4
    #do
    # echo $rk 'on ' $glmp5
    # mpirun --oversubscribe -np 5 python $ablation --data $glmp5 --output $oudr_ab --log $log_ab --rank $rk --axfit $axfit
    #done
    #echo 'regression on ' $gal
    #python $multfit --input $glmp5 --output ${oudr_r}${mult1}/ --split --nside $nside --axfit $axfit # we don't do lin regression
    #mpirun --oversubscribe -np 5 python $nnfit --input $glmp5 --output ${oudr_r}${nn1}/ --ablog ${oudr_ab}${log_ab} --nside $nside
    #mpirun --oversubscribe -np 5 python $nnfit --input $glmp5 --output ${oudr_r}${nn3}/ --nside $nside --axfit $axfit

     ## remove the extreme weight pixels
     #python make_common_mask-data.py /Volumes/TimeMachine/data/DR8/alternative/mask_${gal}_${nside}.fits /Volumes/TimeMachine/data/DR8/alternative/mask_${gal}_${nside}.cut.fits ${oudr_r}*/*-weights.hp${nside}.fits > $maskclog

     ##Clustering
     ## no correction, linear, quadratic
    for wname in uni lin quad
    do
      wmap=${oudr_r}${mult1}/${wname}-weights.hp${nside}.fits
      mpirun --oversubscribe -np 4 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap $wmap --clfile cl_$wname --nnbar nnbar_$wname --corfile xi_$wname --nside $nside --lmax $lmax --axfit $axfit 
    done
    # nn w ablation, nn plain
    for nni in $nn1 $nn3
    do
      wmap=${oudr_r}${nni}/nn-weights.hp${nside}.fits
      mpirun --oversubscribe -np 4 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap $wmap --nnbar nnbar_$nni --clfile cl_$nni  --corfile xi_$nni --nside $nside --lmax $lmax --axfit $axfit
    done
    #
    # auto C_l for systematics
    mpirun --oversubscribe -np 4 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap none --clsys cl_sys --corsys xi_sys --nside ${nside} --lmax $lmax --axfit $axfit

done
#
