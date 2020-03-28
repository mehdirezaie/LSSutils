#!/bin/bash
if [ -f "~/.bash_profile" ]
then
	source ~/.bash_profile
fi

if [ $HOST == "lakme" ]
then 
    eval "$(/home/mehdi/miniconda3/bin/conda shell.bash hook)"
    DATA=${HOME}/data
fi
conda activate py3p6

# codes

<<<<<<< HEAD
# ablation and regression 221 min
# clustering took 700 min
# Oct 8: ELG NGC/SGC split
# Oct 9: run w/o MJD
# regression + nnbar 115 min
for gal in elg
=======
ablation=${HOME}/github/LSSutils/scripts/analysis/ablation_tf_old.py
multfit=${HOME}/github/LSSutils/scripts/analysis/mult_fit.py
nnfit=${HOME}/github/LSSutils/scripts/analysis/nn_fit_tf_old.py
docl=${HOME}/github/LSSutils/scripts/analysis/run_pipeline.py
elnet=${HOME}/github/LSSutils/scripts/analysis/elnet_fit.py
pk=${HOME}/github/LSSutils/scripts/analysis/run_pk.py



path2data=/home/mehdi/data/alternative/

# ablation + regression took 402 min
for gal in elg lrg 
>>>>>>> c91e03440d0e302e06d849921964c4027e56dd7b
do 
   for cap in decaln decals bmzls
   do
       nside=256
       lmax=512
       glmp=${path2data}${gal}_gal${nside}.fits
       glmp5=${path2data}dr8_${gal}_${cap}_${nside}_5r.npy # dr8_elg_bmzls_256_5r.npy
       drfeat=${path2data}dr8_${gal}_${cap}_${nside}.fits
       rnmp=${path2data}frac_${gal}_${nside}.fits  # frac_elg_256.fits
       oudr_ab=${path2data}results_${gal}/ablation_${cap}/
       oudr_r=${path2data}results_${gal}/regression_${cap}/
       oudr_c=${path2data}results_${gal}/clustering_${cap}/
       
       maskc=${path2data}mask_${gal}_${cap}_${nside}.fits    # remove pixels with extreme 
       
       maskclog=${path2data}mask_${gal}_${nside}.cut.log
       mult1=mult_all
       mult2=mult_depz
       mult3=mult_ab
       mult4=mult_f
       log_ab=dr8.log
       nn1=nn_ab
       nn3=nn_p
       nn4=nn_p_womjd
       ## axfit MUST change depending on the imaging maps
<<<<<<< HEAD
       #axfit='0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20'
       axfit='0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17'

=======
       axfit='0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20'
       
       
       #du -h $glmp5 
      # du -h $glmp
      # du -h $drfeat
      # du -h $rnmp
      # du -h $maskc
       
>>>>>>> c91e03440d0e302e06d849921964c4027e56dd7b
        ## REGRESSION
        #echo 'ablation on ' $gal $cap 'with ' $axfit 
        #for rk in 0 1 2 3 4
        #do
        # echo $rk 'on ' $glmp5
        # mpirun -np 5 python $ablation --data $glmp5 --output $oudr_ab --log $log_ab --rank $rk --axfit $axfit
        #done
        #echo 'regression on ' $gal $cap
        #python $multfit --input $glmp5 --output ${oudr_r}${mult1}/ --split --nside $nside --axfit $axfit # we don't do lin regression
<<<<<<< HEAD
        #mpirun --oversubscribe -np 5 python $nnfit --input $glmp5 --output ${oudr_r}${nn1}/ --ablog ${oudr_ab}${log_ab} --nside $nside
        #mpirun --oversubscribe -np 5 python $nnfit --input $glmp5 --output ${oudr_r}${nn3}/ --nside $nside --axfit $axfit
        mpirun -np 5 python $nnfit --input $glmp5 --output ${oudr_r}${nn4}/ --nside $nside --axfit $axfit
=======
        #mpirun -np 5 python $nnfit --input $glmp5 --output ${oudr_r}${nn1}/ --ablog ${oudr_ab}${log_ab} --nside $nside
        #mpirun -np 5 python $nnfit --input $glmp5 --output ${oudr_r}${nn3}/ --nside $nside --axfit $axfit
>>>>>>> c91e03440d0e302e06d849921964c4027e56dd7b

        ## remove the extreme weight pixels
        #python make_common_mask-data.py /Volumes/TimeMachine/data/DR8/alternative/mask_${gal}_${nside}.fits /Volumes/TimeMachine/data/DR8/alternative/mask_${gal}_${nside}.cut.fits ${oudr_r}*/*-weights.hp${nside}.fits > $maskclog

<<<<<<< HEAD
        ##Clustering
        ## no correction, linear, quadratic
        #for wname in uni lin quad
        #do
        #  wmap=${oudr_r}${mult1}/${wname}-weights.hp${nside}.fits
        #  python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap $wmap --nnbar nnbar_${wname}_${cap} --nside $nside --axfit $axfit 
        #done
        # nn w ablation, nn plain
        #for nni in $nn1 $nn3
        for nni in $nn4
        do
          wmap=${oudr_r}${nni}/nn-weights.hp${nside}.fits
          python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap $wmap --nnbar nnbar_${nni}_${cap} --nside $nside --axfit $axfit
        done
=======
         ##Clustering
         ## no correction, linear, quadratic
        #for wname in uni lin quad
#         do
#           wmap=${oudr_r}${mult1}/${wname}-weights.hp${nside}.fits
#           python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap $wmap --nnbar nnbar_${wname}_${cap} --nside $nside --axfit $axfit 
#         done
#         # nn w ablation, nn plain
        #for nni in uni $nn1 $nn3
        #do
        #  wmap=${oudr_r}${nni}/nn-weights.hp${nside}.fits
        #  mpirun -np 8 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap $wmap --nnbar nnbar_${gal}_${nni}_${cap} --clfile cl_${gal}_${nni}_${cap} --corfile xi_${gal}_${nni}_${cap} --nside $nside --axfit $axfit
        #done
>>>>>>> c91e03440d0e302e06d849921964c4027e56dd7b
        #
        # auto C_l for systematics
        #mpirun -np 8 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap none --clsys cl_sys_${gal}_${cap} --corsys xi_sys_${gal}_${cap} --nside ${nside} --lmax $lmax --axfit $axfit

<<<<<<< HEAD
=======

        # maps trained on full sky applied on different chunks
         oudr_r=${path2data}results_${gal}/regression/
         nni=nn_p
        
         wmap=${oudr_r}${nni}/nn-weights.hp${nside}.fits
         mpirun -np 8 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap $wmap --nnbar nnbar_${gal}_${nni}_all_${cap} --clfile cl_${gal}_${nni}_all_${cap} --corfile xi_${gal}_${nni}_all_${cap} --nside $nside --axfit $axfit        
   
>>>>>>> c91e03440d0e302e06d849921964c4027e56dd7b
   done
done






# ablation and regression 221 min
# clustering took 700 min
# Oct 8: ELG NGC/SGC split
# for gal in elg
# do 
#    for cap in ngc sgc
#    do
#        nside=256
#        lmax=512
#        glmp=/Volumes/TimeMachine/data/DR8/alternative/${gal}_gal${nside}.fits
#        glmp5=/Volumes/TimeMachine/data/DR8/alternative/dr8_${gal}_${nside}_5r.npy
#        drfeat=/Volumes/TimeMachine/data/DR8/alternative/dr8_${gal}_${nside}.fits
#        rnmp=/Volumes/TimeMachine/data/DR8/alternative/frac_${gal}_${nside}.fits  # frac_elg_256.fits
#        oudr_ab=/Volumes/TimeMachine/data/DR8/alternative/results_${gal}/ablation/
#        oudr_r=/Volumes/TimeMachine/data/DR8/alternative/results_${gal}/regression/
#        oudr_c=/Volumes/TimeMachine/data/DR8/alternative/results_${gal}/clustering/
#        maskc=/Volumes/TimeMachine/data/DR8/alternative/mask_${gal}_${cap}_${nside}.cut.fits    # remove pixels with extreme 
#        maskclog=/Volumes/TimeMachine/data/DR8/alternative/mask_${gal}_${nside}.cut.log
#        mult1=mult_all
#        mult2=mult_depz
#        mult3=mult_ab
#        mult4=mult_f
#        log_ab=dr8.log
#        nn1=nn_ab
#        nn3=nn_p
#        ## axfit MUST change depending on the imaging maps
#        axfit='0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20'

#         ## REGRESSION
#         #echo 'ablation on ' $gal 'with ' $axfit 
#         #for rk in 0 1 2 3 4
#         #do
#         # echo $rk 'on ' $glmp5
#         # mpirun --oversubscribe -np 5 python $ablation --data $glmp5 --output $oudr_ab --log $log_ab --rank $rk --axfit $axfit
#         #done
#         #echo 'regression on ' $gal
#         #python $multfit --input $glmp5 --output ${oudr_r}${mult1}/ --split --nside $nside --axfit $axfit # we don't do lin regression
#         #mpirun --oversubscribe -np 5 python $nnfit --input $glmp5 --output ${oudr_r}${nn1}/ --ablog ${oudr_ab}${log_ab} --nside $nside
#         #mpirun --oversubscribe -np 5 python $nnfit --input $glmp5 --output ${oudr_r}${nn3}/ --nside $nside --axfit $axfit

#          ## remove the extreme weight pixels
#          #python make_common_mask-data.py /Volumes/TimeMachine/data/DR8/alternative/mask_${gal}_${nside}.fits /Volumes/TimeMachine/data/DR8/alternative/mask_${gal}_${nside}.cut.fits ${oudr_r}*/*-weights.hp${nside}.fits > $maskclog

#          ##Clustering
#          ## no correction, linear, quadratic
#         for wname in uni lin quad
#         do
#           wmap=${oudr_r}${mult1}/${wname}-weights.hp${nside}.fits
#           python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap $wmap --nnbar nnbar_${wname}_${cap} --nside $nside --axfit $axfit 
#         done
#         # nn w ablation, nn plain
#         for nni in $nn1 $nn3
#         do
#           wmap=${oudr_r}${nni}/nn-weights.hp${nside}.fits
#           python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap $wmap --nnbar nnbar_${nni}_${cap} --nside $nside --axfit $axfit
#         done
#         #
#         # auto C_l for systematics
#         #mpirun --oversubscribe -np 4 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap none --clsys cl_sys --corsys xi_sys --nside ${nside} --lmax $lmax --axfit $axfit

   
#    done
# done
#


# for gal in elg lrg
# do 
#    nside=256
#    lmax=512
#    glmp=/Volumes/TimeMachine/data/DR8/alternative/${gal}_gal${nside}.fits
#    glmp5=/Volumes/TimeMachine/data/DR8/alternative/dr8_${gal}_${nside}_5r.npy
#    drfeat=/Volumes/TimeMachine/data/DR8/alternative/dr8_${gal}_${nside}.fits
#    rnmp=/Volumes/TimeMachine/data/DR8/alternative/frac_${gal}_${nside}.fits  # frac_elg_256.fits
#    oudr_ab=/Volumes/TimeMachine/data/DR8/alternative/results_${gal}/ablation/
#    oudr_r=/Volumes/TimeMachine/data/DR8/alternative/results_${gal}/regression/
#    oudr_c=/Volumes/TimeMachine/data/DR8/alternative/results_${gal}/clustering/
#    maskc=/Volumes/TimeMachine/data/DR8/alternative/mask_${gal}_${nside}.cut.fits    # remove pixels with extreme weights
#    maskclog=/Volumes/TimeMachine/data/DR8/alternative/mask_${gal}_${nside}.cut.log
#    mult1=mult_all
#    mult2=mult_depz
#    mult3=mult_ab
#    mult4=mult_f
#    log_ab=dr8.log
#    nn1=nn_ab
#    nn3=nn_p
#    ## axfit MUST change depending on the imaging maps
#    axfit='0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20'
   
#     ## REGRESSION
#     #echo 'ablation on ' $gal 'with ' $axfit 
#     #for rk in 0 1 2 3 4
#     #do
#     # echo $rk 'on ' $glmp5
#     # mpirun --oversubscribe -np 5 python $ablation --data $glmp5 --output $oudr_ab --log $log_ab --rank $rk --axfit $axfit
#     #done
#     #echo 'regression on ' $gal
#     #python $multfit --input $glmp5 --output ${oudr_r}${mult1}/ --split --nside $nside --axfit $axfit # we don't do lin regression
#     #mpirun --oversubscribe -np 5 python $nnfit --input $glmp5 --output ${oudr_r}${nn1}/ --ablog ${oudr_ab}${log_ab} --nside $nside
#     #mpirun --oversubscribe -np 5 python $nnfit --input $glmp5 --output ${oudr_r}${nn3}/ --nside $nside --axfit $axfit

#      ## remove the extreme weight pixels
#      #python make_common_mask-data.py /Volumes/TimeMachine/data/DR8/alternative/mask_${gal}_${nside}.fits /Volumes/TimeMachine/data/DR8/alternative/mask_${gal}_${nside}.cut.fits ${oudr_r}*/*-weights.hp${nside}.fits > $maskclog

#      ##Clustering
#      ## no correction, linear, quadratic
#     for wname in uni lin quad
#     do
#       wmap=${oudr_r}${mult1}/${wname}-weights.hp${nside}.fits
#       mpirun --oversubscribe -np 4 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap $wmap --clfile cl_$wname --nnbar nnbar_$wname --corfile xi_$wname --nside $nside --lmax $lmax --axfit $axfit 
#     done
#     # nn w ablation, nn plain
#     for nni in $nn1 $nn3
#     do
#       wmap=${oudr_r}${nni}/nn-weights.hp${nside}.fits
#       mpirun --oversubscribe -np 4 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap $wmap --nnbar nnbar_$nni --clfile cl_$nni  --corfile xi_$nni --nside $nside --lmax $lmax --axfit $axfit
#     done
#     #
#     # auto C_l for systematics
#     mpirun --oversubscribe -np 4 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap none --clsys cl_sys --corsys xi_sys --nside ${nside} --lmax $lmax --axfit $axfit

# done
# #
