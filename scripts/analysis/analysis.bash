#!/bin/bash

source activate py3p6

# mask, sysmaps, Nran
SCRATCH=/Volumes/TimeMachine/data
mask=${SCRATCH}/mocks/mock.hpmask.dr7.fits
feat=${SCRATCH}/mocks/dr7mock-features.fits 
nran=${SCRATCH}/mocks/mock.fracNgalhpmap.fits
mock=${SCRATCH}/mocks/3dbox/
nsid=256
labl=.hp.${nsid}.fits
lab5=.hp.${nsid}.5.r.npy
# export lab5=.hp.${nsid}.5.s.npy


#
#    Uncontaminted mocks
#
#
#   FEB 11, 2019
#
# map mocks onto DR7
# create the mocks
# python mocks/mock_on_data.py /Volumes/TimeMachine/data/DR7_Feb10/sysmaps/DECaLS_DR7/nside256_oversamp4/features.fits  /Volumes/TimeMachine/data/mocks/mock.fracNgalhpmap.fits /Volumes/TimeMachine/data/mocks/dr7mock-features.fits /Volumes/TimeMachine/data/mocks/mock.hpmask.dr7.fits

# add DR7 attrs to mocks
#  mpirun --oversubscribe -np 4 python add_features-split.py --hpmap /Volumes/TimeMachine/data/mocks/3dbox/ --ext */*.hp.256.fits --features /Volumes/TimeMachine/data/mocks/dr7mock-features.fits --split r


#
#   FEB 11, 2019
#   Regression and clustering for uncontaminate mocks
#for i in $(seq -f "%03g" 1 100)
#do
  #
  # regression
#  echo ${i} 
#  glmp5=${mock}${i}/${i}${lab5}  
#  echo regression on $glmp5
#  # linear / quad fit
#  oudr=${mock}${i}/regression/mult-all/
#  python ./fit.py --input ${glmp5} --output ${oudr} --split
  # clustering
  # no weights
#  glmp=${mock}${i}/${i}${labl}  
#  echo clustering on $glmp
#  oudr=${mock}${i}/clustering/uni/
#  clnm=cl_uni
#  wmap=none
#  mpirun --oversubscribe -np 4 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr} 
  #
  # lin
#  oudr=${mock}${i}/clustering/mult-all/
#  clnm=cl_lin
#  wmap=${mock}${i}/regression/mult-all/lin-weights.hp256.fits
#  mpirun --oversubscribe -np 4 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr}
  #
  # quad
#  oudr=${mock}${i}/clustering/mult-all/
#  clnm=cl_quad
#  wmap=${mock}${i}/regression/mult-all/quad-weights.hp256.fits
#  mpirun --oversubscribe -np 4 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr}
#done


# Feb 12 -- fit ablation axes to DR7 data
#python fit.py --input /Volumes/TimeMachine/data/DR7_Feb10/eboss-ngc-dr7.1.cut.hp256.5.r.npy --output /Volumes/TimeMachine/data/DR7_Feb10/regression/mult-ab/ --split --ax 0 1 2 6 9 10 11 13 15 16


# contaminate using Linear model
# python contaminate.py --features /Volumes/TimeMachine/data/mocks/dr7mock-features.fits --bfitparams /Volumes/TimeMachine/data/DR7_Feb10/regression/mult-ab/regression_log.npy --mocksdir /Volumes/TimeMachine/data/mocks/3dbox/ --tag cablin --model lin

#mpirun --oversubscribe -np 4 python add_features-split.py --hpmap /Volumes/TimeMachine/data/mocks/3dbox/ --ext */cablin/*.cablin.hp.256.fits --features /Volumes/TimeMachine/data/mocks/dr7mock-features.fits --split r

#   Regression and clustering for contaminate mocks
#labs=cablin
#for i in $(seq -f "%03g" 1 100)
#do
  #
  # regression
#  echo ${i}
#  glmp5=${mock}${i}/${labs}/${i}.${labs}${lab5}
#  echo regression on $glmp5
#  # linear / quad fit
#  oudr=${mock}${i}/${labs}/regression/mult-all/
#  python ./fit.py --input ${glmp5} --output ${oudr} --split
 # clustering
  # no weights
#  glmp=${mock}${i}/${labs}/${i}.${labs}${labl}  
#  echo clustering on $glmp
#  oudr=${mock}${i}/${labs}/clustering/uni/
#  clnm=cl_uni
#  wmap=none
#  mpirun --oversubscribe -np 4 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr} 
#  #
  # lin
#  oudr=${mock}${i}/${labs}/clustering/mult-all/
#  clnm=cl_lin
#  wmap=${mock}${i}/${labs}/regression/mult-all/lin-weights.hp256.fits
#  mpirun --oversubscribe -np 4 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr}
  #
  # quad
#  oudr=${mock}${i}/${labs}/clustering/mult-all/
#  clnm=cl_quad
#  wmap=${mock}${i}/${labs}/regression/mult-all/quad-weights.hp256.fits
#  mpirun --oversubscribe -np 4 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr}
#done

labs=cablin
for i in $(seq -f "%03g" 1 100);
do
  echo 'running ... '${i};
  export glmp5=${mock}${i}/${i}${lab5}
  echo 'running ' $glmp5
  export oudr=${mock}${i}/ablation/r/
  time mpirun --oversubscribe -np 5 python run_ablationall.py --data ${glmp5} --output $oudr --index 0 17 --log ${i}.log;

  export glmp5=${mock}${i}/${labs}/${i}.${labs}${lab5}
  echo 'running ' $glmp5
  export oudr=${mock}${i}/${labs}/ablation/r/
  time mpirun --oversubscribe -np 5 python run_ablationall.py --data ${glmp5} --output $oudr --index 0 17 --log ${i}.log;
done

#  loop
# for i in $(seq -f "%05g" 10 15)
# printf -v j "%05d" $i
# for i in `jot -s " " -w '%03d' 10 `;do echo $i;done
# seq -w equal width sequence
#for i in `seq -w 1 100`
# 1..10 is already processed 



#   nn fit
#   export oudr=${mock}${i}/regression/nn-r/
#   time mpirun --oversubscribe -np 5 python ./validate.py --input ${glmp5} --output ${oudr}
#   export oudr=${mock}${i}/regression/nn-ab/
#   time mpirun --oversubscribe -np 5 python ./validate-ab.py --input ${glmp5} --output ${oudr} --ablationlog ${mock}${i}/ablation/r/${i}.log.npy 

#   #--------------------------------------------------
#   # cont.
#   #--------------------------------------------------
#   #export glmp5=${mock}${i}/${labs}/${i}.${labs}${lab5}
#   #echo regression on $glmp5

#   # lin/quad fit
#   #export oudr=${mock}${i}/${labs}/regression/mult-all/
#   #time python ./fit.py --input ${glmp5} --output ${oudr} --split

#   # nn fit
#   #export oudr=${mock}${i}/${labs}/regression/nn-s/
#   #export oudr=${mock}${i}/${labs}/regression/nn-r/
#   #time mpirun --oversubscribe -np 5 python ./validate.py --input ${glmp5} --output ${oudr}

#   #export oudr=${mock}${i}/${labs}/regression/nn-ab/
#   #time mpirun --oversubscribe -np 5 python ./validate-ab.py --input ${glmp5} --output ${oudr} --ablationlog ${mock}${i}/${labs}/ablation/r/${i}.log.npy 
 


#   #time mpirun --oversubscribe -np 5 python ./validate_all.py --input ${glmp5} --output ${oudr}
#   #export oudr=${mock}${i}/${labs}/regression/nn-r-cf/
#   #export oudr=${mock}${i}/${labs}/regression/nn-s-cf/
#   #time mpirun --oversubscribe -np 5 python ./validate-cf.py --input ${glmp5} --output ${oudr}
#   #export oudr=${mock}${i}/${labs}/regression/nn-s-cf-wosh/
#   #time mpirun --oversubscribe -np 5 python ./validate-cf-wosh.py --input ${glmp5} --output ${oudr}



  
#   #export oudr=${mock}${i}/clustering/nn-s/
#   #export wmap=${mock}${i}/regression/nn-s/nn-weights.hp256.fits
#   #export oudr=${mock}${i}/clustering/nn-r/
#   #export wmap=${mock}${i}/regression/nn-r/nn-weights.hp256.fits
#   export oudr=${mock}${i}/clustering/nn-ab-smooth/
#   export wmap=${mock}${i}/regression/nn-ab/nn-weights.hp256.fits
#   #export oudr=${mock}${i}/clustering/nn-s-cf-wosh/
#   #export wmap=${mock}${i}/regression/nn-s-cf-wosh/nn-weights.hp256.fits
#   export clnm=cl_nn  
#   time mpirun --oversubscribe -np 3 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr} --smooth
  
#   #export wmap=${mock}${i}/regression/nn-r-cf/nn-weights.hp256.fits
#   #export oudr=${mock}${i}/clustering/nn-r-cf/
#   #export wmap=${mock}${i}/regression/nn-s-cf/nn-weights.hp256.fits
#   #export oudr=${mock}${i}/clustering/nn-s-cf/
#   #export clnm=cl_nn
#   #mpirun --oversubscribe -np 3 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr}


#   #export oudr=${mock}${i}/clustering/nn-nov27/
#   #export clnm=cl_nn
#   #export wmap=${mock}${i}/regression/nn-nov27/nn-weights.hp256.fits
#   #mpirun -n 2 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr}

#   #export oudr=${mock}${i}/clustering/nn-dec06/
#   #export clnm=cl_nn
#   #export wmap=${mock}${i}/regression/nn-dec06/nn-weights.hp256.fits
#   #mpirun --oversubscribe -n 3 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr}

#   #export oudr=${mock}${i}/clustering/nn-jan15/
#   #export clnm=cl_nn
#   #export wmap=${mock}${i}/regression/nn-jan15/nn-weights.hp256.fits
#   #mpirun --oversubscribe -n 3 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr}

#   #export oudr=${mock}${i}/clustering/quad-split/
#   #export clnm=cl_quad
#   #export wmap=${mock}${i}/regression/multivar-all-split/quad-weight.hp256.fits
#   #mpirun --oversubscribe -n 3 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr}

#   #export oudr=${mock}${i}/clustering/lin-one/
#   #export clnm=cl_lin
#   #export wmap=${mock}${i}/regression/multivar-one/lin-weight.hp256.fits
#   #mpirun --oversubscribe -n 3 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr}

#   #export oudr=${mock}${i}/clustering/quad-one/
#   #export clnm=cl_quad
#   #export wmap=${mock}${i}/regression/multivar-one/quad-weight.hp256.fits
#   #mpirun --oversubscribe -n 3 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr}
#   #--------------------------------------------------
#   # cont.
#   #--------------------------------------------------
#   export glmp=${mock}${i}/${labs}/${i}.${labs}${labl}
#   echo clustering on $glmp

#   #export oudr=${mock}${i}/${labs}/clustering/uni/
#   #export clnm=cl_uni
#   #export wmap=none
#   #mpirun --oversubscribe -np 3 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr} --verbose
  
#   #export oudr=${mock}${i}/${labs}/clustering/lin-one/
#   #export clnm=cl_lin
#   #export wmap=${mock}${i}/${labs}/regression/multivar-one/lin-weight.hp256.fits
#   #mpirun --oversubscribe -n 3 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr}
#   #
#   #
#   #export oudr=${mock}${i}/${labs}/clustering/mult-all/
#   #export clnm=cl_lin
#   #export wmap=${mock}${i}/${labs}/regression/mult-all/lin-weights.hp256.fits
#   #mpirun --oversubscribe -np 3 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr} --verbose

#   #export oudr=${mock}${i}/${labs}/clustering/mult-all/
#   #export clnm=cl_quad
#   #export wmap=${mock}${i}/${labs}/regression/mult-all/quad-weights.hp256.fits
#   #mpirun --oversubscribe -np 3 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr} --verbose

#   #export oudr=${mock}${i}/${labs}/clustering/quad-one/
#   #export clnm=cl_quad
#   #export wmap=${mock}${i}/${labs}/regression/multivar-one/quad-weight.hp256.fits
#   #mpirun --oversubscribe -n 3 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr}
  
  
#   #export oudr=${mock}${i}/${labs}/clustering/quad-split/
#   #export clnm=cl_quad
#   #export wmap=${mock}${i}/${labs}/regression/multivar-all-split/quad-weight.hp256.fits
#   #mpirun --oversubscribe -n 3 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr}

#   #export oudr=${mock}${i}/${labs}/clustering/nn-dec06/
#   #export clnm=cl_nn
#   #export wmap=${mock}${i}/${labs}/regression/nn-dec06/nn-weights.hp256.fits
#   #mpirun --oversubscribe -n 3 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr}

  
#   export clnm=cl_nn
#   #export oudr=${mock}${i}/${labs}/clustering/nn-s/  
#   #export wmap=${mock}${i}/${labs}/regression/nn-s/nn-weights.hp256.fits
#   #export oudr=${mock}${i}/${labs}/clustering/nn-r/  
#   #export wmap=${mock}${i}/${labs}/regression/nn-r/nn-weights.hp256.fits
#   export oudr=${mock}${i}/${labs}/clustering/nn-ab-smooth/  
#   export wmap=${mock}${i}/${labs}/regression/nn-ab/nn-weights.hp256.fits
#   #export oudr=${mock}${i}/${labs}/clustering/nn-s-cf-wosh/  
#   #export wmap=${mock}${i}/${labs}/regression/nn-s-cf-wosh/nn-weights.hp256.fits
#   time mpirun --oversubscribe -n 3 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr} --smooth

#   #export clnm=cl_nn
#   #export oudr=${mock}${i}/${labs}/clustering/nn-r-cf/
#   #export wmap=${mock}${i}/${labs}/regression/nn-r-cf/nn-weights.hp256.fits
#   #export oudr=${mock}${i}/${labs}/clustering/nn-s-cf/
#   #export wmap=${mock}${i}/${labs}/regression/nn-s-cf/nn-weights.hp256.fits 
#   #mpirun --oversubscribe -n 3 python ./run_pipeline.py --galmap ${glmp} --ranmap ${nran} --photattrs ${feat} --wmap ${wmap} --mask ${mask} --clfile ${clnm} --oudir ${oudr}
