#!/bin/bash

# RUN :
# (py3p6) mehdi@lakme:~/github/desimocks> bash step1.bash txt2fits ~/data/desimocks/UNIT_DESI_Shadab_HOD_snap97_ELG_v0.txt  ~/data/desimocks/UNIT_DESI_Shadab_HOD_snap97_ELG_v0.fits 1000. UNIT
#189M    /home/mehdi/data/desimocks/UNIT_DESI_Shadab_HOD_snap97_ELG_v0.txt
#txt2fits on  /home/mehdi/data/desimocks/UNIT_DESI_Shadab_HOD_snap97_ELG_v0.txt
#Going to read /home/mehdi/data/desimocks/UNIT_DESI_Shadab_HOD_snap97_ELG_v0.txt  and write /home/mehdi/data/desimocks/UNIT_DESI_Shadab_HOD_snap97_ELG_v0.fits
#done in 24.154889822006226 sec


#
# Mehdi Rezaie, mr095415@ohio.edu
# 
#
# activate environment


eval "$(/home/mehdi/miniconda3/bin/conda shell.bash hook)"
conda activate py3p6

#
#   CODES
#
# txt2fits for nbodykit
Txt2fits=/home/mehdi/github/desimocks/txt2fits.py

# DD(s,mu)
DDsmu=/home/mehdi/github/LSSutils/scripts/analysis/Box2pc.py



#
# DATA path
#
ACTION=$1
PATH2DATA=$HOME/data/desimocks/
INPUTTXT1=${PATH2DATA}UNIT_DESI_Shadab_HOD_snap97_ELG_v0.txt
INPUTTXT2=${PATH2DATA}DESI_ELG_z0.76_catalogue.dat
INPUTFIT1=${PATH2DATA}UNIT_DESI_Shadab_HOD_snap97_ELG_v0.fits
INPUTFIT2=${PATH2DATA}DESI_ELG_z0.76_catalogue.fits



if [ $ACTION == "txt2fits" ]
then
    du -h $INPUTTXT1 $INPUTTXT2
	echo $1 'on ' $INPUTTXT1

    time python $Txt2fits $INPUTTXT1 $INPUTFIT1
    time python $Txt2fits $INPUTTXT2 $INPUTFIT2

elif [ $ACTION == "ddrmu" ] || [ $ACTION == "ddsmu" ]
then

    class=ELG
    ext=v0
    LASTNAME=REZAIE
    BINNING=lin
    ESTIMATOR1=xi2D
    ESTIMATOR2=xil
    version=1

    # mock 1 
    MOCKNAME=UNIT_${ACTION}
    BOX=1000.
    out1=${PATH2DATA}${ESTIMATOR1}_${BINNING}_${LASTNAME}_${MOCKNAME}_${version}.txt
    out2=${PATH2DATA}${ESTIMATOR2}_${BINNING}_${LASTNAME}_${MOCKNAME}_${version}.txt
    echo $1 'on ' $INPUTFIT1 'produce '$out1 $out2
	du -h $INPUTFIT1
    time mpirun -np 8 python $DDsmu $INPUTFIT1 $out1 $out2 $ACTION $BOX

    # mock 2
    MOCKNAME=PMILL_${ACTION}
    BOX=542.16
    out1=${PATH2DATA}${ESTIMATOR1}_${BINNING}_${LASTNAME}_${MOCKNAME}_${version}.txt
    out2=${PATH2DATA}${ESTIMATOR2}_${BINNING}_${LASTNAME}_${MOCKNAME}_${version}.txt
    echo $1 'on ' $INPUTFIT2 'produce '$out1 $out2
	du -h $INPUTFIT2
    time mpirun -np 8 python $DDsmu $INPUTFIT2 $out1 $out2 $ACTION $BOX

fi



#
# STEP 1
#
#for class in ELG LRG QSO
#do 
#	echo $class
	#
        # TXT to FITS
	#
	#inputTXT=${PATH2DATA}UNIT_DESI_Shadab_HOD_snap97_${class}_v0.txt
	#python $Txt2fits $inputTXT
	#du -h ${PATH2DATA}UNIT_DESI_Shadab_HOD_snap97_${class}_v0.fits
	#inputTXT=${PATH2DATA}UNIT_DESI_Shadab_HOD_snap97_${class}_v0_inverse.txt
	#python $Txt2fits $inputTXT
	#du -h ${PATH2DATA}UNIT_DESI_Shadab_HOD_snap97_${class}_v0_inverse.fits

	#
	# Run Nbodykit to get BOX DD(s, mu)
	# 79 min -- 8 processes
#	for ext in v0  v0_inverse
#	do
#		inputFIT=${PATH2DATA}UNIT_DESI_Shadab_HOD_snap97_${class}_${ext}.fits
#		outJSON=${PATH2DATA}UNIT_DESI_Shadab_HOD_snap97_${class}_${ext}_DDsmu.json
#                outJSON_rsd=${PATH2DATA}UNIT_DESI_Shadab_HOD_snap97_${class}_${ext}_DDsmu_rsd.json
		#du -h ${inputFIT}
		#echo $outJSON
#		mpirun -np 8 python $DDsmu $inputFIT $outJSON
#		mpirun -np 8 python $DDsmu_rsd $inputFIT $outJSON_rsd
#	done
#done
