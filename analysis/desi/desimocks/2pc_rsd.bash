#!/bin/bash
#
# Mehdi Rezaie, mr095415@ohio.edu
# 
#
#
#   CODES
#
# txt2fits for nbodykit
Txt2fits=/home/mehdi/github/desimocks/txt2fits.py

# DD(s,mu)
DDsmu=/home/mehdi/github/LSSutils/scripts/analysis/Box2pc_rsd.py

#
# DATA path
#
PATH2DATA=$HOME/data/redshift0.9873/
#
# STEP 1
#
for class in ELG LRG QSO
do 
	echo $class
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
	for ext in v0  v0_inverse
	do
		inputFIT=${PATH2DATA}UNIT_DESI_Shadab_HOD_snap97_${class}_${ext}.fits
		outJSON=${PATH2DATA}UNIT_DESI_Shadab_HOD_snap97_${class}_${ext}_DDsmu_rsd.json
		#du -h ${inputFIT}
		#echo $outJSON
		mpirun -np 8 python $DDsmu $inputFIT $outJSON
	done
done
