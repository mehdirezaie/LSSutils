for i in {0001..0999}; do FName=/fs/ess/PHS0336/data/v7/1.0/measurements/nnbar/nnbar_SGC_known_mainhighz_512_v7_1_${i}_main_512.npy; if [ ! -f "$FName" ]; then echo $FName; fi; done
