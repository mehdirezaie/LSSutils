
## directories to sync
a='rongpu tanveer templates lognormal'
b=' '
for ai in $a;do b=${b}' '/home/mehdi/data/${ai};done
echo ${b}

rsync -rcv ${b} medirz90@owens.osc.edu:/fs/ess/PHS0336/data/
