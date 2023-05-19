# directories to sync
a='rongpu lognormal tanveer'
b=' '
for ai in $a;do b=${b}' '/fs/ess/PHS0336/data/${ai};done
echo ${b}
rsync -rcv ${b}  mehdi@moneta.phy.ohio.edu:/DATA2/mehdi/
