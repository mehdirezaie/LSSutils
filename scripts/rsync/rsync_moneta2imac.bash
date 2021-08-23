
# directories to sync
a='alternative dr9v0.57.0 eboss rongpu tanveer templates formehdi lognormal'
b=' '
for ai in $a;do b=${b}' '/home/mehdi/data/${ai};done
echo ${b}

rsync -rcv ${b} rezaie@132.235.24.101:/Volumes/TimeMachine/data/
