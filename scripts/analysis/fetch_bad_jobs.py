"""
	1. use grep to find out what ids failed
		$> grep 'killed' *.o* > failed.txt
	2. then run this script 
		$> python fetch.py failed.txt indices.txt
"""
import sys

try:
	input_fn = sys.argv[1]
	output_fn = sys.argv[2]
except:
	print('run with [script name].py input output')
	exit('exit')

with open(input_fn, 'r') as fl:
     lines = fl.readlines()

with open(output_fn, 'w') as fo:
	for line in lines:
	    fo.write(line.split('_')[1].split('.')[0]+' \n')
