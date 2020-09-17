
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
	    fo.write(line.split('-')[1].split(':')[0]+' \n')
