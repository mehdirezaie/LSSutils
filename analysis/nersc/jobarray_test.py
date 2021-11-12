

import numpy as np
from time import time
from argparse import ArgumentParser


def main(inputFile, outputFile):
    inputData = np.loadtxt(inputFile)
    myfile    = open(outputFile, 'w')
    myfile.write("These are BAO fit parameters\n")
    myfile.write(f"dimensions of the input file : {inputData.shape}\n")
    myfile.close()


ap = ArgumentParser(description='BAO Fit Test')
ap.add_argument('--input')
ap.add_argument('--output')
ns = ap.parse_args()

t0 = time()
main(ns.input, ns.output)
print(f"Took {time()-t0} secs")
