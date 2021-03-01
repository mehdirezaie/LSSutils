import sys
sys.path.append('/Users/mehdi/github/lssutils')
from lssutils.lab import *
from lssutils import setup_logging

setup_logging("info")



biases = [1.0, 2.0, 3.0, 4.0]

# initialize the task manager to run the tasks
with TaskManager(cpus_per_task=1, use_all_cpus=True) as tm:

    # set up the linear power spectrum

    # iterate through the bias values
    for bias in tm.iterate(biases):
        print(2*bias)
