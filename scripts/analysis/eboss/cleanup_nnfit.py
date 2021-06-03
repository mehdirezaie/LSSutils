""" Clean up NN regression outputs """
import os
import sys
from glob import glob

from sysnet.sources import tar_models

model_path = sys.argv[1]


models = glob('{}'.format(os.path.join(model_path, 'model_*_*')))

print(f'# models: {len(models)}')
if len(models) < 1:
   exit('no file found!')
print(models[0], models[-1])

tar_models(model_path)
