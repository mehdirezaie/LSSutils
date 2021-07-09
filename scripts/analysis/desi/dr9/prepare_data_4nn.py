import sys
import os
import fitsio as ft

sys.path.insert(0, '/Users/mehdi/github/LSSutils')
from lssutils.utils import DR9Data

if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise ValueError('run with $> python prepare_data_4nn.py [target] [region]')


    target = sys.argv[1]
    region = sys.argv[2]
        
    if not target in ['elg', 'lrg', 'qso']:
        raise ValueError(f'{target} must be elg, lrg, or qso')
    if not region in ['N', 'S']:
        raise ValueError(f'{region} must be N or S')

    input_path = '/Users/mehdi/dr9/data/'
    output_path = '/Users/mehdi/dr9/results/'

    input_cat = os.path.join(input_path, 'dr9m-mypixweight-dark.fits')
    output_cat = os.path.join(output_path, f'dr9m_{target}_{region}.fits')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    msg = f'input: {input_cat}\n'
    msg += f'output: {output_cat}'
    print(msg)

    dr9 = DR9Data(input_cat)
    dt = dr9.run(target, region)

    for i in range(dt['features'].shape[1]):
        print(i, dr9.features_names[i], dt['features'][:, i].min(), dt['features'][:, i].max())

    if os.path.exists(output_cat):
        raise RuntimeError(f'{output_cat} exists!')
    ft.write(output_cat, dt, clobber=True)