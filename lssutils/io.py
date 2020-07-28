
import numpy as np

def readnbodykit(filename):
    with open(filename, 'r') as infile:
        lines = infile.readlines()

        shotnoise = None
        values = []
        for line in lines:
            if '#shotnoise' in line:
                shotnoise = float(line.split(':')[-1])
                #print(f'shotnoise {shotnoise}')
            else:
                if line.startswith('#'):
                    continue
                else:
                    strings = line.split(' ')
                    values.append([float(s) for s in strings])
        if values != []:
            values = np.array(values)
        else:
            raise RuntimeError('reading failed')
    return values, shotnoise