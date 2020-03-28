'''

    File to extract pks from json to dat

'''

import nbodykit.lab as nb
import sys

def main(inFile, ouFile):
    
    print(f'inFile : {inFile}')
    print(f'ouFile : {ouFile}')

    # --- read
    pk = nb.ConvolvedFFTPower.load(inFile)

    # --- export to .dat
    with open(ouFile, 'w') as output:

       output.write('#--- Nbodykit parameters ---\n')
       for attr in pk.attrs:
           output.write(f'# {attr:30s} : {pk.attrs[attr]}\n')

       nbins = len(pk.poles.coords['k'])
       output.write('# kmid, kavg, P0, P2, P4, Nmodes\n')
       for i in range(nbins):
           output.write('{} {} {} {} {} {}\n'.format(pk.poles.coords['k'][i], 
                                                     pk.poles['k'][i], 
                                                     pk.poles['power_0'][i].real, 
                                                     pk.poles['power_2'][i].real, 
                                                     pk.poles['power_4'][i].real, 
                                                     pk.poles['modes'][i]))
       
if __name__ == '__main__':
   
   inFile = sys.argv[1] 
   ouFile = inFile.replace('.json', '.dat')
   
   main(inFile, ouFile)

