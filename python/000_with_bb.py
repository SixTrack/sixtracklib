import pysixtrack
import pickle

import export_to_cobjects as etc

pyst_example = 'bbsimple'
#pyst_example = 'beambeam'

# Test on pysixtrack example
pyst_path = pysixtrack.__file__
input_folder = '/'.join(pyst_path.split('/')[:-2]+['examples', pyst_example])

with open(input_folder+'/line.pkl', 'rb') as fid:
	line = pickle.load(fid)

etc.line2cobject( line, pyst_example+'_st_input.bin')
etc.sixdump2cobject( input_folder, input_folder+'/res/dump3.dat',  pyst_example+'_st_dump.bin')
