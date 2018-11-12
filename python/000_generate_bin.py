import pysixtrack
import pickle
import testdata as st

from cobjects import CBuffer
import export_to_cobjects as etc

import os

if  __name__ == '__main__':
    pyst_example = 'bbsimple'
    #pyst_example = 'beambeam'

    input_folder  = os.path.join( st.PATH_TO_TESTDATA_DIR, pyst_example )
    output_folder = os.path.join( st.PATH_TO_TESTDATA_DIR, pyst_example )

    with open( os.path.join( input_folder, 'line.pkl' ), 'rb' ) as fid:
        line = pickle.load( fid )
        line_buffer = CBuffer()
        etc.line2cobject( line, cbuffer=line_buffer )
        line_buffer.to_file( os.path.join( output_folder, 'beam_elements.bin' ) )

    etc.sixdump2cobject( os.path.join( input_folder,  'dump3.dat' ),
                         os.path.join( output_folder, 'particles_buffer_sixtrack.bin' ) )
