import sixtracktools
import pysixtrack
import pickle
import os

os.system('./runsix')

import numpy as np

##############
# Build line #
##############

# Read sixtrack input
sixinput = sixtracktools.SixInput('.')
p0c_eV = sixinput.initialconditions[-3] * 1e6

# Build pysixtrack line from sixtrack input
line, other_data = pysixtrack.Line.from_sixinput(sixinput)

# Info on sixtrack->pyblep conversion
iconv = other_data['iconv']


########################################################
#                  Search closed orbit                 #
# (for comparison purposes we the orbit from sixtrack) #
########################################################

# Load sixtrack tracking data
sixdump_all = sixtracktools.SixDump101('res/dump3.dat')
# Assume first particle to be on the closed orbit
Nele_st = len(iconv)
sixdump_CO = sixdump_all[::2][:Nele_st]

# Disable BB elements
line.disable_beambeam()

# Find closed orbit
guess_from_sixtrack = [getattr(sixdump_CO, att)[0]
                       for att in 'x px y py sigma delta'.split()]
part_on_CO = line.find_closed_orbit(
    guess=guess_from_sixtrack, method='get_guess', p0c=p0c_eV)

print('Closed orbit at start machine:')
print('x px y py sigma delta:')
print(part_on_CO)

#######################################################
#  Store closed orbit and dipole kicks at BB elements #
#######################################################

line.beambeam_store_closed_orbit_and_dipolar_kicks(
    part_on_CO,
    separation_given_wrt_closed_orbit_4D=True,
    separation_given_wrt_closed_orbit_6D=True)


#################################
# Save machine in pyblep format #
#################################

with open('line.pkl', 'wb') as fid:
    pickle.dump(line.to_dict(keepextra=True), fid)

#########################################
# Save particle on closed orbit as dict #
#########################################

with open('particle_on_CO.pkl', 'wb') as fid:
    pickle.dump(part_on_CO.to_dict(), fid)

#########################################
# Save sixtrack->pyblep conversion info #
#########################################

with open('iconv.pkl', 'wb') as fid:
    pickle.dump(iconv, fid)
