import pickle
import os
import shutil

import numpy as np

import pysixtrack
import sixtracktools
import sixtracklib

from compare import compare

testname = 'lhcbeambeam_from_sixtrack'

# Clean up
shutil.rmtree(testname+'/res', ignore_errors=True)

# Prepare for sixtrack run
os.mkdir(testname+'/res')
for ff in ['fort.2', 'fort.3']:
    shutil.copyfile(testname+'/'+ff, testname+'/res/'+ff)
# Run sixtrack
os.system(f"""
(cd {testname}/res; sixtrack >fort.6)""")

##############
# Build line #
##############

# Read sixtrack input
sixinput = sixtracktools.SixInput(testname)
p0c_eV = sixinput.initialconditions[-3]*1e6

# Build pysixtrack line from sixtrack input
line, other_data = pysixtrack.Line.from_sixinput(sixinput)

# Info on sixtrack->pyblep conversion 
iconv = other_data['iconv']


########################################################
#                  Search closed orbit                 #
# (for comparison purposes we the orbit from sixtrack) #
########################################################

# Load sixtrack tracking data
sixdump_all = sixtracktools.SixDump101(testname + '/res/dump3.dat')
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
        separation_given_wrt_closed_orbit_4D = True,
        separation_given_wrt_closed_orbit_6D = True)


##########################################################
# Compare sixtrack against pysixtrack and dump particles #
##########################################################

sixdump = sixdump_all[1::2]

# Create the two particle sets
pset_pysixtrack = sixtracklib.ParticlesSet()
pset_sixtrack = sixtracklib.ParticlesSet()

print("")
i_ele = 0
for ii in range(1, len(iconv)):
    
    jja = iconv[ii-1]
    jjb = iconv[ii]
    
    prun = pysixtrack.Particles(**sixdump[ii-1].get_minimal_beam())
   
    # Some extra info needed by sixtracklib
    prun.partid = 0
    prun.state = 1 
    prun.elemid = i_ele 
    prun.turn = 0 

    # Dump sixtrack particle
    part_sixtrack = pset_sixtrack.Particles(num_particles=1)
    part_sixtrack.from_pysixtrack(prun, 0)
    
    pbench_prev = prun.copy()
    
    print(f"\n-----sixtrack={ii} sixtracklib={jja} --------------")
    #print(f"pysixtr {jja}, x={prun.x}, px={prun.px}")
    for jj in range(jja+1, jjb+1):
        label = line.element_names[jj]
        elem = line.elements[jj]
        
        part_pysixtrack = pset_pysixtrack.Particles(num_particles=1)
        part_pysixtrack.from_pysixtrack(prun, 0)

        elem.track(prun)
        i_ele += 1
        prun.elemid = i_ele
        print(f"{jj} {label},{str(elem)[:50]}")
    pbench = pysixtrack.Particles(**sixdump[ii].get_minimal_beam())
    #print(f"sixdump {ii}, x={pbench.x}, px={pbench.px}")
    print("-----------------------")
    error = compare(prun, pbench, pbench_prev)
    print("-----------------------\n\n")

    if error:
        print('Error detected')
        break

# Build elements buffer
elements=sixtracklib.Elements()
elements.append_line(line)

# Dump first test
elements.cbuffer.tofile(testname+'_trackedbysixtrack_elements.bin')
pset_sixtrack.cbuffer.tofile(testname+'_trackedbysixtrack_particles.bin')

# Dump second test
elements.cbuffer.tofile(testname+'_trackedbypysixtrack_elements.bin')
pset_pysixtrack.cbuffer.tofile(testname+'_trackedbypysixtrack_particles.bin')

