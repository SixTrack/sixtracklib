import pysixtrack
import sixtracklib
import cobjects

from compare import compare

binary_folder = '../'

testname = 'lhcbeambeam_from_sixtrack_trackedbypysixtrack'
reltol = 1e-8; abstol = 1e-11

testname = 'lhcbeambeam_from_sixtrack_trackedbysixtrack'
reltol = 1e-8; abstol = 1e-11

testname = 'simplebb_from_sixtrack_trackedbypysixtrack'
reltol = 1e-5; abstol = 9e-10

testname = 'simplebb_from_sixtrack_trackedbysixtrack'
reltol = 1e-5; abstol = 9e-10


# Load testdata
ebuf = cobjects.CBuffer.fromfile(binary_folder + '/' + testname + '_elements.bin')
pbuf = cobjects.CBuffer.fromfile(binary_folder + '/' + testname + '_particles.bin')

# Build particle for tracking 
trackbuf = cobjects.CBuffer()
sixtracklib.particles.makeCopy(
        pbuf.get_object(0, cls=sixtracklib.Particles), trackbuf)
ptrack = trackbuf.get_object(0, cls=sixtracklib.Particles)

# Build the job
job = sixtracklib.TrackJob(ebuf, trackbuf)

# Comparison
N_part_test = pbuf.n_objects
pref_start = pysixtrack.Particles()
pref_end = pysixtrack.Particles()
ptest = pysixtrack.Particles()
for ii in range(1, N_part_test):
    
    i_ele_start = pbuf.get_object(ii-1, cls=sixtracklib.Particles).at_element
    i_ele_end = pbuf.get_object(ii, cls=sixtracklib.Particles).at_element
    
    # Copy reference to pref
    pbuf.get_object(ii-1, cls=sixtracklib.Particles).to_pysixtrack(pref_start, 0)
    pbuf.get_object(ii, cls=sixtracklib.Particles).to_pysixtrack(pref_end, 0)

    job.particles_buffer.get_object(0, cls=sixtracklib.Particles).from_pysixtrack(pref_start, 0)
    job.particles_buffer.get_object(0, cls=sixtracklib.Particles).at_element[0] = i_ele_start
    
    # job.push_particles() # New API to come
    job.track_line(i_ele_start, i_ele_end)
    # job.collet_particles() # New API to come

    job.particles_buffer.get_object(0, cls=sixtracklib.Particles).to_pysixtrack(ptest, 0)

    print("-----------------------")
    print(f"element {ii}")
    error = compare(ptest, pref_end, pref_start, reltol, abstol)
    print("-----------------------\n\n")

    if error:
        print('Error detected')
        break

