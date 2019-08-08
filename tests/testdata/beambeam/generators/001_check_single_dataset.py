import pysixtrack
import sixtracklib
import cobjects

from compare import compare

testname = 'lhcbeambeam_from_sixtrack_trackedbypysixtrack'

# Load testdata
ebuf = cobjects.CBuffer.fromfile(testname + '_elements.bin')
pbuf = cobjects.CBuffer.fromfile(testname + '_particles.bin')

# Build particle for tracking 
trackbuf = cobjects.CBuffer()
sixtracklib.particles.makeCopy(
        pbuf.get_object(0, cls=sixtracklib.Particles), trackbuf)
ptrack = trackbuf.get_object(0, cls=sixtracklib.Particles)

# Build the job
job = sixtracklib.TrackJob(ebuf, trackbuf, until_turn_elem_by_elem=1)

# Track and collect
job.track_elem_by_elem(until_turn=1)
job.collect()

ebe_outp = job.output_buffer.get_object(
        job.elem_by_elem_output_offset(), cls=sixtracklib.Particles)


# Comparison
N_part_test = pbuf.n_objects
pref = pysixtrack.Particles()
ptest = pysixtrack.Particles()
for ii in range(N_part_test):
    # Copy reference to pref
    pp = pbuf.get_object(ii, cls=sixtracklib.Particles)
    eleid = pp.at_element[0]
    
    pp.to_pysixtrack(pref, 0)


    # Copy job outp to ptest
    ebe_outp.to_pysixtrack(ptest, eleid)

    if ii == 0:
        p_prev = pref.copy()

    print("-----------------------")
    error = compare(ptest, pref, p_prev)
    print("-----------------------\n\n")

    if error:
        print('Error detected')
        break

    p_prev = pref.copy()
