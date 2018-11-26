import sixtracktools
import pysixtrack
import pysixtrack.helpers as hp
import pickle
import os

os.system('./runsix')



import numpy as np

# Read sixtrack input
six = sixtracktools.SixInput('.')
p0c_eV = six.initialconditions[-3]*1e6

# Build pysixtrack line
line, rest, iconv = six.expand_struct(convert=pysixtrack.element_types)

# Disable BB elements
ind_BB4D, namelistBB4D, listBB4D = hp.get_elems_of_type(line, 'BeamBeam4D')
for bb in listBB4D:
    bb.enabled = False
ind_BB6D, namelistBB6D, listBB6D = hp.get_elems_of_type(line, 'BeamBeam6D')
for bb in listBB6D:
    bb.enabled = False

# Load sixtrack tracking data
sixdump_all = sixtracktools.SixDump101('res/dump3.dat')
# Assume first particle to be on the closed orbit
Nele_st = len(iconv)
sixdump_CO = sixdump_all[::2][:Nele_st]

# Find closed orbit
ring = hp.Ring(line, p0c=p0c_eV)
guess = [getattr(sixdump_CO, att)[0]
         for att in 'x px y py sigma delta'.split()]
closed_orbit = ring.find_closed_orbit(guess=guess, method='get_guess')

print('Closed orbit at start machine:')
print('x px y py sigma delta:', guess)


# Check that closed orbit is closed
pstart = closed_orbit[0].copy()
pstart_st = pysixtrack.Particles(**sixdump_CO[0].get_minimal_beam())

with open('particle_on_CO.pkl', 'wb') as fid:
    pickle.dump(sixdump_CO[0].get_minimal_beam(), fid)

print('STsigma, Sigma, Stdelta, delta, Stpx, px')
for iturn in range(10):
    ring.track(pstart)
    ring.track(pstart_st)
    print('%e, %e, %e, %e, %e, %e' % (pstart_st.sigma, pstart.sigma,
                                      pstart_st.delta, pstart.delta, pstart_st.px, pstart.px))




import matplotlib.pyplot as plt

# Compare closed orbit against sixtrack
for att in 'x px y py delta sigma'.split():
    att_CO = np.array([getattr(pp, att) for pp in closed_orbit])
    att_CO_at_st_ele = att_CO[iconv]
    print('Max C.O. discrepancy in %s %.2e' %
          (att, np.max(np.abs(att_CO_at_st_ele-getattr(sixdump_CO, att)))))

plt.figure(1)
plt.plot(sixdump_CO.s, sixdump_CO.x)


# Re-enable beam-beam
for bb in listBB4D:
    bb.enabled = True
for bb in listBB6D:
    bb.enabled = True

# Add closed orbit to separation for BB4D (as assumed in sixtrack)
for bb, ibb in zip(listBB4D, ind_BB4D):
    bb.Delta_x += closed_orbit[ibb].x
    bb.Delta_y += closed_orbit[ibb].y

# Evaluate kick at CO location BB4D
for bb, ibb in zip(listBB4D, ind_BB4D):

    ptemp = closed_orbit[ibb].copy()
    ptempin = ptemp.copy()

    bb.track(ptemp)

    Dpx = ptemp.px - ptempin.px
    Dpy = ptemp.py - ptempin.py

    bb.Dpx_sub = Dpx
    bb.Dpy_sub = Dpy

# Provide closed orbit to BB6D
for bb, ibb in zip(listBB6D, ind_BB6D):

    bb.x_CO = closed_orbit[ibb].x
    bb.px_CO = closed_orbit[ibb].px
    bb.y_CO = closed_orbit[ibb].y
    bb.py_CO = closed_orbit[ibb].py
    bb.sigma_CO = closed_orbit[ibb].zeta
    bb.delta_CO = closed_orbit[ibb].delta


# Evaluate kick at CO location BB6D
for bb, ibb in zip(listBB6D, ind_BB6D):

    # For debug
    bb.Dx_sub = 0.
    bb.Dpx_sub = 0.
    bb.Dy_sub = 0.
    bb.Dpy_sub = 0.
    bb.Dsigma_sub = 0.
    bb.Ddelta_sub = 0.
    ######

    ptemp = closed_orbit[ibb].copy()
    ptempin = ptemp.copy()

    bb.track(ptemp)
    print('Estimated x orbit kick', ptemp.x - ptempin.x)

    bb.Dx_sub = ptemp.x - ptempin.x
    bb.Dpx_sub = ptemp.px - ptempin.px
    bb.Dy_sub = ptemp.y - ptempin.y
    bb.Dpy_sub = ptemp.py - ptempin.py
    bb.Dsigma_sub = ptemp.zeta - ptempin.zeta
    bb.Ddelta_sub = ptemp.delta - ptempin.delta

# Check that the closed orbit is not kicked
for bb, ibb in zip(listBB6D, ind_BB6D):

    ptemp = closed_orbit[ibb].copy()
    ptempin = ptemp.copy()

    bb.track(ptemp)

    print('Again kick', ptemp.x - ptempin.x)


with open('line.pkl', 'wb') as fid:
    pickle.dump(line, fid)


lineobj=pysixtrack.Line(elements=[elem for label,elem_type,elem in line])
with open('lineobj.pkl', 'wb') as fid:
    pickle.dump(lineobj, fid)


# Compare tracking results
sixdump = sixdump_all[1::2]  # Particle with deviation from CO
# sixdump = sixdump_all[::2] # Particle on CO

p_in_st = pysixtrack.Particles(**sixdump[0].get_minimal_beam())
p_out_st = pysixtrack.Particles(**sixdump[1].get_minimal_beam())

p_in_pyst = p_in_st.copy()
p_out_pyst = p_in_pyst.copy()

if listBB6D:
    listBB6D[0].track(p_out_pyst)


for att in 'x px y py delta sigma'.split():
    attin = getattr(p_in_st, att)
    attout = getattr(p_out_st, att)
    print('SxTr: Change in '+att+': %e' % (attout-attin))

    attin_pyst = getattr(p_in_pyst, att)
    attout_pyst = getattr(p_out_pyst, att)
    print('PyST: Change in '+att+': %e' % (attout_pyst-attin_pyst))


def compare(prun, pbench, pbench_prev):
    out = []
    out_rel = []
    error = False
    for att in 'x px y py delta sigma'.split():
        vrun = getattr(prun, att)
        vbench = getattr(pbench, att)
        vbench_prev = getattr(pbench_prev, att)
        diff = vrun-vbench
        diffrel = abs(1.-abs(vrun-vbench_prev)/abs(vbench-vbench_prev))
        out.append(abs(diff))
        out_rel.append(diffrel)
        print(f"{att:<5} {vrun:22.13e} {vbench:22.13e} {diff:22.13g} {diffrel:22.13g}")
        if diffrel > 1e-8 or np.isnan(diffrel):
            if diff > 1e-11:
                print('Too large discrepancy!')
                error = True
    print(f"\nmax {max(out):21.12e} maxrel {max(out_rel):22.12e}")
    return error


print("")
for ii in range(1, len(iconv)):
    jja = iconv[ii-1]
    jjb = iconv[ii]
    prun = pysixtrack.Particles(**sixdump[ii-1].get_minimal_beam())
    pbench_prev = prun.copy()
    print(f"\n-----sixtrack={ii} sixtracklib={jja} --------------")
    #print(f"pysixtr {jja}, x={prun.x}, px={prun.px}")
    for jj in range(jja+1, jjb+1):
        label, elem_type, elem = line[jj]
        pin = prun.copy()
        elem.track(prun)
        print(f"{jj} {label},{str(elem)[:50]}")
    pbench = pysixtrack.Particles(**sixdump[ii].get_minimal_beam())
    #print(f"sixdump {ii}, x={pbench.x}, px={pbench.px}")
    print("-----------------------")
    error = compare(prun, pbench, pbench_prev)
    print("-----------------------\n\n")

    if error:
        print('Error detected')
        break
