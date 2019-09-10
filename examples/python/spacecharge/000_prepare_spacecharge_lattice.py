import matplotlib.patches as patches
import pickle

import numpy as np
import matplotlib.pylab as plt
from scipy.constants import physical_constants

from cpymad.madx import Madx

import pysixtrack
from pysixtrack.particles import Particles
import pysixtrack.be_beamfields.tools as bt

#sc_mode = 'Coasting'
sc_mode = 'Bunched'

# For bunched
number_of_particles = 2e11
bunchlength_rms = 0.22

# For coasting
line_density = 2e11 / 0.5

mass = physical_constants['proton mass energy equivalent in MeV'][0] * 1e6
p0c = 25.92e9
neps_x = 2e-6
neps_y = 2e-6
delta_rms = 1.5e-3
V_RF_MV = 4.5
lag_RF_deg = 180.
n_SCkicks = 10  # 216 #80
length_fuzzy = 1.5
seq_name = 'sps'

mad = Madx()
mad.options.echo = False
mad.options.info = False
mad.warn = False
mad.chdir('madx')
mad.call('sps_thin.madx')
mad.use(seq_name)

# Determine space charge locations
temp_line, other = pysixtrack.Line.from_madx_sequence(mad.sequence.sps)
sc_locations, sc_lengths = bt.determine_sc_locations(
    temp_line, n_SCkicks, length_fuzzy)

# Install spacecharge place holders
sc_names = ['sc%d' % number for number in range(len(sc_locations))]
bt.install_sc_placeholders(mad, seq_name, sc_names, sc_locations, mode=sc_mode)

# twiss
twtable = mad.twiss()

# Generate line with spacecharge
line, other = pysixtrack.Line.from_madx_sequence(mad.sequence.sps)

# Get sc info from optics
mad_sc_names, sc_points, sc_twdata = bt.get_spacecharge_names_madpoints_twdata(
    mad, seq_name, mode=sc_mode)

# Check consistency
if sc_mode == 'Bunched':
    sc_elements, sc_names = line.get_elements_of_type(
        pysixtrack.elements.SpaceChargeBunched)
elif sc_mode == 'Coasting':
    sc_elements, sc_names = line.get_elements_of_type(
        pysixtrack.elements.SpaceChargeCoasting)
else:
    raise ValueError('mode not understood')
bt.check_spacecharge_consistency(
    sc_elements, sc_names, sc_lengths, mad_sc_names)

# Setup spacecharge in the line
if sc_mode == 'Bunched':
    bt.setup_spacecharge_bunched_in_line(
        sc_elements,
        sc_lengths,
        sc_twdata,
        sc_points,
        p0c,
        mass,
        number_of_particles,
        bunchlength_rms,
        delta_rms,
        neps_x,
        neps_y)
elif sc_mode == 'Coasting':
    bt.setup_spacecharge_coasting_in_line(
        sc_elements,
        sc_lengths,
        sc_twdata,
        sc_points,
        p0c,
        mass,
        line_density,
        delta_rms,
        neps_x,
        neps_y)
else:
    raise ValueError('mode not understood')
# enable RF
i_cavity = line.element_names.index('acta.31637')
line.elements[i_cavity].voltage = V_RF_MV * 1e6
line.elements[i_cavity].lag = lag_RF_deg


with open('line.pkl', 'wb') as fid:
    pickle.dump(line.to_dict(keepextra=True), fid)

part_on_CO = line.find_closed_orbit(
    guess=[
        twtable['x'][0],
        twtable['px'][0],
        twtable['y'][0],
        twtable['py'][0],
        0.,
        0.],
    p0c=p0c,
    method='get_guess')

# Save particle on CO
with open('particle_on_CO.pkl', 'wb') as fid:
    pickle.dump(part_on_CO.to_dict(), fid)

# Save twiss at start ring
with open('twiss_at_start.pkl', 'wb') as fid:
    pickle.dump({
        'betx': twtable.betx[0],
        'bety': twtable.bety[0]}, fid)

''

if 0:
    plt.close('all')

    f, ax = plt.subplots()
    ax.hist(sc_lengths, bins=np.linspace(0, max(sc_lengths) + 0.1, 100))
    ax.set_xlabel('length of SC kick (m)')
    ax.set_ylabel('counts')
    ax.set_xlim(left=0)
    plt.show()

    f, ax = plt.subplots(figsize=(14, 5))
    ax.plot(twtable.s, twtable.betx, 'b', label='x', lw=2)
    ax.plot(twtable.s, twtable.bety, 'g', label='x', lw=2)
    for s in sc_locations:
        ax.axvline(s, linewidth=1, color='r', linestyle='--')
    ax.set_xlim(0, 1100)
    ax.set_ylim(0, 120)
    ax.set_xlabel('s (m)')
    ax.set_ylabel('beta functions (m)')
    ax.legend(loc=3)
    plt.show()


''
