import numpy as np
import os
import sixtracktools
import pysixtrack


def vectorize_all_coords(Dx_wrt_CO_m, Dpx_wrt_CO_rad,
                         Dy_wrt_CO_m, Dpy_wrt_CO_rad,
                         Dsigma_wrt_CO_m, Ddelta_wrt_CO):

    n_part = 1
    for var in [Dx_wrt_CO_m, Dpx_wrt_CO_rad,
                Dy_wrt_CO_m, Dpy_wrt_CO_rad,
                Dsigma_wrt_CO_m, Ddelta_wrt_CO]:
        if hasattr(var, '__iter__'):
            if n_part == 1:
                n_part = len(var)
            assert len(var) == n_part

    Dx_wrt_CO_m = Dx_wrt_CO_m + np.zeros(n_part)
    Dpx_wrt_CO_rad = Dpx_wrt_CO_rad + np.zeros(n_part)
    Dy_wrt_CO_m = Dy_wrt_CO_m + np.zeros(n_part)
    Dpy_wrt_CO_rad = Dpy_wrt_CO_rad + np.zeros(n_part)
    Dsigma_wrt_CO_m = Dsigma_wrt_CO_m + np.zeros(n_part)
    Ddelta_wrt_CO = Ddelta_wrt_CO + np.zeros(n_part)

    return Dx_wrt_CO_m, Dpx_wrt_CO_rad,\
     Dy_wrt_CO_m, Dpy_wrt_CO_rad,\
     Dsigma_wrt_CO_m, Ddelta_wrt_CO


def track_particle_sixtrack(
                            partCO, Dx_wrt_CO_m, Dpx_wrt_CO_rad,
                            Dy_wrt_CO_m, Dpy_wrt_CO_rad,
                            Dsigma_wrt_CO_m, Ddelta_wrt_CO, n_turns
                            ):

    Dx_wrt_CO_m, Dpx_wrt_CO_rad,\
    Dy_wrt_CO_m, Dpy_wrt_CO_rad,\
    Dsigma_wrt_CO_m, Ddelta_wrt_CO = vectorize_all_coords(
                         Dx_wrt_CO_m, Dpx_wrt_CO_rad,
                         Dy_wrt_CO_m, Dpy_wrt_CO_rad,
                         Dsigma_wrt_CO_m, Ddelta_wrt_CO)

    n_part = len(Dx_wrt_CO_m)

    wfold = 'temp_trackfun'

    if not os.path.exists(wfold):
        os.mkdir(wfold)

    os.system('cp fort.* %s' % wfold)

    with open('fort.3', 'r') as fid:
        lines_f3 = fid.readlines()

    # Set initial coordinates
    i_start_ini = None
    for ii, ll in enumerate(lines_f3):
        if ll.startswith('INITIAL COO'):
            i_start_ini = ii
            break

    lines_f3[i_start_ini + 2] = '    0.\n'
    lines_f3[i_start_ini + 3] = '    0.\n'
    lines_f3[i_start_ini + 4] = '    0.\n'
    lines_f3[i_start_ini + 5] = '    0.\n'
    lines_f3[i_start_ini + 6] = '    0.\n'
    lines_f3[i_start_ini + 7] = '    0.\n'

    lines_f3[i_start_ini + 2 + 6] = '    0.\n'
    lines_f3[i_start_ini + 3 + 6] = '    0.\n'
    lines_f3[i_start_ini + 4 + 6] = '    0.\n'
    lines_f3[i_start_ini + 5 + 6] = '    0.\n'
    lines_f3[i_start_ini + 6 + 6] = '    0.\n'
    lines_f3[i_start_ini + 7 + 6] = '    0.\n'

    lines_f13 = []

    temp_part = pysixtrack.Particles(**partCO)

    for i_part in range(n_part):

        if Ddelta_wrt_CO[i_part] != 0.:
            raise ValueError('Not implemented!')

        lines_f13.append('%.10e\n' % ((Dx_wrt_CO_m[i_part] + temp_part.x) * 1e3))
        lines_f13.append('%.10e\n' % ((Dpx_wrt_CO_rad[i_part] + temp_part.px) * temp_part.rpp * 1e3))
        lines_f13.append('%.10e\n' % ((Dy_wrt_CO_m[i_part] + temp_part.y) * 1e3))
        lines_f13.append('%.10e\n' % ((Dpy_wrt_CO_rad[i_part] + temp_part.py) * temp_part.rpp * 1e3))
        lines_f13.append('%.10e\n' % ((Dsigma_wrt_CO_m[i_part] + temp_part.sigma) * 1e3))
        lines_f13.append('%.10e\n' % ((Ddelta_wrt_CO[i_part] + temp_part.delta)))
        if i_part % 2 == 1:
            lines_f13.append('%.10e\n' % (temp_part.energy0 * 1e-6))
            lines_f13.append('%.10e\n' % (temp_part.Energy * 1e-6))
            lines_f13.append('%.10e\n' % (temp_part.Energy * 1e-6))

    with open(wfold + '/fort.13', 'w') as fid:
        fid.writelines(lines_f13)

    if np.mod(n_part, 2) != 0:
        raise ValueError('SixTrack does not like this!')

    i_start_tk = None
    for ii, ll in enumerate(lines_f3):
        if ll.startswith('TRACKING PAR'):
            i_start_tk = ii
            break
    # Set number of turns and number of particles
    temp_list = lines_f3[i_start_tk + 1].split(' ')
    temp_list[0] = '%d' % n_turns
    temp_list[2] = '%d' % (n_part / 2)
    lines_f3[i_start_tk + 1] = ' '.join(temp_list)
    # Set number of idfor = 2
    temp_list = lines_f3[i_start_tk + 2].split(' ')
    temp_list[2] = '2'
    lines_f3[i_start_tk + 2] = ' '.join(temp_list)

    # Setup turn-by-turn dump
    i_start_dp = None
    for ii, ll in enumerate(lines_f3):
        if ll.startswith('DUMP'):
            i_start_dp = ii
            break

    lines_f3[i_start_dp + 1] = 'StartDUMP 1 664 101 dumtemp.dat\n'

    with open(wfold + '/fort.3', 'w') as fid:
        fid.writelines(lines_f3)

    os.system('./runsix_trackfun')

    # Load sixtrack tracking data
    sixdump_all = sixtracktools.SixDump101('%s/dumtemp.dat' % wfold)

    x_tbt = np.zeros((n_turns, n_part))
    px_tbt = np.zeros((n_turns, n_part))
    y_tbt = np.zeros((n_turns, n_part))
    py_tbt = np.zeros((n_turns, n_part))
    sigma_tbt = np.zeros((n_turns, n_part))
    delta_tbt = np.zeros((n_turns, n_part))

    for i_part in range(n_part):
        sixdump_part = sixdump_all[i_part::n_part]
        x_tbt[:, i_part] = sixdump_part.x
        px_tbt[:, i_part] = sixdump_part.px
        y_tbt[:, i_part] = sixdump_part.y
        py_tbt[:, i_part] = sixdump_part.py
        sigma_tbt[:, i_part] = sixdump_part.sigma
        delta_tbt[:, i_part] = sixdump_part.delta

    return x_tbt, px_tbt, y_tbt, py_tbt, sigma_tbt, delta_tbt


def track_particle_pysixtrack(line, part, Dx_wrt_CO_m, Dpx_wrt_CO_rad,
                              Dy_wrt_CO_m, Dpy_wrt_CO_rad,
                              Dsigma_wrt_CO_m, Ddelta_wrt_CO, n_turns, verbose=False):

    Dx_wrt_CO_m, Dpx_wrt_CO_rad,\
        Dy_wrt_CO_m, Dpy_wrt_CO_rad,\
        Dsigma_wrt_CO_m, Ddelta_wrt_CO = vectorize_all_coords(
                             Dx_wrt_CO_m, Dpx_wrt_CO_rad,
                             Dy_wrt_CO_m, Dpy_wrt_CO_rad,
                             Dsigma_wrt_CO_m, Ddelta_wrt_CO)

    part.x += Dx_wrt_CO_m
    part.px += Dpx_wrt_CO_rad
    part.y += Dy_wrt_CO_m
    part.py += Dpy_wrt_CO_rad
    part.sigma += Dsigma_wrt_CO_m
    part.delta += Ddelta_wrt_CO

    x_tbt = []
    px_tbt = []
    y_tbt = []
    py_tbt = []
    sigma_tbt = []
    delta_tbt = []

    for i_turn in range(n_turns):
        if verbose:
            print('Turn %d/%d' % (i_turn, n_turns))

        x_tbt.append(part.x.copy())
        px_tbt.append(part.px.copy())
        y_tbt.append(part.y.copy())
        py_tbt.append(part.py.copy())
        sigma_tbt.append(part.sigma.copy())
        delta_tbt.append(part.delta.copy())

        line.track(part)

    x_tbt = np.array(x_tbt)
    px_tbt = np.array(px_tbt)
    y_tbt = np.array(y_tbt)
    py_tbt = np.array(py_tbt)
    sigma_tbt = np.array(sigma_tbt)
    delta_tbt = np.array(delta_tbt)

    return x_tbt, px_tbt, y_tbt, py_tbt, sigma_tbt, delta_tbt

def track_particle_sixtracklib(
                            line, partCO, Dx_wrt_CO_m, Dpx_wrt_CO_rad,
                            Dy_wrt_CO_m, Dpy_wrt_CO_rad,
                            Dsigma_wrt_CO_m, Ddelta_wrt_CO, n_turns,
                            device=None):


    Dx_wrt_CO_m, Dpx_wrt_CO_rad,\
        Dy_wrt_CO_m, Dpy_wrt_CO_rad,\
        Dsigma_wrt_CO_m, Ddelta_wrt_CO = vectorize_all_coords(
                             Dx_wrt_CO_m, Dpx_wrt_CO_rad,
                             Dy_wrt_CO_m, Dpy_wrt_CO_rad,
                             Dsigma_wrt_CO_m, Ddelta_wrt_CO)

    part = pysixtrack.Particles(**partCO)

    import pysixtracklib
    elements=pysixtracklib.Elements()
    elements.BeamMonitor(num_stores=n_turns)
    elements.append_line(line)

    n_part = len(Dx_wrt_CO_m)

    # Build PyST particle

    ps = pysixtracklib.ParticlesSet()
    p = ps.Particles(num_particles=n_part)

    for i_part in range(n_part):

        part = pysixtrack.Particles(**partCO)
        part.x += Dx_wrt_CO_m[i_part]
        part.px += Dpx_wrt_CO_rad[i_part]
        part.y += Dy_wrt_CO_m[i_part]
        part.py += Dpy_wrt_CO_rad[i_part]
        part.sigma += Dsigma_wrt_CO_m[i_part]
        part.delta += Ddelta_wrt_CO[i_part]

        part.partid = i_part
        part.state = 1
        part.elemid = 0
        part.turn = 0

        p.from_pysixtrack(part, i_part)

    if device is None:
        job = pysixtracklib.TrackJob(elements, ps)
    else:
        job = pysixtracklib.TrackJob(elements, ps, device=device)

    job.track(n_turns)
    job.collect()

    res = job.output

    x_tbt = res.particles[0].x.reshape(n_turns, n_part)
    px_tbt = res.particles[0].px.reshape(n_turns, n_part)
    y_tbt = res.particles[0].y.reshape(n_turns, n_part)
    py_tbt = res.particles[0].py.reshape(n_turns, n_part)
    sigma_tbt = res.particles[0].sigma.reshape(n_turns, n_part)
    delta_tbt = res.particles[0].delta.reshape(n_turns, n_part)

    # For now data are saved at the end of the turn by STlib and at the beginning by the others
    #x_tbt[1:, :] = x_tbt[:-1, :]
    #px_tbt[1:, :] = px_tbt[:-1, :]
    #y_tbt[1:, :] = y_tbt[:-1, :]
    #py_tbt[1:, :] = py_tbt[:-1, :]
    #sigma_tbt[1:, :] = sigma_tbt[:-1, :]
    #delta_tbt[1:, :] = delta_tbt[:-1, :]
    #x_tbt[0, :] = p.x
    #px_tbt[0, :] = p.px
    #y_tbt[0, :] = p.y
    #py_tbt[0, :] = p.py
    #sigma_tbt[0, :] = p.sigma
    #delta_tbt[0, :] = p.delta

    print('Done loading!')

    return x_tbt, px_tbt, y_tbt, py_tbt, sigma_tbt, delta_tbt


def betafun_from_ellip(x_tbt, px_tbt):

    x_max = np.max(x_tbt)
    mask = np.logical_and(np.abs(x_tbt) < x_max / 5., px_tbt > 0)
    x_masked = x_tbt[mask]
    px_masked = px_tbt[mask]
    ind_sorted = np.argsort(x_masked)
    x_sorted = np.take(x_masked, ind_sorted)
    px_sorted = np.take(px_masked, ind_sorted)

    px_cut = np.interp(0, x_sorted, px_sorted)

    beta_x = x_max / px_cut

    return beta_x, x_max, px_cut
