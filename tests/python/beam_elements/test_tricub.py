import sixtracklib as st
import sixtracklib_test as sttest
import pysixtrack
import cobjects
import numpy as np
import pickle
import os


def test_kicks(
    cmp_file_name="precomputed_kicks.pickle",
    device_str=None,
    abs_tol=1e-15,
    rel_tol=0.0,
):

    ####### load file with correct kicks #######
    path_to_testdir = sttest.config.PATH_TO_TESTDATA_DIR
    assert path_to_testdir is not None
    assert os.path.exists(path_to_testdir)
    assert os.path.isdir(path_to_testdir)

    path_to_cmp_file = os.path.join(path_to_testdir, "tricub", cmp_file_name)
    assert os.path.exists(path_to_cmp_file)

    with open(path_to_cmp_file, "rb") as fp:
        n_part, prng_seed, kicks = pickle.load(fp)

    assert n_part > 0
    assert prng_seed is not None
    assert kicks is not None

    np.random.seed(int(prng_seed))

    lattice = st.Elements()
    tc_index = lattice.cbuffer.n_objects
    tc = st.TriCub(cbuffer=lattice.cbuffer)
    tc.length = 1.0

    particles_set = st.ParticlesSet()
    particles = particles_set.Particles(num_particles=n_part)

    nx = 5
    ny = 7
    nz = 9
    A = np.random.rand(nx, ny, nz, 8) * 1.0e-3
    dx = 0.001
    dy = 0.002
    dz = 0.003
    x0 = -(nx // 2) * dx
    y0 = -(ny // 2) * dy
    z0 = -(nz // 2) * dz

    test_x = x0 + (nx - 2) * dx * np.random.rand(n_part)
    test_y = y0 + (ny - 2) * dy * np.random.rand(n_part)
    test_z = z0 + (nz - 2) * dz * np.random.rand(n_part)

    for i_part in range(n_part):
        part = pysixtrack.Particles()
        part.x = test_x[i_part]
        part.y = test_y[i_part]
        part.tau = test_z[i_part]

        part.partid = i_part
        part.state = 1
        part.elemid = 0
        part.turn = 0
        particles.from_pysixtrack(part, i_part)

    job = st.TrackJob(lattice, particles_set, device=device_str)

    tricub_data_buffer = cobjects.CBuffer()
    tc_data_index = tricub_data_buffer.n_objects
    tc_data = st.TriCubData(cbuffer=tricub_data_buffer, nx=nx, ny=ny, nz=nz)

    tc_data.x0 = x0
    tc_data.y0 = y0
    tc_data.z0 = z0
    tc_data.dx = dx
    tc_data.dy = dy
    tc_data.dz = dz
    tc_data.mirror_x = 0
    tc_data.mirror_y = 0
    tc_data.mirror_z = 0
    scale = [1.0, dx, dy, dz, dx * dy, dx * dz, dy * dz, (dx * dy) * dz]
    for ii in range(nx):
        for jj in range(ny):
            for kk in range(nz):
                for ll in range(8):
                    tc_data.table_addr[ll + 8 * (ii + nx * (jj + ny * kk))] = (
                        A[ii, jj, kk, ll] * scale[ll]
                    )

    tricub_data_buffer_id = job.add_stored_buffer(cbuffer=tricub_data_buffer)

    st.TriCub_buffer_create_assign_address_item(
        job, tc_index, tricub_data_buffer_id, tc_data_index
    )

    job.commit_address_assignments()
    job.assign_all_addresses()
    job.track_until(1)
    job.collect()

    assert np.allclose(kicks[:, 0], particles.px, rel_tol, abs_tol)
    assert np.allclose(kicks[:, 1], particles.py, rel_tol, abs_tol)
    assert np.allclose(kicks[:, 2], particles.ptau, rel_tol, abs_tol)


if __name__ == "__main__":
    if st.config.TRACK_TRICUB == "enabled":
        test_kicks(
            cmp_file_name="precomputed_kicks.pickle",
            rel_tol=1e-12,
            abs_tol=1e-13,
        )
    else:
        print("Disabled TriCub beam-element -> skip test")
