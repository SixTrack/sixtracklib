import sixtracklib as st
import cobjects
import numpy as np
from cobjects import CBuffer


def gaussian_dist(z, mu=0.0, sigma=1.0):
    assert np.all(np.abs(sigma) > 0.0)
    norm_z = (z - mu) / (np.sqrt(2.0) * sigma)
    return 1.0 / np.sqrt(2 * np.pi * sigma) * np.exp(-norm_z * norm_z)


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # A) Create buffer for storing the line density datasets:
    interpol_buffer = CBuffer()

    # Prepare two line density datasets from gaussian distributions:
    # NOTE: We keep track on the indices of the datasets -> this will be used
    #       later during assignment

    # A.1) gaussian, sigma=1.0, 24 points, linear interpolation
    lin_absc = np.linspace(-8.0, 8.0, num=24, dtype="float64")
    sc_data0_idx = interpol_buffer.n_objects
    sc_data0 = st.LineDensityProfileData(
        cbuffer=interpol_buffer,
        capacity=len(lin_absc),
        z0=lin_absc[0],
        dz=lin_absc[1] - lin_absc[0],
        method="linear",
    )

    lin_values = gaussian_dist(lin_absc, mu=0.0, sigma=1.0)
    sc_data0.values[:] = lin_values
    sc_data0.prepare_interpolation()

    # A.2) gaussian, sigma=2.0, 12 points, cubic spline interpolation
    cub_absc = np.linspace(-8.0, 8.0, num=12, dtype="float64")
    sc_data1_idx = interpol_buffer.n_objects
    sc_data1 = st.LineDensityProfileData(
        cbuffer=interpol_buffer,
        capacity=len(cub_absc),
        z0=cub_absc[0],
        dz=cub_absc[1] - cub_absc[0],
        method="cubic",
    )

    sc_data1.values[:] = gaussian_dist(cub_absc, mu=0.0, sigma=2.0)
    sc_data1.prepare_interpolation()

    # A.3) optional: demonstrate that interpolation works ->
    # set plot_data = True!
    plot_data = False

    if plot_data:
        from matplotlib import pyplot as plt

        z_absc = np.linspace(-8.0, +8.0, num=512, dtype="float64")
        y_exact1 = gaussian_dist(z_absc, mu=0.0, sigma=1.0)
        sc_data0 = interpol_buffer.get_object(sc_data0_idx)
        y_interp_lin = np.array([sc_data0.interpol(zz) for zz in z_absc])

        y_exact2 = gaussian_dist(z_absc, mu=0.0, sigma=2.0)
        sc_data1 = interpol_buffer.get_object(sc_data1_idx)
        y_interp_cub = np.array([sc_data1.interpol(zz) for zz in z_absc])

        plt.figure()
        plt.subplot(211)
        plt.plot(
            lin_absc,
            sc_data0.values,
            "bo",
            z_absc,
            y_exact1,
            "k",
            z_absc,
            y_interp_lin,
            "b-",
        )

        plt.subplot(212)
        plt.plot(
            cub_absc,
            sc_data0.values,
            "ro",
            z_absc,
            y_exact2,
            "k",
            z_absc,
            y_interp_cub,
            "r-",
        )
        plt.show()

    # -------------------------------------------------------------------------
    # B) Init the particle set
    beam = st.ParticlesSet()
    particles = beam.Particles(num_particles=100, p0c=6.5e12)

    # -------------------------------------------------------------------------
    # C) Build the lattice. We add three interpolated space charge elements
    #    and keep track of the indices at which they are available
    lattice = st.Elements()

    sc0_index = lattice.cbuffer.n_objects  # index of sc0 element
    sc0 = lattice.SCInterpolatedProfile(
        number_of_particles=particles.num_particles
    )
    dr0 = lattice.Drift(length=1.0)
    q0 = lattice.Multipole(knl=[0.0, 0.1])

    sc1_index = lattice.cbuffer.n_objects  # index of sc1 element
    sc1 = lattice.SCInterpolatedProfile(
        number_of_particles=particles.num_particles
    )
    dr1 = lattice.Drift(length=1.0)
    q1 = lattice.Multipole(knl=[0.0, -0.1])

    sc2_index = lattice.cbuffer.n_objects  # index of sc2 element
    sc2 = lattice.SCInterpolatedProfile(
        number_of_particles=particles.num_particles
    )

    # --------------------------------------------------------------------------
    # D) Create the track-job
    # Create the track-job
    job = st.TrackJob(lattice, beam)
    # job = st.TrackJob(lattice, beam, device="opencl:1.0")
    # job = st.CudaTrackJob(lattice, beam)

    # --------------------------------------------------------------------------
    # E) Add the interpol_buffer to the track-job. This allows the track job
    #    to push/pull this buffer like the other buffers via the returned id

    interpol_buffer_id = job.add_stored_buffer(cbuffer=interpol_buffer)
    print(f"interpol_buffer_id = {interpol_buffer_id}")

    # --------------------------------------------------------------------------
    # F) Create the assignments of the line profile datasets to the space
    #    charge elements. Instead of doing it using the track-job API directly,
    #    we use a convenience function which hides all the gritty details

    # create first assignments:
    # sc_data0 @ sc_data0_idx -> sc0 @ sc0_index
    success = st.LineDensityProfileData_create_buffer_assignment(
        job, sc0_index, interpol_buffer_id, sc_data0_idx
    )
    assert success

    # create second assignments:
    # sc_data1 @ sc_data1_idx -> sc1 @ sc1_index
    success = st.LineDensityProfileData_create_buffer_assignment(
        job, sc1_index, interpol_buffer_id, sc_data1_idx
    )
    assert success

    # create third assignments:
    # sc_data1 @ sc_data1_idx -> sc2 @ sc2_index
    success = st.LineDensityProfileData_create_buffer_assignment(
        job, sc2_index, interpol_buffer_id, sc_data1_idx
    )
    assert success

    # --------------------------------------------------------------------------
    # G) Perform the address assignments

    job.commit_address_assignments()
    job.assign_all_addresses()

    # --------------------------------------------------------------------------
    # F) Check whether the assignments actually worked

    job.collect_beam_elements()
    job.collect_stored_buffer(interpol_buffer_id)

    sc0 = lattice.cbuffer.get_object(sc0_index)
    sc1 = lattice.cbuffer.get_object(sc1_index)
    sc2 = lattice.cbuffer.get_object(sc2_index)

    if job.arch_str == "cpu":
        print(
            f"""
        sc0.data_addr = {sc0.interpol_data_addr:#018x} <- sc_data0 @ {sc_data0._get_address():#018x}
        sc1.data_addr = {sc1.interpol_data_addr:#018x} <- sc_data1 @ {sc_data1._get_address():#018x}
        sc2.data_addr = {sc2.interpol_data_addr:#018x} <- sc_data1 @ {sc_data1._get_address():#018x}
        """
        )
    else:
        print(
            f"""
        sc0.data_addr = {sc0.interpol_data_addr:#018x}
        sc1.data_addr = {sc1.interpol_data_addr:#018x}
        sc2.data_addr = {sc2.interpol_data_addr:#018x}"""
        )
