import sixtracklib as st
import cobjects
import numpy as np
from cobjects import CBuffer

def gaussian_dist( z, mu=0.0, sigma=1.0):
    assert np.all( np.abs(sigma) > 0.0 )
    norm_z = ( z - mu ) / ( np.sqrt( 2.0 ) * sigma )
    return 1.0 / np.sqrt( 2 * np.pi * sigma ) * np.exp( -norm_z * norm_z )

if __name__ == '__main__':
    # Create buffer for storing the line density datasets:
    interpol_buffer = CBuffer()
    plot_data = True

    # Prepare two line density datasets from gaussian distributions:
    lin_absc = np.linspace(-8.0, 8.0, num=24, dtype="float64")
    lin_values = gaussian_dist(lin_absc, mu=0.0, sigma=1.0)
    line_prof_lin = st.LineDensityProfileData(
        cbuffer=interpol_buffer,
        values=lin_values,
        z0=lin_absc[0],
        dz=lin_absc[1]-lin_absc[0],
        method="linear")

    cub_absc = np.linspace(-8.0, 8.0, num=12, dtype="float64")
    cub_values = gaussian_dist(cub_absc, mu=0.0, sigma=2.0)
    line_prof_cub = st.LineDensityProfileData(
        cbuffer=interpol_buffer,
        values=cub_values,
        z0=cub_absc[0],
        dz=cub_absc[1] - cub_absc[0],
        method="cubic")

    import pdb
    pdb.set_trace()

    line_prof_lin = interpol_buffer.get_object( 0 )
    line_prof_lin.prepare_interpolation()

    line_prof_cub = interpol_buffer.get_object( 1 )
    line_prof_cub.prepare_interpolation()

    if plot_data:
        from matplotlib import pyplot as plt

        z_absc = np.linspace( -8.0, +8.0, num=512, dtype="float64" )
        y_exact1 = gaussian_dist(z_absc, mu=0.0, sigma=1.0)
        y_interp_lin = np.array( [ line_prof_lin.interpol( zz ) for zz in z_absc ] )

        y_exact2 = gaussian_dist(z_absc, mu=0.0, sigma=2.0)
        y_interp_cub = np.array( [ line_prof_cub.interpol( zz ) for zz in z_absc ] )

        plt.figure()
        plt.subplot(211)
        plt.plot(lin_absc, lin_values, 'bo', z_absc, y_exact1, 'k', z_absc, y_interp_lin, 'b-' )

        plt.subplot(212)
        plt.plot(cub_absc, cub_values, 'ro', z_absc, y_exact2, 'k', z_absc, y_interp_cub, 'r-' )
        plt.show()

    # Initialize the line density data with data from a gaussian distribution

    lattice = st.Elements()
    dr0 = lattice.Drift(length=1.0)
    q0 = lattice.Multipole(knl=[0.0, 0.1])
    sc0 = lattice.SpaceChargeInterpolatedProfile(num_particles=100.0)



