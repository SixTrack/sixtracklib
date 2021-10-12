import numpy as np
import ctypes as ct
from scipy.special import wofz as wofz_scipy
from sixtracklib.stcommon import st_cerrf_alg680_q1

if __name__ == '__main__':
    x_oe = np.logspace( -8, 8, 101, dtype=np.float64 )
    y_oe = np.logspace( -8, 8, 101, dtype=np.float64 )

    n_re = len( x_oe )
    n_im = len( y_oe )

    wz_re_cmp = np.arange( n_im * n_re, dtype=np.float64 ).reshape( n_im, n_re )
    wz_im_cmp = np.arange( n_im * n_re, dtype=np.float64 ).reshape( n_im, n_re )

    for jj, y in enumerate( y_oe ):
        for ii, x in enumerate( x_oe ):
            wz = wofz_scipy( x + 1.0j * y )
            wz_re_cmp[ jj, ii ] = wz.real
            wz_im_cmp[ jj, ii ] = wz.imag

    # --------------------------------------------------------------------------
    # ACM algorithm 680

    wz_re_alg680 = np.arange(
        n_im * n_re, dtype=np.float64 ).reshape( n_im, n_re )

    wz_im_alg680 = np.arange(
        n_im * n_re, dtype=np.float64 ).reshape( n_im, n_re )

    out_re = ct.c_double( 0. )
    out_im = ct.c_double( 0. )

    for jj, y in enumerate( y_oe ):
        for ii, x in enumerate( x_oe ):
            st_cerrf_alg680_q1( ct.c_double( x ), ct.c_double( y ),
                ct.byref( out_re ), ct.byref( out_im ) )
            wz_re_alg680[ jj, ii ] = np.float64( out_re )
            wz_im_alg680[ jj, ii ] = np.float64( out_im )

    assert np.fabs( wz_re_cmp - wz_re_alg680 ).max() < 0.5e-14
    assert np.fabs( wz_im_cmp - wz_im_alg680 ).max() < 0.5e-14
