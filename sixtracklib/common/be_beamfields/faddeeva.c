#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_beamfields/faddeeva.h"

    #if ( defined( SIXTRL_CERRF_ABQ2011_USE_TAYLOR_POLE_APPROX ) && \
                 ( SIXTRL_CERRF_ABQ2011_USE_TAYLOR_POLE_APPROX  == 1 ) ) || \
        ( defined( SIXTRL_CERRF_ABQ2011_USE_COEFF ) && \
                 ( SIXTRL_CERRF_ABQ2011_USE_COEFF == 1 ) )
        #include "sixtracklib/common/be_beamfields/abq2011_coeff.h"
    #endif /* SIXTRL_CERRF_ABQ2011_USE_TAYLOR_POLE_APPROX ||
              SIXTRL_CERRF_ABQ2011_USE_COEFF */

    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 )
        #include "sixtracklib/common/be_beamfields/dawson_approx.h"

        #if defined( SIXTRL_CERRF_USE_DAWSON_COEFF ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
            #include "sixtracklib/common/be_beamfields/dawson_coeff.h"
        #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 ) */
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE )

void NS(cerrf_cernlib_c_baseline_q1_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT
{
    NS(cerrf_cernlib_c_baseline_q1)( x, y, out_real, out_imag );
}

void NS(cerrf_cernlib_c_upstream_q1_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT
{
    NS(cerrf_cernlib_c_upstream_q1)( x, y, out_real, out_imag );
}

void NS(cerrf_cernlib_c_optimised_q1_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT
{
    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
        defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF   >= 1 )

            #if ( SIXTRL_CERRF_USE_DAWSON_COEFF == 1 )
                NS(cerrf_cernlib_c_optimised_q1)( x, y, out_real, out_imag,
                    &NS(CERRF_DAWSON_XI)[ 0 ],&NS(CERRF_DAWSON_FZ_XI)[ 0 ],
                        &NS(CERRF_DAWSON_NT_XI_REL_D14)[ 0 ] );

            #else /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
                NS(cerrf_cernlib_c_optimised_q1)( x, y, out_real, out_imag,
                    &NS(CERRF_DAWSON_XI)[ 0 ],&NS(CERRF_DAWSON_FZ_XI)[ 0 ],
                        &NS(CERRF_DAWSON_NT_XI_REL_D14)[ 0 ],
                            &NS(CERRF_DAWSON_FZ_KK_XI)[ 0 ] );

            #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 ) */
    #else /* !SIXTRL_CERRF_USE_DAWSON_APPROX || SIXTRL_CERRF_USE_DAWSON_COEFF == 0 */
        NS(cerrf_cernlib_c_optimised_q1)( x, y, out_real, out_imag );
    #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX && SIXTRL_CERRF_USE_DAWSON_COEFF */
}

void NS(cerrf_cernlib_c_optimised_fixed_q1_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT
{
    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
        defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF   >= 1 )

            #if ( SIXTRL_CERRF_USE_DAWSON_COEFF == 1 )
                NS(cerrf_cernlib_c_optimised_fixed_q1)( x, y, out_real, out_imag,
                    &NS(CERRF_DAWSON_XI)[ 0 ],&NS(CERRF_DAWSON_FZ_XI)[ 0 ],
                        &NS(CERRF_DAWSON_NT_XI_REL_D14)[ 0 ] );

            #else /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
                NS(cerrf_cernlib_c_optimised_fixed_q1)( x, y, out_real, out_imag,
                    &NS(CERRF_DAWSON_XI)[ 0 ],&NS(CERRF_DAWSON_FZ_XI)[ 0 ],
                        &NS(CERRF_DAWSON_NT_XI_REL_D14)[ 0 ],
                            &NS(CERRF_DAWSON_FZ_KK_XI)[ 0 ] );

            #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 ) */
    #else /* !SIXTRL_CERRF_USE_DAWSON_APPROX || SIXTRL_CERRF_USE_DAWSON_COEFF == 0 */
        NS(cerrf_cernlib_c_optimised_fixed_q1)( x, y, out_real, out_imag );
    #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX && SIXTRL_CERRF_USE_DAWSON_COEFF */
}

void NS(cerrf_alg680_q1_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT
{
    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
        defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF   >= 1 )

            #if ( SIXTRL_CERRF_USE_DAWSON_COEFF == 1 )
                NS(cerrf_alg680_q1)( x, y, out_real, out_imag,
                    &NS(CERRF_DAWSON_XI)[ 0 ],&NS(CERRF_DAWSON_FZ_XI)[ 0 ],
                        &NS(CERRF_DAWSON_NT_XI_REL_D14)[ 0 ] );

            #else /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
                NS(cerrf_alg680_q1)( x, y, out_real, out_imag,
                    &NS(CERRF_DAWSON_XI)[ 0 ],&NS(CERRF_DAWSON_FZ_XI)[ 0 ],
                        &NS(CERRF_DAWSON_NT_XI_REL_D14)[ 0 ],
                            &NS(CERRF_DAWSON_FZ_KK_XI)[ 0 ] );

            #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 ) */
    #else /* !SIXTRL_CERRF_USE_DAWSON_APPROX || SIXTRL_CERRF_USE_DAWSON_COEFF == 0 */
        NS(cerrf_alg680_q1)( x, y, out_real, out_imag );
    #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX && SIXTRL_CERRF_USE_DAWSON_COEFF */
}

SIXTRL_REAL_T NS(cerrf_abq2011_a_m_coeff_ext)( int const m ) SIXTRL_NOEXCEPT {
    return NS(cerrf_abq2011_a_m_coeff)( m ); }

void NS(cerrf_abq2011_q1_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT
{
    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
        defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF   >= 1 )

            #if ( SIXTRL_CERRF_USE_DAWSON_COEFF == 1 )
                NS(cerrf_abq2011_q1)( x, y, out_real, out_imag,
                    &NS(CERRF_DAWSON_XI)[ 0 ],&NS(CERRF_DAWSON_FZ_XI)[ 0 ],
                        &NS(CERRF_DAWSON_NT_XI_REL_D14)[ 0 ] );

            #else /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
                NS(cerrf_abq2011_q1)( x, y, out_real, out_imag,
                    &NS(CERRF_DAWSON_XI)[ 0 ],&NS(CERRF_DAWSON_FZ_XI)[ 0 ],
                        &NS(CERRF_DAWSON_NT_XI_REL_D14)[ 0 ],
                            &NS(CERRF_DAWSON_FZ_KK_XI)[ 0 ] );

            #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 ) */
    #else /* !SIXTRL_CERRF_USE_DAWSON_APPROX || SIXTRL_CERRF_USE_DAWSON_COEFF == 0 */
        NS(cerrf_abq2011_q1)( x, y, out_real, out_imag );
    #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX && SIXTRL_CERRF_USE_DAWSON_COEFF */
}

void NS(cerrf_abq2011_q1_coeff_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag,
    SIXTRL_CERRF_ABQ2011_FOURIER_COEFF_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT a_m,
    SIXTRL_CERRF_ABQ2011_TAYLOR_COEFF_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT b_n
) SIXTRL_NOEXCEPT
{
    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
        defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF   >= 1 )

            #if ( SIXTRL_CERRF_USE_DAWSON_COEFF == 1 )
                NS(cerrf_abq2011_q1_coeff)( x, y, out_real, out_imag, a_m,
                    b_n, &NS(CERRF_DAWSON_XI)[ 0 ],&NS(CERRF_DAWSON_FZ_XI)[ 0 ],
                        &NS(CERRF_DAWSON_NT_XI_REL_D14)[ 0 ] );

            #else /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
                NS(cerrf_abq2011_q1_coeff)( x, y, out_real, out_imag, a_m,
                    b_n, &NS(CERRF_DAWSON_XI)[ 0 ],&NS(CERRF_DAWSON_FZ_XI)[ 0 ],
                        &NS(CERRF_DAWSON_NT_XI_REL_D14)[ 0 ],
                            &NS(CERRF_DAWSON_FZ_KK_XI)[ 0 ] );

            #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 ) */
    #else /* !SIXTRL_CERRF_USE_DAWSON_APPROX || SIXTRL_CERRF_USE_DAWSON_COEFF == 0 */
        NS(cerrf_abq2011_q1_coeff)( x, y, out_real, out_imag, a_m, b_n );
    #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX && SIXTRL_CERRF_USE_DAWSON_COEFF */
}

void NS(cerrf_q1_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT
{
    #if ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_CERNLIB_BASELINE )
    NS(cerrf_cernlib_c_baseline_q1_ext)( x, y, out_real, out_imag );
    #elif ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_CERNLIB_UPSTREAM )
    NS(cerrf_cernlib_c_upstream_q1_ext)( x, y, out_real, out_imag );
    #elif ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_CERNLIB_FIXED )
    NS(cerrf_cernlib_c_optimised_q1_fixed_ext)( x, y, out_real, out_imag );
    #elif ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_ALG680 )
    NS(cerrf_alg680_q1_ext)( x, y, out_real, out_imag );
    #elif ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_ABQ2011 )
    NS(cerrf_abq2011_q1_ext)( x, y, out_real, out_imag );
    #else /* SIXTRL_CERRF_METHOD == SIXTRL_CERRF_CERNLIB_OPTIMISED */
    NS(cerrf_cernlib_c_optimised_q1_ext)( x, y, out_real, out_imag );
    #endif /* SIXTRL_CERRF_METHOD */
}

void NS(cerrf_ext)(
    SIXTRL_REAL_T x, SIXTRL_REAL_T y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_x,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_y
) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T real_type;

    real_type const sign_x = ( real_type )( ( x >= ( real_type )0. ) -
                                            ( x <  ( real_type )0. ) );

    real_type const sign_y = ( real_type )( ( y >= ( real_type )0. ) -
                                            ( y <  ( real_type )0. ) );

    SIXTRL_CERRF_RESULT_DEC real_type Wx;
    SIXTRL_CERRF_RESULT_DEC real_type Wy;

    x *= sign_x;
    y *= sign_y;

    #if ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_CERNLIB_BASELINE )
    NS(cerrf_cernlib_c_baseline_q1_ext)( x, y, &Wx, &Wy );
    #elif ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_CERNLIB_UPSTREAM )
    NS(cerrf_cernlib_c_upstream_q1_ext)( x, y, &Wx, &Wy );
    #elif ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_CERNLIB_FIXED )
    NS(cerrf_cernlib_c_upstream_fixed_q1_ext)( x, y, &Wx, &Wy );
    #elif ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_ALG680 )
    NS(cerrf_alg680_q1_ext)( x, y, &Wx, &Wy );
    #elif ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_ABQ2011 )
    NS(cerrf_abq2011_q1_ext)( x, y, &Wx, &Wy );
    #else /* SIXTRL_CERRF_METHOD == SIXTRL_CERRF_CERNLIB_OPTIMISED */
    NS(cerrf_cernlib_c_optimised_q1_ext)( x, y, &Wx, &Wy );
    #endif /* SIXTRL_CERRF_METHOD */

    if( sign_y < ( real_type )0.0 )  /* Quadrants Q3 and Q4 */
    {
        real_type const exp_arg  = ( y - x ) * ( y + x );
        real_type const trig_arg = ( real_type )2. * x * y;
        real_type const exp_factor = ( real_type )2. * exp( exp_arg );

        SIXTRL_RESULT_PTR_DEC real_type sin_arg;
        SIXTRL_RESULT_PTR_DEC real_type cos_arg;

        NS(sincos)( trig_arg, &sin_arg, &cos_arg );
        Wx = exp_factor * cos_arg - Wx;
        Wy = exp_factor * sin_arg + Wy;
    }

    *out_x = Wx;
    *out_y = sign_x * Wy; /* Takes care of Quadrants Q2 and Q3 */
}

#endif /* !defined( _GPUCODE ) */
