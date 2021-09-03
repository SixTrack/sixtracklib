#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_beamfields/dawson_approx.h"

    #if defined( SIXTRL_CERRF_USE_DAWSON_COEFF ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
        #include "sixtracklib/common/be_beamfields/dawson_coeff.h"
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 ) */

#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE )

int NS(dawson_n_interval_ext)( SIXTRL_REAL_T x ) SIXTRL_NOEXCEPT {
    return NS(dawson_n_interval)( x ); }

SIXTRL_REAL_T NS(dawson_xi_ext)( int n_interval ) SIXTRL_NOEXCEPT {
    return NS(dawson_xi)( n_interval ); }

SIXTRL_REAL_T NS(dawson_fz_xi_ext)( int n_interval ) SIXTRL_NOEXCEPT {
    return NS(dawson_fz_xi)( n_interval ); }

int NS(dawson_nt_xi_abs_d10_ext)( int n_interval ) SIXTRL_NOEXCEPT {
    return NS(dawson_nt_xi_abs_d10)( n_interval ); }

void NS(dawson_cerrf_nocoeff_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT
{
    NS(dawson_cerrf)( x, y, out_real, out_imag );
}

void NS(dawson_cerrf_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT
{
    #if defined( SIXTRL_CERRF_USE_DAWSON_COEFF ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )

        #if ( SIXTRL_CERRF_USE_DAWSON_COEFF == 1 )
            #if ( SIXTRL_CERRF_METHOD == 0 ) || ( SIXTRL_CERRF_METHOD == 1 )
                NS(dawson_cerrf_coeff)( x, y, out_real, out_imag,
                    &NS(CERRF_DAWSON_XI)[ 0 ], &NS(CERRF_DAWSON_FZ_XI)[ 0 ],
                        &NS(CERRF_DAWSON_NT_XI_ABS_D10)[ 0 ] );
            #else /* ( SIXTRL_CERRF_METHOD > 1 ) */
                NS(dawson_cerrf_coeff)( x, y, out_real, out_imag,
                    &NS(CERRF_DAWSON_XI)[ 0 ], &NS(CERRF_DAWSON_FZ_XI)[ 0 ],
                        &NS(CERRF_DAWSON_NT_XI_REL_D14)[ 0 ] );
            #endif /* ( SIXTRL_CERRF_METHOD ) */
        #else /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
            #if ( SIXTRL_CERRF_METHOD == 0 ) || ( SIXTRL_CERRF_METHOD == 1 )
                NS(dawson_cerrf_coeff)( x, y, out_real, out_imag,
                    &NS(CERRF_DAWSON_XI)[ 0 ], &NS(CERRF_DAWSON_FZ_XI)[ 0 ],
                        &NS(CERRF_DAWSON_NT_XI_ABS_D10)[ 0 ],
                            &NS(CERRF_DAWSON_FZ_KK_XI)[ 0 ] );
                                      );
            #else /* ( SIXTRL_CERRF_METHOD > 1 ) */
                NS(dawson_cerrf_coeff)( x, y, out_real, out_imag,
                    &NS(CERRF_DAWSON_XI)[ 0 ], &NS(CERRF_DAWSON_FZ_XI)[ 0 ],
                        &NS(CERRF_DAWSON_NT_XI_REL_D14)[ 0 ],
                            &NS(CERRF_DAWSON_FZ_KK_XI)[ 0 ] );
            #endif /* ( SIXTRL_CERRF_METHOD ) */
        #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 ) */
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 ) */
}

#endif /* !defined( _GPUCODE ) */

