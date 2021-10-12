#ifndef SIXTACKLIB_COMMON_BE_BEAMFIELDS_FADDEEVA_HEADER_H__
#define SIXTACKLIB_COMMON_BE_BEAMFIELDS_FADDEEVA_HEADER_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES ) && !defined( __cplusplus )
    #include <stdbool.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) && !defined( __cplusplus ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_beamfields/definitions.h"
    #include "sixtracklib/common/internal/math_constants.h"
    #include "sixtracklib/common/internal/math_functions.h"
    #include "sixtracklib/opencl/helpers.h"

    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 )
        #include "sixtracklib/common/be_beamfields/dawson_approx.h"
        #if defined( SIXTRL_CERRF_USE_DAWSON_COEFF ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
            #include "sixtracklib/common/be_beamfields/dawson_coeff.h"
        #endif  /* ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 ) */
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* !defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN void NS(cerrf_cernlib_c_baseline_q1)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(cerrf_cernlib_c_upstream_q1)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(cerrf_cernlib_c_optimised_q1)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
        defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_XI_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT xi,
    SIXTRL_CERRF_DAWSON_COEFF_FZ_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT Fz_xi,
    SIXTRL_CERRF_DAWSON_COEFF_NT_DEC SIXTRL_INT32_TYPE const* SIXTRL_RESTRICT Fz_nt
    #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_TAYLOR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT Fz_kk_xi
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
    #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX &&  SIXTRL_CERRF_USE_DAWSON_COEFF */
) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(cerrf_cernlib_c_optimised_fixed_q1)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
        defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_XI_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT xi,
    SIXTRL_CERRF_DAWSON_COEFF_FZ_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT Fz_xi,
    SIXTRL_CERRF_DAWSON_COEFF_NT_DEC SIXTRL_INT32_TYPE const* SIXTRL_RESTRICT Fz_nt
    #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_TAYLOR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT Fz_kk_xi
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
    #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX &&  SIXTRL_CERRF_USE_DAWSON_COEFF */
) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(cerrf_alg680_q1)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
        defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_XI_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT xi,
    SIXTRL_CERRF_DAWSON_COEFF_FZ_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT Fz_xi,
    SIXTRL_CERRF_DAWSON_COEFF_NT_DEC SIXTRL_INT32_TYPE const* SIXTRL_RESTRICT Fz_nt
    #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_TAYLOR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT Fz_kk_xi
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
    #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX &&  SIXTRL_CERRF_USE_DAWSON_COEFF */
) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(cerrf_abq2011_a_m_coeff)(
    int const m ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(cerrf_abq2011_q1)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
        defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_XI_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT xi,
    SIXTRL_CERRF_DAWSON_COEFF_FZ_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT Fz_xi,
    SIXTRL_CERRF_DAWSON_COEFF_NT_DEC SIXTRL_INT32_TYPE const* SIXTRL_RESTRICT Fz_nt
    #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_TAYLOR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT Fz_kk_xi
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
    #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX &&  SIXTRL_CERRF_USE_DAWSON_COEFF */
) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(cerrf_abq2011_q1_coeff)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag,
    SIXTRL_CERRF_ABQ2011_FOURIER_COEFF_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT a_m,
    SIXTRL_CERRF_ABQ2011_TAYLOR_COEFF_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT b_m
    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
        defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_XI_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT xi,
    SIXTRL_CERRF_DAWSON_COEFF_FZ_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT Fz_xi,
    SIXTRL_CERRF_DAWSON_COEFF_NT_DEC SIXTRL_INT32_TYPE const* SIXTRL_RESTRICT Fz_nt
    #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_TAYLOR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT Fz_kk_xi
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
    #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX &&  SIXTRL_CERRF_USE_DAWSON_COEFF */
) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(cerrf_q1)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
        defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_XI_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT xi,
    SIXTRL_CERRF_DAWSON_COEFF_FZ_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT Fz_xi,
    SIXTRL_CERRF_DAWSON_COEFF_NT_DEC SIXTRL_INT32_TYPE const* SIXTRL_RESTRICT Fz_nt
    #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_TAYLOR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT Fz_kk_xi
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
    #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX &&  SIXTRL_CERRF_USE_DAWSON_COEFF */
) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(cerrf)(
    SIXTRL_REAL_T in_real, SIXTRL_REAL_T in_imag,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
        defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_XI_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT xi,
    SIXTRL_CERRF_DAWSON_COEFF_FZ_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT Fz_xi,
    SIXTRL_CERRF_DAWSON_COEFF_NT_DEC SIXTRL_INT32_TYPE const* SIXTRL_RESTRICT Fz_nt
    #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_TAYLOR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT Fz_kk_xi
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
    #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX &&  SIXTRL_CERRF_USE_DAWSON_COEFF */
) SIXTRL_NOEXCEPT;


#if !defined( _GPUCODE )
SIXTRL_EXTERN SIXTRL_HOST_FN void  NS(cerrf_cernlib_c_baseline_q1_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(cerrf_cernlib_c_upstream_q1_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(cerrf_cernlib_c_optimised_q1_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(cerrf_cernlib_c_optimised_fixed_q1_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(cerrf_alg680_q1_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_REAL_T
NS(cerrf_abq2011_a_m_coeff_ext)( int const m ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(cerrf_abq2011_q1_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(cerrf_abq2011_cf_q1_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(cerrf_abq2011_cf_daw_q1_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(cerrf_abq2011_root_q1_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(cerrf_abq2011_q1_coeff_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag,
    SIXTRL_CERRF_ABQ2011_FOURIER_COEFF_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT a_m,
    SIXTRL_CERRF_ABQ2011_TAYLOR_COEFF_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT b_m
) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(cerrf_q1_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(cerrf_ext)(
    SIXTRL_REAL_T in_real, SIXTRL_REAL_T in_imag,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT;
#endif /* !defined( _GPUCODE ) */

/* ************************************************************************* */

SIXTRL_INLINE void NS(cerrf_cernlib_c_baseline_q1)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T real_type;

    /* This function calculates the SIXTRL_REAL_T precision complex error fnct.
    based on the algorithm of the FORTRAN function written at CERN by K. Koelbig
    Program C335, 1970. See also M. Bassetti and G.A. Erskine, "Closed
    expression for the electric field of a two-dimensional Gaussian charge
    density", CERN-ISR-TH/80-06; */

    int n, nc, nu;
    real_type const a_constant = ( real_type )1.12837916709551;
    real_type const xLim = ( real_type )5.33;
    real_type const yLim = ( real_type )4.29;
    real_type h, q, Saux, Sx, Sy, Tn, Tx, Ty, Wx, Wy, xh, xl, yh;
    real_type Rx[ 33 ];
    real_type Ry[ 33 ];

    if( ( y < yLim ) && ( x < xLim ) )
    {
        q = ( ( real_type )1.0 - y / yLim ) *
            NS(sqrt)( ( real_type )1.0 - ( x / xLim ) * ( x / xLim ) );
        h  = 1.0 / (3.2 * q);
        nc = 7 + (int) (23.0 * q);
        xl = NS(pow)(h, ( real_type )( 1 - nc ) );
        xh = y + 0.5 / h;
        yh = x;
        nu = 10 + (int) (21.0 * q);
        Rx[nu] = 0.;
        Ry[nu] = 0.;
        for (n = nu; n > 0; n--){
            Tx = xh + n * Rx[n];
            Ty = yh - n * Ry[n];
            Tn = Tx*Tx + Ty*Ty;
            Rx[n-1] = 0.5 * Tx / Tn;
            Ry[n-1] = 0.5 * Ty / Tn;
            }
        /* .... */
        Sx = 0.;
        Sy = 0.;
        for (n = nc; n>0; n--){
            Saux = Sx + xl;
            Sx = Rx[n-1] * Saux - Ry[n-1] * Sy;
            Sy = Rx[n-1] * Sy + Ry[n-1] * Saux;
            xl = h * xl;
        };
        Wx = a_constant * Sx;
        Wy = a_constant * Sy;
    }
    else{
        xh = y;
        yh = x;
        Rx[0] = 0.;
        Ry[0] = 0.;
        for (n = 9; n>0; n--){
            Tx = xh + n * Rx[0];
            Ty = yh - n * Ry[0];
            Tn = Tx * Tx + Ty * Ty;
            Rx[0] = 0.5 * Tx / Tn;
            Ry[0] = 0.5 * Ty / Tn;
        };
        Wx = a_constant * Rx[0];
        Wy = a_constant * Ry[0];
    }
    if (y == 0.) {Wx = NS(exp)(-x * x);}

    *out_real = Wx;
    *out_imag = Wy;
}

SIXTRL_INLINE void NS(cerrf_cernlib_c_upstream_q1)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T real_type;

    /* This function calculates the SIXTRL_REAL_T precision complex error fnct.
    based on the algorithm of the FORTRAN function written at CERN by K. Koelbig
    Program C335, 1970. See also M. Bassetti and G.A. Erskine, "Closed
    expression for the electric field of a two-dimensional Gaussian charge
    density", CERN-ISR-TH/80-06; */

    int n  = ( int )0u;
    int N  = ( int )0u;
    int nu = ( int )SIXTRL_CERRF_CERNLIB_UPSTREAM_K;

    real_type h         = ( real_type )0.0;
    real_type two_h_n   = ( real_type )0.0;
    real_type inv_two_h = ( real_type )1.0;
    real_type y_plus_h  = y;

    real_type Saux, Sx, Sy, Tn, Tx, Ty, Wx, Wy;
    real_type Rx[ SIXTRL_CERRF_CERNLIB_UPSTREAM_NMAX ];
    real_type Ry[ SIXTRL_CERRF_CERNLIB_UPSTREAM_NMAX ];

    if( ( y < ( real_type )SIXTRL_CERRF_CERNLIB_UPSTREAM_Y0 ) &&
        ( x < ( real_type )SIXTRL_CERRF_CERNLIB_UPSTREAM_X0 ) )
    {
        N         = ( int )SIXTRL_CERRF_CERNLIB_UPSTREAM_N;
        nu        = ( int )SIXTRL_CERRF_CERNLIB_UPSTREAM_NU;
        h         = ( real_type )SIXTRL_CERRF_CERNLIB_UPSTREAM_H_0;

        two_h_n   = ( real_type )2. * h;
        y_plus_h += h;
        inv_two_h = ( real_type )1. / two_h_n;
        two_h_n   = NS(pow)( two_h_n, N - 1 );

        Rx[ nu ] = ( real_type )0.;
        Ry[ nu ] = ( real_type )0.;

        for( n = nu ; n > 0 ; --n )
        {
            Tx          = y_plus_h + n * Rx[ n ];
            Ty          = x        - n * Ry[ n ];
            Tn          = ( Tx * Tx ) + ( Ty * Ty );
            Rx[ n - 1 ] = ( real_type )0.5 * Tx / Tn;
            Ry[ n - 1 ] = ( real_type )0.5 * Ty / Tn;
        }

        Sx = Sy = ( real_type )0.0;

        for( n = N; n > 0 ; --n )
        {
            Saux     = Sx + two_h_n;
            two_h_n *= inv_two_h;
            Sx       = Rx[ n - 1 ] * Saux - Ry[ n - 1 ] * Sy;
            Sy       = Rx[ n - 1 ] * Sy   + Ry[ n - 1 ] * Saux;
        }

        Wx = NS(MathConst_two_over_sqrt_pi)() * Sx;
        Wy = NS(MathConst_two_over_sqrt_pi)() * Sy;
    }
    else
    {
        Rx[ 0 ] = Ry[ 0 ] = ( real_type )0.0;
        n = ( int )SIXTRL_CERRF_CERNLIB_UPSTREAM_K;

        for( ; n > 0 ; --n )
        {
            Tx      = y_plus_h    + ( n  * Rx[ 0 ] );
            Ty      = x           - ( n  * Ry[ 0 ] );
            Tn      = ( Tx * Tx ) + ( Ty * Ty );
            Rx[ 0 ] = ( real_type )0.5 * Tx / Tn;
            Ry[ 0 ] = ( real_type )0.5 * Ty / Tn;
        }

        Wx = NS(MathConst_two_over_sqrt_pi)() * Rx[0];
        Wy = NS(MathConst_two_over_sqrt_pi)() * Ry[0];
    }

    if( y < ( real_type )SIXTRL_CERRF_CERNLIB_UPSTREAM_MIN_Y )
        Wx = NS(exp)( -x * x );

    *out_real = Wx;
    *out_imag = Wy;
}

/** \fn void cerrf_cernlib_c_optimised_q1( double const, double const, double*, double* )
 *  \brief calculates the Faddeeva function w(z) for z = x + i * y in Q1
 *
 *  \param[in] x real component of argument z
 *  \param[in] y imaginary component of argument z
 *  \param[out] out_x pointer to real component of result
 *  \param[out] out_y pointer to imanginary component of result
 *
 *  \warning This function assumes that x and y are > 0 i.e., that z is
 *           from the first quadrant Q1 of the complex plane. Use cerrf if
 *           you need a more general function
 *
 *  \note    Based upon the algorithm developed by W. Gautschi 1970,
 *           "Efficient Computation of the Complex Error Function",
 *           SIAM Journal on Numerical Analysis, Vol. 7, Issue 1. 1970,
 *           pages 187-198, https://epubs.siam.org/doi/10.1137/0707012
 */

SIXTRL_INLINE void NS(cerrf_cernlib_c_optimised_q1)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_x,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_y
    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
        defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_XI_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT xi,
    SIXTRL_CERRF_DAWSON_COEFF_FZ_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT Fz_xi,
    SIXTRL_CERRF_DAWSON_COEFF_NT_DEC SIXTRL_INT32_TYPE const* SIXTRL_RESTRICT Fz_nt
    #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_TAYLOR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT Fz_kk_xi
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 ) */
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T real_type;

    /* This implementation corresponds closely to the previously used
     * "CERNLib C" version, translated from the FORTRAN function written at
     * CERN by K. Koelbig, Program C335, 1970. The main difference to
     * Gautschi's formulation is a split in the main loop and the introduction
     * of arrays to store the intermediate results as a consequence of this.
     * The version implemented here should perform roughly equally well or even
     * slightly better on modern out-of-order super-scalar CPUs but has
     * drastically improved performance on GPUs and GPU-like systems.
     *
     * See also M. Bassetti and G.A. Erskine,
     * "Closed expression for the electric field of a two-dimensional Gaussian
     *  charge density", CERN-ISR-TH/80-06; */

    real_type inv_h2   = ( real_type )1.0;
    real_type y_plus_h = y;
    real_type temp, Rx, Ry, Sx, Sy, Wx, Wy, h2_n, nn;

    int nu = ( int )SIXTRL_CERRF_CERNLIB_K;
    int N  = 0;
    int n  = 0;

    bool use_taylor_sum = (
        ( y < ( real_type )SIXTRL_CERRF_CERNLIB_Y0 ) &&
        ( x < ( real_type )SIXTRL_CERRF_CERNLIB_X0 ) );

    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 )
    bool const use_dawson_approx = (
        ( x >= ( real_type )SIXTRL_CERRF_CERNLIB_DAWSON_APPROX_MIN_X ) &&
        ( x <= ( real_type )SIXTRL_CERRF_CERNLIB_DAWSON_APPROX_MAX_X ) &&
        ( y <= ( real_type )SIXTRL_CERRF_CERNLIB_USE_DAWSON_APPROX_MAX_Y ) );
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 ) */

    #if defined( _GPUCODE ) && defined( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N )

        #if ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N == 2 ) || \
            ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N == 3 )
            unsigned int nu_minus_n;
            SIXTRL_SHARED_DEC unsigned int nu_minus_n_w[ SIXTRL_WORKGROUP_SIZE ];
        #endif /* ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N >= 2 ) */

        #if ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N == 3 )
            unsigned int N_max;
            SIXTRL_SHARED_DEC unsigned int n_w[ SIXTRL_WORKGROUP_SIZE ];
        #endif /* ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N == 3 ) */

        #if ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N == 4 )
            uint2 local_nu_n;
            SIXTRL_SHARED_DEC uint2 nu_n[ SIXTRL_WORKGROUP_SIZE ];
        #endif /* SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N == 3 ) */

    #endif

    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 )
    if( use_dawson_approx )
    {
        use_taylor_sum = false;
        nu = 0;
    }
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 ) */

    Rx = Ry = Sx = Sy = h2_n = ( real_type )0.0;

    /* R_0 ... rectangle with width SIXTRL_CERRF_CERNLIB_X0 and
     *         height SIXTRL_CERRF_CERNLIB_Y0. Inside R_0, w(z) is calculated using
     *         a truncated Taylor expansion. Outside, a Gauss--Hermite
     *         quadrature in the guise of a continuos fraction is used */

	if( use_taylor_sum )
    {
        #if !defined( SIXTRL_CERRF_CERNLIB_NO_GZ_WEIGHT_FN ) || \
                    ( SIXTRL_CERRF_CERNLIB_NO_GZ_WEIGHT_FN == 0 )
        /* calculate g(z) = sqrt( 1 - (x/x0)^2 ) * ( 1 - y/y0 ) */
        temp  = x * ( real_type )SIXTRL_CERRF_CERNLIB_INV_X0;
        temp  = ( ( real_type )1. +  temp ) * ( ( real_type )1. - temp );
        temp  = sqrt( temp );

        temp *= ( ( real_type )1. - y * (
                    real_type )SIXTRL_CERRF_CERNLIB_INV_Y0 );
        /*now: temp = g(z) */

        #elif ( SIXTRL_CERRF_CERNLIB_NO_GZ_WEIGHT_FN == 1 ) && \
              defined( SIXTRL_CERRF_CERNLIB_GZ_WEIGHT_VALUE )
        temp = ( real_type )SIXTRL_CERRF_CERNLIB_GZ_WEIGHT_VALUE;

        #else /* !defined( FADDEEVA_NO_GZ_WEIGHT_FN ) */
        temp = ( real_type )1.;

        #endif /* defined( FADDEEVA_NO_GZ_WEIGHT_FN ) */

        h2_n      = ( real_type )SIXTRL_CERRF_CERNLIB_H_0 * temp;
        y_plus_h += h2_n;
        h2_n     *= ( real_type )2.;
        inv_h2    = ( real_type )1. / h2_n;

        N = ( int )SIXTRL_CERRF_CERNLIB_N_0 +
            ( int )( ( double )SIXTRL_CERRF_CERNLIB_N_1 * temp );

        #if !defined( _GPUCODE ) || \
            !defined( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N ) || \
                    ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N == 0 ) || \
                    ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N == 1 ) || \
                    ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N == 2 )

        h2_n = NS(pow_int_exp)( h2_n, N - 1 );
        use_taylor_sum = ( h2_n > ( real_type )SIXTRL_CERRF_CERNLIB_MIN_POW_2H_N );

        #endif /* ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N <= 2 ) */

        #if !defined( _GPUCODE ) || \
            !defined( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N ) || \
                    ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N != 1 )

            #if  defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
                        ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 )
            nu   = ( int )SIXTRL_CERRF_CERNLIB_NU_0 +
                   ( int )( ( double )SIXTRL_CERRF_CERNLIB_NU_1 * temp );

            #else
            nu   = ( y > ( real_type )SIXTRL_CERRF_CERNLIB_MIN_Y )
                 ? ( int )SIXTRL_CERRF_CERNLIB_NU_0 +
                   ( int )( ( double )SIXTRL_CERRF_CERNLIB_NU_1 * temp )
                 : 0;
            #endif /* ( SIXTRL_CERRF_USE_DAWSON_APPROX ) */

        #elif ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N  == 1 )
            #if  defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
                        ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 )
            nu  = N + ( int )SIXTRL_CERRF_CERNLIB_K;
            #else
            nu   = ( y > ( real_type )SIXTRL_CERRF_CERNLIB_MIN_Y )
                 ? N + ( int )SIXTRL_CERRF_CERNLIB_K : 0;
            #endif /* ( SIXTRL_CERRF_USE_DAWSON_APPROX ) */

        #endif /* SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N */
    }

    #if defined( _GPUCODE ) && \
        defined( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N ) && \
               ( ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N == 2 ) || \
                 ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N == 3 ) )

        SIXTRL_ASSERT( nu >= N );
        SIXTRL_SHARED_BUILD_ARRAY( unsigned int, nu_minus_n_w, nu - N );
        SIXTRL_SHARED_FIND_MAX_PER_W( unsigned int, nu_minus_n_w,
                                    ( unsigned int )SIXTRL_W_SIZE, nu_minus_n );

        #if ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N == 3 )
            SIXTRL_SHARED_BUILD_ARRAY( unsigned int, n_w, N );
            SIXTRL_SHARED_FIND_MAX_PER_W( unsigned int, n_w,
                                        ( unsigned int )SIXTRL_W_SIZE, N_max );

            N = ( use_taylor_sum ) ? ( int )N_max : ( int )0;

        #endif /* ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N == 3 ) */
        nu = nu_minus_n + N;

    #endif /* ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N <= 3 ) */

    #if defined( _GPUCODE ) && \
        defined( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N ) && \
               ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N == 4 )
        SIXTRL_ASSERT( nu >= N );
        local_nu_n.x = nu - N;
        local_nu_n.y = N;

        SIXTRL_SHARED_BUILD_ARRAY( uint2, nu_n, local_nu_n );
        SIXTRL_SHARED_FIND_MAX_PER_W( uint2, nu_n,
            ( unsigned int )SIXTRL_W_SIZE, local_nu_n );

        N  = ( use_taylor_sum ) ? ( int )local_nu_n.y : ( int )0;
        nu = ( int )local_nu_n.x + N;

    #endif /* ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N == 4 ) */

    /* If h(z) is so close to 0 that it is practically 0, there is no
     * point in doing the extra work for the Taylor series -> in that
     * very unlikely case, use the continuos fraction & verify result! */

    #if defined( _GPUCODE ) && \
        defined( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N ) && \
               ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N == 3 ) || \
               ( SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N == 4 )

    if( use_taylor_sum ) h2_n = NS(pow_int_exp)( h2_n, N - 1 );
    use_taylor_sum = ( h2_n > ( real_type )SIXTRL_CERRF_CERNLIB_MIN_POW_2H_N );

    #endif  /* SIXTRL_CERRF_CERNLIB_FIND_MAX_NU_N */

    #if !defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) || \
                ( SIXTRL_CERRF_USE_DAWSON_APPROX != 1 )
    if( y <= ( real_type )SIXTRL_CERRF_CERNLIB_MIN_Y )
        Rx = exp( -x * x ) / NS(MathConst_two_over_sqrt_pi)();

    #else /* ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 ) */
    if( !use_dawson_approx )
    {
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 ) */

        n = nu;
        nn = ( real_type )n;

        /* z outside of R_0: continued fraction / Gauss - Hermite quadrature
        * z inside  of R_0: first iterations of recursion until n == N */
        for( ; n > N ; --n, nn -= ( real_type )1.0 )
        {
            Wx     = y_plus_h + nn * Rx;
            Wy     = x - nn * Ry;
            temp   = ( Wx * Wx ) + ( Wy * Wy );
            Rx     = ( real_type )0.5 * Wx;
            Ry     = ( real_type )0.5 * Wy;
            temp   = ( real_type )1.0 / temp;
            Rx    *= temp;
            Ry    *= temp;
        }

        /* loop rejects everything if z is not in R_0 because then n == 0 already;
        * otherwise, N iterations until taylor expansion is summed up */
        for( ; n > 0 ; --n, nn -= ( real_type )1.0 )
        {
            Wx     = y_plus_h + nn * Rx;
            Wy     = x - nn * Ry;
            temp   = ( Wx * Wx ) + ( Wy * Wy );
            Rx     = ( real_type )0.5 * Wx;
            Ry     = ( real_type )0.5 * Wy;
            temp   = ( real_type )1.0 / temp;
            Rx    *= temp;
            Ry    *= temp;

            Wx     = h2_n + Sx;
            h2_n  *= inv_h2;
            Sx     = Rx * Wx - Ry * Sy;
            Sy     = Ry * Wx + Rx * Sy;
        }

        *out_x = NS(MathConst_two_over_sqrt_pi)() * (
            ( use_taylor_sum ) ? Sx : Rx );

        *out_y = NS(MathConst_two_over_sqrt_pi)() * (
            ( use_taylor_sum ) ? Sy : Ry );

    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 )
    }
    else
    {
        #if defined( SIXTRL_CERRF_USE_DAWSON_COEFF ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_COEFF == 1 )
        NS(dawson_cerrf_coeff)( x, y, out_x, out_y, xi, Fz_xi, Fz_nt );
        #elif defined( SIXTRL_CERRF_USE_DAWSON_COEFF ) && \
                     ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
        NS(dawson_cerrf_coeff)( x, y, out_x, out_y, xi, Fz_xi, Fz_nt, Fz_kk_xi );
        #else
        NS(dawson_cerrf)( x, y, out_x, out_y );
        #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF ) */
    }
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 ) */
}


/** \fn void cerrf_cernlib_c_optimised_fixed_q1( double const, double const, double*, double* )
 *  \brief calculates the Faddeeva function w(z) for z = x + i * y in Q1
 *
 *  \param[in] x real component of argument z
 *  \param[in] y imaginary component of argument z
 *  \param[out] out_x pointer to real component of result
 *  \param[out] out_y pointer to imanginary component of result
 *
 *  \warning This function assumes that x and y are > 0 i.e., that z is
 *           from the first quadrant Q1 of the complex plane. Use cerrf if
 *           you need a more general function
 *
 *  \note    Based upon the algorithm developed by W. Gautschi 1970,
 *           "Efficient Computation of the Complex Error Function",
 *           SIAM Journal on Numerical Analysis, Vol. 7, Issue 1. 1970,
 *           pages 187-198, https://epubs.siam.org/doi/10.1137/0707012
 */

SIXTRL_INLINE void NS(cerrf_cernlib_c_optimised_fixed_q1)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_x,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_y
    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
        defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_XI_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT xi,
    SIXTRL_CERRF_DAWSON_COEFF_FZ_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT Fz_xi,
    SIXTRL_CERRF_DAWSON_COEFF_NT_DEC SIXTRL_INT32_TYPE const* SIXTRL_RESTRICT Fz_nt
    #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_TAYLOR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT Fz_kk_xi
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 ) */
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T real_type;

    /* This implementation corresponds closely to the previously used
     * "CERNLib C" version, translated from the FORTRAN function written at
     * CERN by K. Koelbig, Program C335, 1970. The main difference to
     * Gautschi's formulation is a split in the main loop and the introduction
     * of arrays to store the intermediate results as a consequence of this.
     * The version implemented here should perform roughly equally well or even
     * slightly better on modern out-of-order super-scalar CPUs but has
     * drastically improved performance on GPUs and GPU-like systems.
     *
     * See also M. Bassetti and G.A. Erskine,
     * "Closed expression for the electric field of a two-dimensional Gaussian
     *  charge density", CERN-ISR-TH/80-06; */

    real_type h2_n;
    real_type inv_h2 = ( real_type )1.0;
    real_type y_plus_h = y;
    int N = 0;
    int nu;

    bool use_taylor_sum = (
        ( y < ( real_type )SIXTRL_CERRF_CERNLIB_Y0 ) &&
        ( x < ( real_type )SIXTRL_CERRF_CERNLIB_X0 ) );

    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 )
    bool const use_dawson_approx = (
        ( x >= ( real_type )SIXTRL_CERRF_CERNLIB_DAWSON_APPROX_MIN_X ) &&
        ( x <= ( real_type )SIXTRL_CERRF_CERNLIB_DAWSON_APPROX_MAX_X ) &&
        ( y <= ( real_type )SIXTRL_CERRF_CERNLIB_USE_DAWSON_APPROX_MAX_Y ) );
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 ) */

    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 )
    use_taylor_sum &= !use_dawson_approx;
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 ) */

    /* R_0 ... rectangle with width SIXTRL_CERRF_CERNLIB_X0 and
     *         height SIXTRL_CERRF_CERNLIB_Y0. Inside R_0, w(z) is calculated using
     *         a truncated Taylor expansion. Outside, a Gauss--Hermite
     *         quadrature in the guise of a continuos fraction is used */

	if( use_taylor_sum )
    {
        y_plus_h += ( real_type )SIXTRL_CERRF_CERNLIB_H_0;
        h2_n      = ( real_type )2. * ( real_type )SIXTRL_CERRF_CERNLIB_H_0;
        inv_h2    = ( real_type )1. / h2_n;

        N    = ( int )SIXTRL_CERRF_CERNLIB_UPSTREAM_N;
        h2_n = NS(pow_int_exp)( h2_n, N - 1 );
        use_taylor_sum = ( h2_n > (
            real_type )SIXTRL_CERRF_CERNLIB_MIN_POW_2H_N );
    }

    nu = ( !use_taylor_sum )
       ? ( int )SIXTRL_CERRF_CERNLIB_K : ( int )SIXTRL_CERRF_CERNLIB_UPSTREAM_NU;

    #if  defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
                ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 )
    if( !use_dawson_approx )
    {
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 ) */

        int n = ( y > ( real_type )SIXTRL_CERRF_CERNLIB_MIN_Y ) ? nu : 0;
        real_type nn = ( real_type )n;
        real_type Rx = ( y > ( real_type )SIXTRL_CERRF_CERNLIB_MIN_Y )
           ? ( real_type )0.0
           : exp( -x * x ) / NS(MathConst_two_over_sqrt_pi)();

        real_type temp, Ry, Sx, Sy, Wx, Wy;
        Ry = Sx = Sy = ( real_type )0.0;

        /* z outside of R_0: continued fraction / Gauss - Hermite quadrature
         * z inside  of R_0: first iterations of recursion until n == N */
        for( ; n > N ; --n, nn -= ( real_type )1.0 )
        {
            Wx     = y_plus_h + nn * Rx;
            Wy     = x - nn * Ry;
            temp   = ( Wx * Wx ) + ( Wy * Wy );
            Rx     = ( real_type )0.5 * Wx;
            Ry     = ( real_type )0.5 * Wy;
            temp   = ( real_type )1.0 / temp;
            Rx    *= temp;
            Ry    *= temp;
        }

        /* loop rejects everything if z is not in R_0 because then n == 0
         * already; otherwise, N iterations until taylor expansion
         * is summed up */
        for( ; n > 0 ; --n, nn -= ( real_type )1.0 )
        {
            Wx     = y_plus_h + nn * Rx;
            Wy     = x - nn * Ry;
            temp   = ( Wx * Wx ) + ( Wy * Wy );
            Rx     = ( real_type )0.5 * Wx;
            Ry     = ( real_type )0.5 * Wy;
            temp   = ( real_type )1.0 / temp;
            Rx    *= temp;
            Ry    *= temp;

            Wx     = h2_n + Sx;
            h2_n  *= inv_h2;
            Sx     = Rx * Wx - Ry * Sy;
            Sy     = Ry * Wx + Rx * Sy;
        }

        *out_x = NS(MathConst_two_over_sqrt_pi)() * (
            ( use_taylor_sum ) ? Sx : Rx );

        *out_y = NS(MathConst_two_over_sqrt_pi)() * (
            ( use_taylor_sum ) ? Sy : Ry );

    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 )
    }
    else
    {
        #if defined( SIXTRL_CERRF_USE_DAWSON_COEFF ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_COEFF == 1 )
        NS(dawson_cerrf_coeff)( x, y, out_x, out_y, xi, Fz_xi, Fz_nt );
        #elif defined( SIXTRL_CERRF_USE_DAWSON_COEFF ) && \
                     ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
        NS(dawson_cerrf_coeff)( x, y, out_x, out_y,
                                xi, Fz_xi, Fz_nt, Fz_kk_xi );
        #else
        NS(dawson_cerrf)( x, y, out_x, out_y );
        #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF ) */
    }
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 ) */
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS(cerrf_alg680_q1)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_x,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_y
    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
        defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_XI_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT xi,
    SIXTRL_CERRF_DAWSON_COEFF_FZ_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT Fz_xi,
    SIXTRL_CERRF_DAWSON_COEFF_NT_DEC SIXTRL_INT32_TYPE const* SIXTRL_RESTRICT Fz_nt
    #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_TAYLOR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT Fz_kk_xi
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 ) */
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T real_type;

    real_type wz_re = ( real_type )0.0;
    real_type wz_im = ( real_type )0.0;

    real_type const xs    = x * ( real_type )SIXTRL_CERRF_ALG680_INV_X0;
    real_type const ys    = y * ( real_type )SIXTRL_CERRF_ALG680_INV_Y0;
    real_type q_rho_squ = ( xs * xs ) + ( ys * ys );

    bool use_power_series = ( q_rho_squ < (
        real_type )SIXTRL_CERRF_ALG680_QRHO_SQU_POWER_SERIES_LIMIT );

    real_type x_quad, y_quad, exp_minus_x_quad, factor_cos, factor_sin;

    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 )
    bool const use_dawson_approx = (
        ( x >= ( real_type )SIXTRL_CERRF_ALG680_USE_DAWSON_APPROX_MIN_X ) &&
        ( x <= ( real_type )SIXTRL_CERRF_ALG680_USE_DAWSON_APPROX_MAX_X ) &&
        ( y <= ( real_type )SIXTRL_CERRF_ALG680_USE_DAWSON_APPROX_MAX_Y ) );

    use_power_series &= !use_dawson_approx;
    #endif /* ( CERRF_USE_DAWSON_FUNCTION == 1 ) */

    SIXTRL_ASSERT( x <= ( real_type )SIXTRL_CERRF_ALG680_REAL_MAX_X );
    SIXTRL_ASSERT( y <= ( real_type )SIXTRL_CERRF_ALG680_REAL_MAX_Y );
    SIXTRL_ASSERT( out_x != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( out_y != SIXTRL_NULLPTR );

    factor_sin = factor_cos = exp_minus_x_quad = ( real_type )1.0;
    x_quad = y_quad = ( real_type )0.0;

    if( use_power_series )
    {
        x_quad = ( x - y ) * ( x + y );
        y_quad = ( real_type )2.0 * x * y;
        exp_minus_x_quad = NS(exp)( -x_quad );
        NS(sincos)( y_quad, &factor_sin, &factor_cos );
    }

    factor_cos *= exp_minus_x_quad;
    factor_sin *= exp_minus_x_quad;

    if( use_power_series )
    {
        real_type temp = ( real_type )0.0;

        real_type const q_rho = NS(sqrt)( q_rho_squ ) * (
            ( real_type )SIXTRL_CERRF_ALG680_QRHO_C0 -
            ( real_type )SIXTRL_CERRF_ALG680_QRHO_C1 * ys );

        int const N = ( int )NS(round)(
                ( real_type )SIXTRL_CERRF_ALG680_N_R0 +
                ( real_type )SIXTRL_CERRF_ALG680_N_R1 * q_rho );

        int ii = N;

        real_type kk = ( real_type )N;
        real_type jj = ( real_type )2.0 * kk + ( real_type )1.0;
        real_type uu = ( real_type )0.0;
        real_type vv = ( real_type )0.0;

        wz_re  = ( real_type )1.0 / jj;

        for( ; ii > 0 ; --ii, kk -= ( real_type )1.0 )
        {
            real_type const c1 = ( wz_re * x_quad ) - ( wz_im * y_quad );
            real_type const c2 = ( wz_re * y_quad ) + ( wz_im * x_quad );
            real_type const inv_kk  = ( real_type )1.0 / kk;

            jj   -= ( real_type )2.0;
            temp  = c1 * inv_kk;
            wz_im = c2 * inv_kk;
            wz_re = temp + ( real_type )1.0 / jj;
        }

        uu  = ( real_type )1.0;
        uu -= NS(MathConst_two_over_sqrt_pi)() * ( wz_re * y + wz_im * x );
        vv  = NS(MathConst_two_over_sqrt_pi)() * ( wz_re * x - wz_im * y );

        wz_re = +uu * factor_cos + vv * factor_sin;
        wz_im = -uu * factor_sin + vv * factor_cos;
    }
    #if defined( CERRF_USE_DAWSON_FUNCTION ) && ( CERRF_USE_DAWSON_FUNCTION == 1 )
    else if( !use_dawson_approx )
    #else /* !CERRF_USE_DAWSON_FUNCTION */
    else
    #endif /* CERRF_USE_DAWSON_FUNCTION */
    {
        bool use_cont_fraction = ( ( real_type )q_rho_squ >= (
            real_type )SIXTRL_CERRF_ALG680_QRHO_SQU_CONT_FRAC_LIMIT );

        real_type const q_rho = ( use_cont_fraction )
            ? NS(sqrt)( q_rho_squ )
            : ( ( real_type )1.0 - ys ) * NS(sqrt)( ( real_type )1.0 - q_rho_squ );

        real_type h = ( real_type )0.0;
        real_type two_h_n = ( real_type )0.0;
        real_type inv_two_h = ( real_type )1.0;

        real_type rx = ( real_type )0.0;
        real_type ry = ( real_type )0.0;
        real_type sx = ( real_type )0.0;
        real_type sy = ( real_type )0.0;
        real_type nn_plus_1;

        int n;
        int N  = -1;
        int nu =  0;

        if( !use_cont_fraction )
        {
            h = ( real_type )SIXTRL_CERRF_ALG680_H1 * q_rho;
            two_h_n = ( real_type )2.0 * h;

            if( two_h_n > ( real_type )SIXTRL_CERRF_ALG680_MIN_TWO_H_VALUE )
            {
                inv_two_h = ( real_type )1.0 / two_h_n;
            }

            N = ( int )NS(round)(
                    ( real_type )SIXTRL_CERRF_ALG680_N_S0 +
                    ( real_type )SIXTRL_CERRF_ALG680_N_S1 * q_rho );

            nu = ( int )NS(round)(
                    ( real_type )SIXTRL_CERRF_ALG680_NU_S0 +
                    ( real_type )SIXTRL_CERRF_ALG680_NU_S1 * q_rho );

            two_h_n = NS(pow_int_exp)( two_h_n, N );

            if( two_h_n <= ( real_type )SIXTRL_CERRF_ALG680_MIN_POW_2H_N )
            {
                use_cont_fraction = true;
                two_h_n   = ( real_type )0.0;
                inv_two_h = ( real_type )1.0;
                N = -1;
            }
        }
        else
        {
            nu = ( int )NS(round)(
                ( real_type )SIXTRL_CERRF_ALG680_K0_CONT_FRACTION +
                ( real_type )SIXTRL_CERRF_ALG680_K1_CONT_FRACTION /
                ( ( real_type )SIXTRL_CERRF_ALG680_K2_CONT_FRACTION +
                  ( real_type )SIXTRL_CERRF_ALG680_K3_CONT_FRACTION * q_rho ) );
        }

        n = nu + 1;
        nn_plus_1 = ( real_type )n;
        --n;

        for( ; n > N ; --n, nn_plus_1 -= ( real_type )1.0 )
        {
            real_type const tx = y + h + nn_plus_1 * rx;
            real_type const ty = x - nn_plus_1 * ry;
            real_type temp = ( tx * tx ) + ( ty * ty );
            rx   = ( real_type )0.5 * tx;
            ry   = ( real_type )0.5 * ty;
            temp = ( real_type )1.0 / temp;
            rx  *= temp;
            ry  *= temp;
        }

        for( ; n >= 0 ; --n, nn_plus_1 -= ( real_type )1.0 )
        {
            real_type const ty = x - nn_plus_1 * ry;
            real_type tx       = y + h + nn_plus_1 * rx;
            real_type temp     = ( tx * tx ) + ( ty * ty );
            rx                 = ( real_type )0.5 * tx;
            ry                 = ( real_type )0.5 * ty;
            temp               = ( real_type )1.0 / temp;
            rx                *= temp;
            ry                *= temp;

            tx                 = two_h_n + sx;
            two_h_n           *= inv_two_h;
            sx                 = ( rx * tx ) - ( ry * sy );
            sy                 = ( ry * tx ) + ( rx * sy );
        }

        if( use_cont_fraction )
        {
            wz_re = NS(MathConst_two_over_sqrt_pi)() * rx;
            wz_im = NS(MathConst_two_over_sqrt_pi)() * ry;
        }
        else
        {
            wz_re = NS(MathConst_two_over_sqrt_pi)() * sx;
            wz_im = NS(MathConst_two_over_sqrt_pi)() * sy;
        }
    }
    #if defined( CERRF_USE_DAWSON_FUNCTION ) && \
               ( CERRF_USE_DAWSON_FUNCTION == 1 )
    else
    {
        SIXTRL_CERRF_RESULT_DEC real_type temp_wz_re;
        SIXTRL_CERRF_RESULT_DEC real_type temp_wz_im;

        #if defined( SIXTRL_CERRF_USE_DAWSON_COEFF ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_COEFF == 1 )
        NS(dawson_cerrf_coeff)( x, y, &temp_wz_re, &temp_wz_im,
                                xi, Fz_xi, Fz_nt );
        #elif defined( SIXTRL_CERRF_USE_DAWSON_COEFF ) && \
                     ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
        NS(dawson_cerrf_coeff)( x, y, &temp_wz_re, &temp_wz_im,
                                xi, Fz_xi, Fz_nt, Fz_kk_xi );
        #else /* ( SIXTRL_CERRF_USE_DAWSON_COEFF == 0 ) */
        NS(dawson_cerrf)( x, y, &temp_wz_re, &temp_wz_im );
        #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF ) */

        wz_re = temp_wz_re;
        wz_im = temp_wz_im;
    }
    #else  /* ( CERRF_USE_DAWSON_FUNCTION != 1 ) */

    if( y < ( real_type )SIXTRL_CERRF_ALG680_MIN_Y )
    {
        real_type const x_squ = x * x;
        wz_re = ( x_squ < ( real_type )SIXTRL_ALG680_MAX_REAL_MAX_EXP )
              ? NS(exp)( -x_squ ) : ( real_type )0.0;
    }
    #endif /* ( CERRF_USE_DAWSON_FUNCTION != 1 ) */

    *out_x = wz_re;
    *out_y = wz_im;
}


SIXTRL_INLINE SIXTRL_REAL_T NS(cerrf_abq2011_a_m_coeff)(
    int const m ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T real_type;
    real_type a_m = ( real_type )0.0;

    SIXTRL_ASSERT( m < ( int )SIXTRL_CERRF_ABQ2011_N_FOURIER );

    #if defined( SIXTRL_CERRF_ABQ2011_N_FOURIER ) &&  \
               ( SIXTRL_CERRF_ABQ2011_N_FOURIER == 24 ) && \
        defined( SIXTRL_CERRF_ABQ2011_TM ) && ( SIXTRL_CERRF_ABQ2011_TM == 12 )

    switch( m )
    {
        case  0: { a_m = ( real_type )0.295408975150919337883027913890;      break; }   /* a_00 */
        case  1: { a_m = ( real_type )0.275840233292177084395258287749;      break; }   /* a_01 */
        case  2: { a_m = ( real_type )0.224573955224615866231619198223;      break; }   /* a_02 */
        case  3: { a_m = ( real_type )0.159414938273911722757388079389;      break; }   /* a_03 */
        case  4: { a_m = ( real_type )0.0986657664154541891084237249422;     break; }   /* a_04 */
        case  5: { a_m = ( real_type )0.0532441407876394120414705837561;     break; }   /* a_05 */
        case  6: { a_m = ( real_type )0.0250521500053936483557475638078;     break; }   /* a_06 */
        case  7: { a_m = ( real_type )0.0102774656705395362477551802420;     break; }   /* a_07 */
        case  8: { a_m = ( real_type )0.00367616433284484706364335443079;    break; }   /* a_08 */
        case  9: { a_m = ( real_type )0.00114649364124223317199757239908;    break; }   /* a_09 */
        case 10: { a_m = ( real_type )0.000311757015046197600406683642851;   break; }   /* a_10 */
        case 11: { a_m = ( real_type )0.0000739143342960301487751427184143;  break; }   /* a_11 */
        case 12: { a_m = ( real_type )0.0000152794934280083634658979605774;  break; }   /* a_12 */
        case 13: { a_m = ( real_type )0.00000275395660822107093261423133381; break; }   /* a_13 */
        case 14: { a_m = ( real_type )4.32785878190124505246159684324E-7;    break; }   /* a_14 */
        case 15: { a_m = ( real_type )5.93003040874588104132914772669E-8;    break; }   /* a_15 */
        case 16: { a_m = ( real_type )7.08449030774820424708618991843E-9;    break; }   /* a_16 */
        case 17: { a_m = ( real_type )7.37952063581678039121116498488E-10;   break; }   /* a_17 */
        case 18: { a_m = ( real_type )6.70217160600200763046136003593E-11;   break; }   /* a_18 */
        case 19: { a_m = ( real_type )5.30726516347079017807414252726E-12;   break; }   /* a_19 */
        case 20: { a_m = ( real_type )3.66432411346763916925386157070E-13;   break; }   /* a_20 */
        case 21: { a_m = ( real_type )2.20589494494103134281934595834E-14;   break; }   /* a_21 */
        case 22: { a_m = ( real_type )1.15782686262855878932236226031E-15;   break; }   /* a_22 */
        case 23: { a_m = ( real_type )5.29871142946730483659082681849E-17;   break; }   /* a_23 */
        default: { a_m = ( real_type )0.0; }
    };

    #endif /* N_FOURIER == 24 && TM = 12 */

    return a_m;
}

#if !defined( SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_ODD_M )
    #define SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_ODD_M( \
        T, m, a_m, exp_cos_tm_x, exp_sin_tm_x, two_over_sqrt_pi, \
        c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  ) \
        \
        c4           =  ( -exp_cos_tm_x ) - ( T )1.;\
        c3           =  ( T )( ( m ) * ( m ) ) * ( T )( two_over_sqrt_pi ) - ( c1 ); \
        temp         =  (   c2_squ    ) + c3 * c3; \
        sn_re        =  (   c3   * c4 ) + ( ( c2 ) * ( exp_sin_tm_x ) );\
        sn_im        =  ( ( c2 ) * c4 ) - (   c3   * ( exp_sin_tm_x ) );\
        sn_re       *=  a_m; \
        sn_im       *=  a_m; \
        temp         = ( T )1.0 / temp; \
        sum_re      += sn_re * temp; \
        sum_im      += sn_im * temp
#endif

#if !defined( SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_EVEN_M )
    #define SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_EVEN_M( \
        T, m, a_m, exp_cos_tm_x, exp_sin_tm_x, two_over_sqrt_pi, \
        c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  ) \
        \
        c4           =  ( exp_cos_tm_x ) - ( T )1.;\
        c3           =  ( T )( ( m ) * ( m ) ) * ( T )( two_over_sqrt_pi ) - ( c1 ); \
        temp         =  (   c2_squ    ) + c3 * c3; \
        sn_re        =  (   c3   * c4 ) - ( ( c2 ) * ( exp_sin_tm_x ) );\
        sn_im        =  ( ( c2 ) * c4 ) + (   c3   * ( exp_sin_tm_x ) );\
        sn_re       *=  a_m; \
        sn_im       *=  a_m; \
        temp         = ( T )1.0 / temp; \
        sum_re      += sn_re * temp; \
        sum_im      += sn_im * temp
#endif

SIXTRL_INLINE void NS(cerrf_abq2011_q1)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_x,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_y
    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
        defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_XI_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT xi,
    SIXTRL_CERRF_DAWSON_COEFF_FZ_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT Fz_xi,
    SIXTRL_CERRF_DAWSON_COEFF_NT_DEC SIXTRL_INT32_TYPE const* SIXTRL_RESTRICT Fz_nt
    #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_TAYLOR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT Fz_kk_xi
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 ) */
) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T real_type;

    real_type temp = ( real_type )0.0;
    real_type const x_squ = x * x;
    real_type const y_squ = y * y;
    bool use_fourier_sum = true;

    #if defined( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION ) && \
               ( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION == 1 )
    bool const use_continued_fraction = ( ( x_squ + y_squ ) >= (
            real_type )SIXTRL_CERRF_ABQ2011_CONT_FRACTION_LIMIT_SQU );
    #endif /* ( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION == 1 ) */

    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 )
    bool const use_dawson_approx = (
        ( x >= ( real_type )SIXTRL_CERRF_ABQ2011_DAWSON_X_MIN ) &&
        ( x <= ( real_type )SIXTRL_CERRF_ABQ2011_DAWSON_X_MAX ) &&
        ( y <= ( real_type )SIXTRL_CERRF_ABQ2011_DAWSON_Y_MAX ) );
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 ) */

    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 )
    use_fourier_sum &= !use_dawson_approx;
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 ) */

    #if defined( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION ) && \
               ( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION == 1 )
    use_fourier_sum &= !use_continued_fraction;
    #endif /* ( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION == 1 ) */

    SIXTRL_ASSERT( out_x != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( out_y != SIXTRL_NULLPTR );

    if( use_fourier_sum )
    {
        SIXTRL_RESULT_PTR_DEC real_type exp_cos_tm_x, exp_sin_tm_x;

        real_type sum_re       = ( real_type )0.0;
        real_type sum_im       = ( real_type )0.0;

        real_type sn_re        = ( real_type )0.0;
        real_type sn_im        = ( real_type )0.0;

        real_type c3           = ( real_type )0.0;
        real_type c4           = ( real_type )1.0;

        real_type wz_re        = ( real_type )0.0;
        real_type wz_im        = ( real_type )0.0;

        real_type const c1     = ( real_type )SIXTRL_CERRF_ABQ2011_TM_SQU *
                                 ( x + y ) * ( x - y );

        real_type const c2     = ( real_type )2.0 *
                                 ( real_type )SIXTRL_CERRF_ABQ2011_TM_SQU * x * y;

        real_type const c2_squ = c2 * c2;
        real_type a_m;

        #if !defined( SIXTRL_CERRF_ABQ2011_FORCE_UNROLLED_LOOP ) || \
                    ( SIXTRL_CERRF_ABQ2011_FORCE_UNROLLED_LOOP != 1 )
        int m = 1;
        #endif /* ( SIXTRL_CERRF_ABQ2011_FORCE_UNROLLED_LOOP != 1 ) */

        temp = ( real_type )SIXTRL_CERRF_ABQ2011_TM * x;
        NS(sincos)( temp, &exp_sin_tm_x, &exp_cos_tm_x );

        temp = NS(exp)( -( ( real_type )SIXTRL_CERRF_ABQ2011_TM * y ) );
        exp_sin_tm_x *= temp;
        exp_cos_tm_x *= temp;

        c4    -= exp_cos_tm_x;

        /* Contribution for m = 0 */
        temp   = x_squ + y_squ;
        temp  *= ( real_type )SIXTRL_CERRF_ABQ2011_TM;
        wz_re  = ( y * c4 ) + ( x * exp_sin_tm_x );
        wz_im  = ( x * c4 ) - ( y * exp_sin_tm_x );
        temp   = ( real_type )1. / temp;
        wz_re *= temp;
        wz_im *= temp;

        #if defined( SIXTRL_CERRF_ABQ2011_FORCE_UNROLLED_LOOP ) && \
                   ( SIXTRL_CERRF_ABQ2011_FORCE_UNROLLED_LOOP == 1 )

        /* Manually unrolled loop */
        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 1 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A01 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A01;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_ODD_M( real_type, 1, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 1 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 2 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A02 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A02;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_EVEN_M( real_type, 2, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 2 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 3 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A03 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A03;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_ODD_M( real_type, 3, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 3 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 4 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A04 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A04;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_EVEN_M( real_type, 4, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 4 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 5 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A05 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A05;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_ODD_M( real_type, 5, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 5 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 6 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A06 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A06;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_EVEN_M( real_type, 6, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 6 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 7 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A07 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A07;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_ODD_M( real_type, 7, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );

        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 7 */
        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 8 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A08 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A08;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_EVEN_M( real_type, 8, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 8 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 9 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A09 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A09;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_ODD_M( real_type, 9, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 9 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 10 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A10 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A10;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_EVEN_M( real_type, 10, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 10 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 11 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A11 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A11;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_ODD_M( real_type, 11, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 11 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 12 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A12 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A12;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_EVEN_M( real_type, 12, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 12 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 13 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A13 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A13;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_ODD_M( real_type, 13, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 13 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 14 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A14 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A14;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_EVEN_M( real_type, 14, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 14 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 15 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A15 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A15;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_ODD_M( real_type, 15, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 15 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 16 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A16 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A16;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_EVEN_M( real_type, 16, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 16 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 17 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A17 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A17;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_ODD_M( real_type, 17, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 17 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 18 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A18 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A18;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_EVEN_M( real_type, 18, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 18 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 19 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A19 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A19;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_ODD_M( real_type, 19, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 19 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 20 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A20 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A20;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_EVEN_M( real_type, 20, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 20 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 21 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A21 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A21;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_ODD_M( real_type, 21, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 221 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 22 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A22 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A22;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_EVEN_M( real_type, 22, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 22 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 23 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A23 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A23;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_ODD_M( real_type, 23, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 23 */

        #if ( SIXTRL_CERRF_ABQ2011_N_FOURIER > 24 ) && \
              defined( SIXTRL_CERRF_ABQ2011_A24 )
        a_m = ( real_type )SIXTRL_CERRF_ABQ2011_A24;
        SIXTRACKLIB_CERRF_ABQ2011_FOURIER_SUM_EVEN_M( real_type, 24, a_m,
            exp_cos_tm_x, exp_sin_tm_x, NS(MathConst_two_over_sqrt_pi(),
            c1, c2, c2_squ, c3, c4, temp, sn_re, sn_im, sum_re, sum_im  );
        #endif /* SIXTRL_CERRF_ABQ2011_N_FOURIER > 24 */

        #else /* ( SIXTRL_CERRF_ABQ2011_FORCE_UNROLLED_LOOP != 1 ) */

        for( ; m < ( int )SIXTRL_CERRF_ABQ2011_N_FOURIER ; ++m )
        {
            exp_cos_tm_x = -exp_cos_tm_x;
            exp_sin_tm_x = -exp_sin_tm_x;
            c4 = exp_cos_tm_x - ( real_type )1.;
            c3 = ( real_type )( m * m ) * NS(MathConst_two_over_sqrt_pi)() - c1;

            temp    =  c2_squ + c3 * c3;
            sn_re   =  ( c3 * c4 ) - ( c2 * exp_sin_tm_x );
            sn_im   =  ( c2 * c4 ) + ( c3 * exp_sin_tm_x );
            a_m     =  NS(cerrf_abq2011_a_m_coeff)( m );
            sn_re  *=  a_m;
            sn_im  *=  a_m;
            temp    = ( real_type )1.0 / temp;
            sum_re += sn_re * temp;
            sum_im += sn_im * temp;
        }

        #endif /* ( SIXTRL_CERRF_ABQ2011_FORCE_UNROLLED_LOOP ) */
        /* normalize the sum + apply common pre-factor i * z */

        temp    = (  x * sum_im ) + ( y * sum_re );
        sum_im  = (  x * sum_re ) - ( y * sum_im );

        wz_re  -= temp   * ( real_type )SIXTRL_CERRF_ABQ2011_TM_SQU_OVER_SQRT_PI;
        wz_im  += sum_im * ( real_type )SIXTRL_CERRF_ABQ2011_TM_SQU_OVER_SQRT_PI;

        *out_x  = wz_re;
        *out_y  = wz_im;
    }

    #if defined( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION ) && \
               ( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION == 1 )
    else if( use_continued_fraction )
    {
        real_type wz_re, wz_im;
        real_type rx = ( real_type )0.0;
        real_type ry = ( real_type )0.0;
        real_type nn = ( real_type )SIXTRL_CERRF_ABQ2011_CONT_FRACTION_K;

        for( ; nn > ( real_type )0. ; nn -= ( real_type )1. )
        {
            wz_re = y + nn * rx;
            wz_im = x - nn * ry;
            temp  = ( wz_re * wz_re + wz_im * wz_im );

            rx    = ( real_type )0.5 * wz_re;
            ry    = ( real_type )0.5 * wz_im;
            temp  = ( real_type )1.0 / temp;

            rx   *= temp;
            ry   *= temp;
        }

        *out_x = NS(MathConst_two_over_sqrt_pi)() * rx;
        *out_y = NS(MathConst_two_over_sqrt_pi)() * ry;
    }

    #endif /* ( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION == 1 ) */

    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 )
    else if( use_dawson_approx )
    {
        #if defined( SIXTRL_CERRF_USE_DAWSON_COEFF ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_COEFF == 1 )
        NS(dawson_cerrf_coeff)( x, y, out_x, out_y, xi, Fz_xi, Fz_nt );
        #elif defined( SIXTRL_CERRF_USE_DAWSON_COEFF ) && \
                     ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
        NS(dawson_cerrf_coeff)( x, y, out_x, out_y, xi, Fz_xi, Fz_nt, Fz_kk_xi );
        #else /* ( SIXTRL_CERRF_USE_DAWSON_COEFF == 1 ) */
        NS(dawson_cerrf)( x, y, out_x, out_y );
        #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF ) */
    }

    #endif /* ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 ) */
}


SIXTRL_INLINE void NS(cerrf_abq2011_q1_coeff)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_x,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_y,
    SIXTRL_CERRF_ABQ2011_FOURIER_COEFF_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT a_m,
    SIXTRL_CERRF_ABQ2011_TAYLOR_COEFF_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT b_n
    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
        defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_XI_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT xi,
    SIXTRL_CERRF_DAWSON_COEFF_FZ_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT Fz_xi,
    SIXTRL_CERRF_DAWSON_COEFF_NT_DEC SIXTRL_INT32_TYPE const* SIXTRL_RESTRICT Fz_nt
    #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_TAYLOR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT Fz_kk_xi
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
    #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX &&  SIXTRL_CERRF_USE_DAWSON_COEFF */
) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T real_type;

    real_type temp = ( real_type )0.0;
    real_type const x_squ = x * x;
    real_type const y_squ = y * y;
    bool use_fourier_sum = true;

    real_type wz_re = ( real_type )0.0;
    real_type wz_im = ( real_type )0.0;

    #if defined( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION ) && \
               ( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION == 1 )
    bool const use_continued_fraction = ( ( x_squ + y_squ ) >= (
            real_type )SIXTRL_CERRF_ABQ2011_CONT_FRACTION_LIMIT_SQU );
    #endif /* ( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION == 1 ) */

    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 )
    bool const use_dawson_approx = (
        ( x >= ( real_type )SIXTRL_CERRF_ABQ2011_DAWSON_X_MIN ) &&
        ( x <= ( real_type )SIXTRL_CERRF_ABQ2011_DAWSON_X_MAX ) &&
        ( y <= ( real_type )SIXTRL_CERRF_ABQ2011_DAWSON_Y_MAX ) );

    #endif /* ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 ) */

    #if defined( SIXTRL_CERRF_ABQ2011_USE_TAYLOR_POLE_APPROX ) && \
               ( SIXTRL_CERRF_ABQ2011_USE_TAYLOR_POLE_APPROX == 1 )
    int N_POLE = ( int )-1;
    bool use_pole_taylor_approx = (
        #if  !defined( SIXTRL_CERRF_ABQ2011_N_TAYLOR ) || \
                     ( SIXTRL_CERRF_ABQ2011_N_TAYLOR < 1 )
        ( false ) &&
        #endif /* ( !SIXTRL_CERRF_ABQ2011_N_TAYLOR  ) */

        #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 )
        ( !use_dawson_approx ) &&
        #endif /* ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 ) */

        #if defined( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION ) && \
                   ( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION == 1 )
        ( !use_continued_fraction ) &&
        #endif /* ( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION == 1 ) */

        ( b_n != SIXTRL_NULLPTR ) &&
        ( y < ( ( real_type )SIXTRL_CERRF_ABQ2011_MIN_POLE_DIST ) ) &&
        ( x < ( ( real_type )SIXTRL_CERRF_ABQ2011_N_FOURIER  *
                ( real_type )SIXTRL_CERRF_ABQ2011_PI_OVER_TM +
                ( real_type )SIXTRL_CERRF_ABQ2011_MIN_POLE_DIST ) ) );

    if( use_pole_taylor_approx )
    {
        real_type d_pole_squ = y_squ;
        N_POLE = ( int )NS(round)( x * ( real_type )SIXTRL_CERRF_ABQ2011_TM_OVER_PI );
        temp   = x - ( ( real_type )SIXTRL_CERRF_ABQ2011_PI_OVER_TM *
                       ( real_type )N_POLE );

        d_pole_squ += temp * temp;
        if( d_pole_squ >= ( real_type )SIXTRL_CERRF_ABQ2011_MIN_POLE_DIST_SQU )
        {
            use_pole_taylor_approx = false;
            N_POLE = -1;
        }

        SIXTRL_ASSERT( ( N_POLE == ( int )-1 ) || ( N_POLE >= ( int )0u ) );
        SIXTRL_ASSERT(   N_POLE <= ( int )SIXTRL_CERRF_ABQ2011_N_FOURIER );
    }
    #else /* ( SIXTRL_CERRF_ABQ2011_USE_TAYLOR_POLE_APPROX != 1 ) */
    ( void )b_n;

    #endif /* ( SIXTRL_CERRF_ABQ2011_USE_TAYLOR_POLE_APPROX ) */

    #if defined( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION ) && \
               ( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION == 1 )
    use_fourier_sum &= !use_continued_fraction;
    #endif /* ( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION == 1 ) */

    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 )
    use_fourier_sum &= !use_dawson_approx;
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 ) */

    #if defined( SIXTRL_CERRF_ABQ2011_USE_TAYLOR_POLE_APPROX ) && \
               ( SIXTRL_CERRF_ABQ2011_USE_TAYLOR_POLE_APPROX == 1 )
    use_fourier_sum &= !use_pole_taylor_approx;
    #endif /* ( SIXTRL_CERRF_ABQ2011_USE_TAYLOR_POLE_APPROX == 1 ) */

    SIXTRL_ASSERT( out_x != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( out_y != SIXTRL_NULLPTR );

    if( use_fourier_sum )
    {
        SIXTRL_RESULT_PTR_DEC real_type exp_cos_tm_x, exp_sin_tm_x;

        real_type sum_re       = ( real_type )0.0;
        real_type sum_im       = ( real_type )0.0;

        real_type sn_re        = ( real_type )0.0;
        real_type sn_im        = ( real_type )0.0;

        real_type c3           = ( real_type )0.0;
        real_type c4           = ( real_type )1.0;

        real_type const c1     = ( real_type )SIXTRL_CERRF_ABQ2011_TM_SQU *
                                 ( x + y ) * ( x - y );

        real_type const c2     = ( real_type )2.0 *
                                 ( real_type )SIXTRL_CERRF_ABQ2011_TM_SQU * x * y;

        real_type const c2_squ = c2 * c2;
        real_type a_m_value;
        int m = ( int )1u;

        temp = ( real_type )SIXTRL_CERRF_ABQ2011_TM * x;
        NS(sincos)( temp, &exp_sin_tm_x, &exp_cos_tm_x );

        temp = NS(exp)( -( ( real_type )SIXTRL_CERRF_ABQ2011_TM * y ) );
        exp_sin_tm_x *= temp;
        exp_cos_tm_x *= temp;

        c4 -= exp_cos_tm_x;

        /* Contribution for m = 0 */
        temp   = x_squ + y_squ;
        temp  *= ( real_type )SIXTRL_CERRF_ABQ2011_TM;
        wz_re  = ( y * c4 ) + ( x * exp_sin_tm_x );
        wz_im  = ( x * c4 ) - ( y * exp_sin_tm_x );
        temp   = ( real_type )1. / temp;
        wz_re *= temp;
        wz_im *= temp;

        for( ; m < ( int )SIXTRL_CERRF_ABQ2011_N_FOURIER ; ++m )
        {
            exp_cos_tm_x = -exp_cos_tm_x;
            exp_sin_tm_x = -exp_sin_tm_x;
            c4 = exp_cos_tm_x - ( real_type )1.;
            c3 = ( real_type )( m * m ) * NS(MathConst_two_over_sqrt_pi)() - c1;

            temp         =  c2_squ + c3 * c3;
            sn_re        =  ( c3 * c4 ) - ( c2 * exp_sin_tm_x );
            sn_im        =  ( c2 * c4 ) + ( c3 * exp_sin_tm_x );
            a_m_value    =  a_m[ m ];
            sn_re       *=  a_m_value;
            sn_im       *=  a_m_value;
            temp         = ( real_type )1.0 / temp;
            sum_re      += sn_re * temp;
            sum_im      += sn_im * temp;
        }

        /* normalize the sum + apply common pre-factor i * z */

        temp    = (  x * sum_im ) + ( y * sum_re );
        sum_im  = (  x * sum_re ) - ( y * sum_im );

        wz_re  -= temp   * ( real_type )SIXTRL_CERRF_ABQ2011_TM_SQU_OVER_SQRT_PI;
        wz_im  += sum_im * ( real_type )SIXTRL_CERRF_ABQ2011_TM_SQU_OVER_SQRT_PI;

        *out_x = wz_re;
        *out_y = wz_im;
    }
    #if defined( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION ) && \
               ( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION == 1 )
    else if( use_continued_fraction )
    {
        real_type rx = ( real_type )0.0;
        real_type ry = ( real_type )0.0;
        real_type nn = ( real_type )SIXTRL_CERRF_ABQ2011_CONT_FRACTION_K;

        for( ; nn > ( real_type )0. ; nn -= ( real_type )1. )
        {
            wz_re = y + nn * rx;
            wz_im = x - nn * ry;
            temp  = ( wz_re * wz_re + wz_im * wz_im );

            rx    = ( real_type )0.5 * wz_re;
            ry    = ( real_type )0.5 * wz_im;
            temp  = ( real_type )1.0 / temp;

            rx   *= temp;
            ry   *= temp;
        }

        *out_x = NS(MathConst_two_over_sqrt_pi)() * rx;
        *out_y = NS(MathConst_two_over_sqrt_pi)() * ry;
    }

    #endif /* ( SIXTRL_CERRF_ABQ2011_USE_CONT_FRACTION == 1 ) */
    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 )
    else if( use_dawson_approx )
    {
        #if defined( SIXTRL_CERRF_USE_DAWSON_COEFF ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_COEFF == 1 )
        NS(dawson_cerrf_coeff)( x, y, out_x, out_y, xi, Fz_xi, Fz_nt );
        #elif defined( SIXTRL_CERRF_USE_DAWSON_COEFF ) && \
                     ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
        NS(dawson_cerrf_coeff)( x, y, out_x, out_y, xi, Fz_xi, Fz_nt, Fz_kk_xi );
        #else
        NS(dawson_cerrf)( x, y, out_x, out_y );
        #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF ) */
    }

    #endif /* ( SIXTRL_CERRF_USE_DAWSON_APPROX == 1 ) */
    #if defined( SIXTRL_CERRF_ABQ2011_USE_TAYLOR_POLE_APPROX ) && \
               ( SIXTRL_CERRF_ABQ2011_USE_TAYLOR_POLE_APPROX == 1 )
    else if( use_pole_taylor_approx )
    {
        real_type dz_nn_im    = y;
        real_type const dz_re = x - ( (
            real_type )SIXTRL_CERRF_ABQ2011_PI_OVER_TM * ( real_type )N_POLE );
        real_type dz_nn_re    = dz_re;
        real_type b_n_value;

        int ii = 2;
        int jj = 2 * N_POLE * ( int )SIXTRL_CERRF_ABQ2011_N_TAYLOR;

        CERRF_ASSERT( N_POLE >= 0 );
        CERRF_ASSERT( N_POLE <= ( int )SIXTRL_CERRF_ABQ2011_N_FOURIER );

        /* wz = Re(b0) + i * Im(b0) */
        wz_re = b_n[ jj++ ];
        wz_im = b_n[ jj++ ];

        /* wz += b1 * ( dz_re + i * in_y ) */
        b_n_value = b_n[ jj++ ];
        wz_re += b_n_value * dz_nn_re;
        wz_im += b_n_value * dz_nn_im;

        b_n_value = b_n[ jj++ ];
        wz_re -= b_n_value * dz_nn_im;
        wz_im += b_n_value * dz_nn_re;

        for( ; ii < ( int )SIXTRL_CERRF_ABQ2011_N_TAYLOR ; ++ii )
        {
            temp      = dz_nn_re * dz_re - dz_nn_im * in_y;
            dz_nn_im *= dz_re;
            dz_nn_im += dz_nn_re * in_y;
            dz_nn_re  = temp;

            b_n_value = b_n[ jj++ ];
            wz_re    += b_n_value * dz_nn_re;
            wz_im    += b_n_value * dz_nn_im;

            b_n_value = b_n[ jj++ ];
            wz_re    -= b_n_value * dz_nn_im;
            wz_im    += b_n_value * dz_nn_re;
        }

        *out_x = wz_re;
        *out_y = wz_im;
    }

    #endif /* ( SIXTRL_CERRF_ABQ2011_USE_TAYLOR_POLE_APPROX == 1 ) */
}


SIXTRL_INLINE void NS(cerrf_q1)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_x,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_y
    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
        defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_XI_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT xi,
    SIXTRL_CERRF_DAWSON_COEFF_FZ_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT Fz_xi,
    SIXTRL_CERRF_DAWSON_COEFF_NT_DEC SIXTRL_INT32_TYPE const* SIXTRL_RESTRICT Fz_nt
    #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_TAYLOR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT Fz_kk_xi
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
    #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX &&  SIXTRL_CERRF_USE_DAWSON_COEFF */
) SIXTRL_NOEXCEPT
{
    #if ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_CERNLIB_BASELINE ) || \
        ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_CERNLIB_UPSTREAM )

        #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
            defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
            ( void )xi;
            ( void )Fz_xi;
            ( void )Fz_nt;
            #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
            ( void )Fz_kk_xi;
            #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
        #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX && SIXTRL_CERRF_USE_DAWSON_COEFF */
    #endif /* no Dawson's approximaion! */

    #if ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_CERNLIB_BASELINE )
    NS(cerrf_cernlib_c_baseline_q1)( x, y, out_x, out_y );
    #elif ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_CERNLIB_UPSTREAM )
    NS(cerrf_cernlib_c_upstream_q1)( x, y, out_x, out_y );
    #elif ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_CERNLIB_FIXED )
        #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) \
            defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
            NS(cerrf_cernlib_c_optimised_fixed_q1)(
                x, y, out_x, out_y, xi, Fz_xi, Fz_nt );
            #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
            NS(cerrf_cernlib_c_optimised_fixed_q1)(
                x, y, out_x, out_y, xi, Fz_xi, Fz_nt, Fz_kk_xi );
            #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
        #else /* !SIXTRL_CERRF_USE_DAWSON_COEFF */
            NS(cerrf_cernlib_c_optimised_fixed_q1)( x, y, out_x, out_y );
        #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX && SIXTRL_CERRF_USE_DAWSON_COEFF */

    #elif SIXTRL_CERRF_METHOD == SIXTRL_CERRF_ALG680
        #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) \
            defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
            NS(cerrf_alg680_q1)( x, y, out_x, out_y, xi, Fz_xi, Fz_nt );
            #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
            NS(cerrf_alg680_q1)( x, y, out_x, out_y,
                                 xi, Fz_xi, Fz_nt, Fz_kk_xi );
            #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
        #else /* !SIXTRL_CERRF_USE_DAWSON_COEFF */
            NS(cerrf_alg680_q1)( x, y, out_x, out_y );
        #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX && SIXTRL_CERRF_USE_DAWSON_COEFF */

    #elif SIXTRL_CERRF_METHOD == SIXTRL_CERRF_ABQ2011
        #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
            defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
            NS(cerrf_abq2011_q1)( x, y, out_x, out_y, xi, Fz_xi, Fz_nt );
            #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
            NS(cerrf_abq2011_q1)( x, y, out_x, out_y,
                                  xi, Fz_xi, Fz_nt, Fz_kk_xi );
            #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
        #else  /* !SIXTRL_CERRF_USE_DAWSON_COEFF */
            NS(cerrf_abq2011_q1)( x, y, out_x, out_y );
        #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX && SIXTRL_CERRF_USE_DAWSON_COEFF */

    #else  /* ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_CERNLIB_OPTIMISED ) */
        #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
            defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
            NS(cerrf_cernlib_c_optimised_q1)(
                    x, y, out_x, out_y, xi, Fz_xi, Fz_nt );

            #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
            NS(cerrf_cernlib_c_optimised_q1)( x, y, out_x, out_y,
                    xi, Fz_xi, Fz_nt, Fz_kk_xi );

            #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
        #else  /* !SIXTRL_CERRF_USE_DAWSON_COEFF */
            NS(cerrf_cernlib_c_optimised_q1)( x, y, out_x, out_y );
        #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX && SIXTRL_CERRF_USE_DAWSON_COEFF */

    #endif /* SIXTRL_CERRF_METHOD */
}


/** \fn void cerrf( double x, double y, double* out_x, double* out_y )
 *  \brief calculates the Faddeeva function w(z) for general z = x + i * y
 *
 *   Calls the correct cerrf_*_q1 function according to SIXTRL_CERRF_METHOD
 *   internally for |x| and |y| on quadrant Q1 and transforms the result to
 *   Q2, Q3, and Q4 before returning them via out_x and out_y.
 *
 *  \param[in] x real component of argument z
 *  \param[in] y imaginary component of argument z
 *  \param[out] out_x pointer to real component of result
 *  \param[out] out_y pointer to imanginary component of result
 *
 */

SIXTRL_INLINE void NS(cerrf)(
    SIXTRL_REAL_T x, SIXTRL_REAL_T y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_x,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_y
    #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
               ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
        defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
               ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_XI_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT xi,
    SIXTRL_CERRF_DAWSON_COEFF_FZ_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT Fz_xi,
    SIXTRL_CERRF_DAWSON_COEFF_NT_DEC SIXTRL_INT32_TYPE const* SIXTRL_RESTRICT Fz_nt
    #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_TAYLOR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT Fz_kk_xi
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
    #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX &&  SIXTRL_CERRF_USE_DAWSON_COEFF */
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

    #if ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_CERNLIB_BASELINE ) || \
        ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_CERNLIB_UPSTREAM )

        #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
            defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
            ( void )xi;
            ( void )Fz_xi;
            ( void )Fz_nt;
            #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
            ( void )Fz_kk_xi;
            #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
        #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX && SIXTRL_CERRF_USE_DAWSON_COEFF */
    #endif /* no Dawson's approximaion! */

    #if ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_CERNLIB_BASELINE )
    NS(cerrf_cernlib_c_baseline_q1)( x, y, &Wx, &Wy );
    #elif ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_CERNLIB_UPSTREAM )
    NS(cerrf_cernlib_c_upstream_q1)( x, y, &Wx, &Wy );
    #elif ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_CERNLIB_FIXED )
        #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
            defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
            NS(cerrf_cernlib_c_optimised_fixed_q1)(
                x, y, &Wx, &Wy, xi, Fz_xi, Fz_nt );
            #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
            NS(cerrf_cernlib_c_optimised_fixed_q1)(
                x, y, &Wx, &Wy, xi, Fz_xi, Fz_nt, Fz_kk_xi );
            #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
        #else /* !SIXTRL_CERRF_USE_DAWSON_COEFF */
            NS(cerrf_cernlib_c_optimised_fixed_q1)( x, y, &Wx, &Wy );
        #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX && SIXTRL_CERRF_USE_DAWSON_COEFF */


    #elif ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_ALG680 )
        #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
            defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
            NS(cerrf_alg680_q1)( x, y, &Wx, &Wy, xi, Fz_xi, Fz_nt );
            #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
            NS(cerrf_alg680_q1)( x, y, &Wx, &Wy, xi, Fz_xi, Fz_nt, Fz_kk_xi );
            #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
        #else /* !SIXTRL_CERRF_USE_DAWSON_COEFF */
            NS(cerrf_alg680_q1)( x, y, &Wx, &Wy );
        #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX && SIXTRL_CERRF_USE_DAWSON_COEFF */

    #elif ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_ABQ2011 )
        #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
            defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
            NS(cerrf_abq2011_q1)( x, y, &Wx, &Wy, xi, Fz_xi, Fz_nt );
            #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
            NS(cerrf_abq2011_q1)( x, y, &Wx, &Wy, xi, Fz_xi, Fz_nt, Fz_kk_xi );
            #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
        #else  /* !SIXTRL_CERRF_USE_DAWSON_COEFF */
            NS(cerrf_abq2011_q1)( x, y, &Wx, &Wy );
        #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX && SIXTRL_CERRF_USE_DAWSON_COEFF */

    #else  /* ( SIXTRL_CERRF_METHOD == SIXTRL_CERRF_CERNLIB_OPTIMISED ) */
        #if defined( SIXTRL_CERRF_USE_DAWSON_APPROX ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_APPROX  == 1 ) && \
            defined( SIXTRL_CERRF_USE_DAWSON_COEFF  ) && \
                   ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )
            NS(cerrf_cernlib_c_optimised_q1)(
                    x, y, &Wx, &Wy, xi, Fz_xi, Fz_nt );

            #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
            NS(cerrf_cernlib_c_optimised_q1)( x, y, &Wx, &Wy,
                    xi, Fz_xi, Fz_nt, Fz_kk_xi );

            #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 ) */
        #else  /* !SIXTRL_CERRF_USE_DAWSON_COEFF */
            NS(cerrf_cernlib_c_optimised_q1)( x, y, &Wx, &Wy );
        #endif /* SIXTRL_CERRF_USE_DAWSON_APPROX && SIXTRL_CERRF_USE_DAWSON_COEFF */

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

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* !defined( __cplusplus ) && !defined( _GPUCODE ) */
#endif /* SIXTACKLIB_COMMON_BE_BEAMFIELDS_FADDEEVA_HEADER_H__ */
