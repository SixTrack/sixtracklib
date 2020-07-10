#ifndef SIXTRACKLIB_COMMON_INTERNAL_QGAUSS_H__
#define SIXTRACKLIB_COMMON_INTERNAL_QGAUSS_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#if !defined( SIXTRL_MATH_QGAUSSIAN_Q_EPS )
    #define SIXTRL_MATH_QGAUSSIAN_Q_EPS 1e-6L
#endif /* SIXTRL_MATH_QGAUSSIAN_Q_EPS */

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_q_gauss_min_support)(
    SIXTRL_REAL_T const q, SIXTRL_REAL_T const sqrt_beta ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_q_gauss_max_support)(
    SIXTRL_REAL_T const q, SIXTRL_REAL_T const sqrt_beta ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_q_gauss_cq)(
    SIXTRL_REAL_T const q ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(Math_q_gauss_sqrt_beta_from_gauss_sigma)(
    SIXTRL_REAL_T const sigma ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_q_gauss_exp_q)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const q ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_q_gauss)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const q, SIXTRL_REAL_T const sqrt_beta,
    SIXTRL_REAL_T const cq ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_q_gauss_shifted)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const q, SIXTRL_REAL_T const sqrt_beta,
    SIXTRL_REAL_T const cq, SIXTRL_REAL_T const mu ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_REAL_T NS(Math_q_gauss_min_support_ext)(
    SIXTRL_REAL_T const q, SIXTRL_REAL_T const sqrt_beta ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_REAL_T NS(Math_q_gauss_max_support_ext)(
    SIXTRL_REAL_T const q, SIXTRL_REAL_T const sqrt_beta ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_REAL_T NS(Math_q_gauss_cq_ext)(
    SIXTRL_REAL_T const q ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_REAL_T
NS(Math_q_gauss_sqrt_beta_from_gauss_sigma_ext)(
    SIXTRL_REAL_T const sigma ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_REAL_T NS(Math_q_gauss_exp_q_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const q ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_REAL_T NS(Math_q_gauss_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const q, SIXTRL_REAL_T const sqrt_beta,
    SIXTRL_REAL_T const cq ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_REAL_T NS(Math_q_gauss_shifted_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const q, SIXTRL_REAL_T const sqrt_beta,
    SIXTRL_REAL_T const cq, SIXTRL_REAL_T const mu ) SIXTRL_NOEXCEPT;

#endif /* Host */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
    #include <float.h>
#endif /* #if !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/internal/math_constants.h"
    #include "sixtracklib/common/internal/math_functions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_q_gauss_min_support)(
    SIXTRL_REAL_T const q, SIXTRL_REAL_T const sqrt_beta ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( q < ( SIXTRL_REAL_T )3 );
    SIXTRL_ASSERT( sqrt_beta > ( SIXTRL_REAL_T )0 );
    return ( q >= ( SIXTRL_REAL_T )1 )
        ? ( SIXTRL_REAL_T )-1e10
        : ( SIXTRL_REAL_T )-1 / NS(sqrt)( sqrt_beta * sqrt_beta *
            ( ( SIXTRL_REAL_T )1 - q ) );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_q_gauss_max_support)(
    SIXTRL_REAL_T const q, SIXTRL_REAL_T const sqrt_beta ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( q < ( SIXTRL_REAL_T )3 );
    SIXTRL_ASSERT( sqrt_beta > ( SIXTRL_REAL_T )0 );
    return ( q >= ( SIXTRL_REAL_T )1 )
        ? ( SIXTRL_REAL_T )1e10
        : ( SIXTRL_REAL_T )1 / NS(sqrt)( sqrt_beta * sqrt_beta *
            ( ( SIXTRL_REAL_T )1 - q ) );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_q_gauss_exp_q)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const q ) SIXTRL_NOEXCEPT
{
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE = ( SIXTRL_REAL_T )1;
    SIXTRL_REAL_T const one_minus_q = ONE - q;
    SIXTRL_REAL_T const abs_one_minus_q = ( one_minus_q >= ( SIXTRL_REAL_T )0 )
					? one_minus_q : -one_minus_q;

    SIXTRL_ASSERT( q < ( SIXTRL_REAL_T )3 );

    if( abs_one_minus_q >= ( SIXTRL_REAL_T )SIXTRL_MATH_QGAUSSIAN_Q_EPS )
    {
        SIXTRL_REAL_T u_plus = ONE + one_minus_q * x;
        if( u_plus < ( SIXTRL_REAL_T )0 ) u_plus = ( SIXTRL_REAL_T )0;
        return NS(pow_positive_base)( u_plus,  ONE / one_minus_q );
    }
    else
    {
        return NS(exp)( x );
    }
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_q_gauss_cq)(
    SIXTRL_REAL_T const q ) SIXTRL_NOEXCEPT
{
    SIXTRL_REAL_T cq = NS(MathConst_sqrt_pi)();
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE = ( SIXTRL_REAL_T )1;

    SIXTRL_ASSERT( q < ( SIXTRL_REAL_T )3 );
    if( q >= ( ONE + ( SIXTRL_REAL_T )SIXTRL_MATH_QGAUSSIAN_Q_EPS ) )
    {
        SIXTRL_REAL_T const q_minus_1 = q - ONE;
        cq *= NS(gamma)( ( ( SIXTRL_REAL_T )3 - q ) /
                         ( ( SIXTRL_REAL_T )2 * q_minus_1 ) );
        cq /= NS(sqrt)( q_minus_1 ) * NS(gamma)( ONE / q_minus_1 );
    }
    else if( q <= ( ONE - ( SIXTRL_REAL_T )SIXTRL_MATH_QGAUSSIAN_Q_EPS ) )
    {
        SIXTRL_REAL_T const one_minus_q = ONE - q;
        SIXTRL_REAL_T const three_minus_q = ( SIXTRL_REAL_T )3 - q;
        cq *= ( SIXTRL_REAL_T )2 * NS(gamma)( ONE / one_minus_q );
        cq /= three_minus_q * NS(sqrt)( one_minus_q ) *
              NS(gamma)( three_minus_q / ( ( SIXTRL_REAL_T )2 * one_minus_q ) );
    }

    return cq;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_q_gauss_sqrt_beta_from_gauss_sigma)(
    SIXTRL_REAL_T const sigma ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sigma > ( SIXTRL_REAL_T )0 );
    return ( SIXTRL_REAL_T )1 / ( NS(sqrt)( ( SIXTRL_REAL_T )2 ) * sigma );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_q_gauss)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const q, SIXTRL_REAL_T const sqrt_beta,
    SIXTRL_REAL_T const cq ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( q  < ( SIXTRL_REAL_T )3 );
    SIXTRL_ASSERT( cq > ( SIXTRL_REAL_T )0 );
    SIXTRL_ASSERT( sqrt_beta > ( SIXTRL_REAL_T )0 );

    return ( sqrt_beta / cq ) * NS(Math_q_gauss_exp_q)(
        -( sqrt_beta * sqrt_beta * x * x ), q );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_q_gauss_shifted)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const q, SIXTRL_REAL_T const sqrt_beta,
    SIXTRL_REAL_T const cq, SIXTRL_REAL_T const mu ) SIXTRL_NOEXCEPT
{
    return NS(Math_q_gauss)( x - mu, q, sqrt_beta, cq );
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */
#endif /* SIXTRACKLIB_COMMON_INTERNAL_QGAUSS_H__ */
