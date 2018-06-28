#ifndef SIXTRACKLIB_COMMON_IMPL_ALIGNMENT_IMPL_H__
#define SIXTRACKLIB_COMMON_IMPL_ALIGNMENT_IMPL_H__

#if !defined ( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
#endif /* SIXTRL_NO_INCLUDES */

#if !defined ( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined ( _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */

SIXTRL_STATIC SIXTRL_UINT64_T NS(log2_floor)( SIXTRL_UINT64_T const x );
SIXTRL_STATIC SIXTRL_UINT64_T NS(log2_ceil)( SIXTRL_UINT64_T const x );

SIXTRL_STATIC SIXTRL_UINT64_T NS(greatest_common_divisor)(
    SIXTRL_UINT64_T a, SIXTRL_UINT64_T b );

SIXTRL_STATIC SIXTRL_UINT64_T NS(least_common_multiple)(
    SIXTRL_UINT64_T a, SIXTRL_UINT64_T b );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_UINT64_T NS(Alignment_calculate_common)(
    SIXTRL_UINT64_T s, SIXTRL_UINT64_T t );

SIXTRL_STATIC SIXTRL_UINT64_T NS(Alignment_calculate_common3)(
    SIXTRL_UINT64_T s, SIXTRL_UINT64_T t, SIXTRL_UINT64_T u );

SIXTRL_STATIC SIXTRL_UINT64_T NS(Alignment_calculate_common4)(
    SIXTRL_UINT64_T s, SIXTRL_UINT64_T t, SIXTRL_UINT64_T u,
    SIXTRL_UINT64_T v );

SIXTRL_STATIC SIXTRL_UINT64_T NS(Alignment_calculate_common5)(
    SIXTRL_UINT64_T s, SIXTRL_UINT64_T t, SIXTRL_UINT64_T u,
    SIXTRL_UINT64_T v, SIXTRL_UINT64_T w );

SIXTRL_STATIC SIXTRL_UINT64_T NS(Alignment_calculate_common6)(
    SIXTRL_UINT64_T s, SIXTRL_UINT64_T t, SIXTRL_UINT64_T u,
    SIXTRL_UINT64_T v, SIXTRL_UINT64_T w, SIXTRL_UINT64_T x );

SIXTRL_STATIC SIXTRL_UINT64_T NS(Alignment_calculate_common7)(
    SIXTRL_UINT64_T s, SIXTRL_UINT64_T t, SIXTRL_UINT64_T u,
    SIXTRL_UINT64_T v, SIXTRL_UINT64_T w, SIXTRL_UINT64_T x,
    SIXTRL_UINT64_T y );

SIXTRL_STATIC SIXTRL_UINT64_T NS(Alignment_calculate_common8)(
    SIXTRL_UINT64_T s, SIXTRL_UINT64_T t, SIXTRL_UINT64_T u,
    SIXTRL_UINT64_T v, SIXTRL_UINT64_T w, SIXTRL_UINT64_T x,
    SIXTRL_UINT64_T y, SIXTRL_UINT64_T z );

/* ========================================================================= */

SIXTRL_INLINE SIXTRL_UINT64_T NS(log2_floor)( SIXTRL_UINT64_T const x )
{
    static SIXTRL_UINT64_T const MAX_TRESHOLD = 0x7FFFFFFFFFFFFFFF;
    SIXTRL_UINT64_T result = ( SIXTRL_UINT64_T )0u;

    if( x > MAX_TRESHOLD )
    {
        result = ( SIXTRL_UINT64_T )63u;
    }
    else if( x > ( SIXTRL_UINT64_T )0u )
    {
        static SIXTRL_UINT64_T const ONE = ( SIXTRL_UINT64_T )1u;
        SIXTRL_UINT64_T N = ( SIXTRL_UINT64_T )1u;

        for( result = 0u ; N < 64 ; result = N++ )
        {
            SIXTRL_UINT64_T const CMP = ( ( ONE << N ) - ONE );
            if( x < CMP ) break;
        }
    }

    return result;
}

SIXTRL_INLINE SIXTRL_UINT64_T NS(log2_ceil)( SIXTRL_UINT64_T const x )
{
    static SIXTRL_UINT64_T const MAX_FLOOR_N = 63u;

    SIXTRL_UINT64_T const floor_result = NS(log2_floor)( x );
    return ( floor_result < MAX_FLOOR_N ) ? floor_result + 1u : floor_result;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_UINT64_T NS(greatest_common_divisor)(
    SIXTRL_UINT64_T a, SIXTRL_UINT64_T b )
{
    static SIXTRL_UINT64_T const ZERO = ( SIXTRL_UINT64_T )0u;

    while( ( a != ZERO ) && ( b != ZERO ) )
    {
        if( a == ZERO ) return b;
        b %= a;

        if( b == ZERO ) return a;
        a %= b;
    }

    return ( a > b ) ? a : b;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_UINT64_T NS(least_common_multiple)(
    SIXTRL_UINT64_T a, SIXTRL_UINT64_T b )
{
    static SIXTRL_UINT64_T const ZERO = ( SIXTRL_UINT64_T )0u;
    SIXTRL_UINT64_T const gcd = NS(greatest_common_divisor)( a, b );

    return ( gcd != ZERO ) ? ( ( a * b ) / gcd ) : ( ZERO );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_UINT64_T NS(Alignment_calculate_common)(
    SIXTRL_UINT64_T s, SIXTRL_UINT64_T t )
{
    return NS(least_common_multiple)( s, t );
}

SIXTRL_INLINE SIXTRL_UINT64_T NS(Alignment_calculate_common3)(
    SIXTRL_UINT64_T s, SIXTRL_UINT64_T t, SIXTRL_UINT64_T u )
{
    return NS(least_common_multiple)(
        NS(least_common_multiple)( s, t ), u );
}

SIXTRL_INLINE SIXTRL_UINT64_T NS(Alignment_calculate_common4)(
    SIXTRL_UINT64_T s, SIXTRL_UINT64_T t,
    SIXTRL_UINT64_T u, SIXTRL_UINT64_T v )
{
    return NS(least_common_multiple)(
        NS(least_common_multiple)( s, t ),
        NS(least_common_multiple)( u, v ) );
}

SIXTRL_INLINE SIXTRL_UINT64_T NS(Alignment_calculate_common5)(
    SIXTRL_UINT64_T s, SIXTRL_UINT64_T t, SIXTRL_UINT64_T u,
    SIXTRL_UINT64_T v, SIXTRL_UINT64_T w )
{
    return NS(Alignment_calculate_common)(
           NS(Alignment_calculate_common4)( s, t, u, v ), w );
}

SIXTRL_INLINE SIXTRL_UINT64_T NS(Alignment_calculate_common6)(
    SIXTRL_UINT64_T s, SIXTRL_UINT64_T t, SIXTRL_UINT64_T u,
    SIXTRL_UINT64_T v, SIXTRL_UINT64_T w, SIXTRL_UINT64_T x )
{
    return NS(Alignment_calculate_common)(
        NS(Alignment_calculate_common4)( s, t, u, v ),
        NS(Alignment_calculate_common)( w, x ) );

}

SIXTRL_INLINE SIXTRL_UINT64_T NS(Alignment_calculate_common7)(
    SIXTRL_UINT64_T s, SIXTRL_UINT64_T t, SIXTRL_UINT64_T u,
    SIXTRL_UINT64_T v, SIXTRL_UINT64_T w, SIXTRL_UINT64_T x,
    SIXTRL_UINT64_T y )
{
    return NS(Alignment_calculate_common)(
        NS(Alignment_calculate_common4)( s, t, u, v ),
        NS(Alignment_calculate_common3)( w, x, y ) );
}

SIXTRL_INLINE SIXTRL_UINT64_T NS(Alignment_calculate_common8)(
    SIXTRL_UINT64_T s, SIXTRL_UINT64_T t, SIXTRL_UINT64_T u,
    SIXTRL_UINT64_T v, SIXTRL_UINT64_T w, SIXTRL_UINT64_T x,
    SIXTRL_UINT64_T y, SIXTRL_UINT64_T z )
{
    return NS(Alignment_calculate_common)(
        NS(Alignment_calculate_common4)( s, t, u, v ),
        NS(Alignment_calculate_common4)( w, x, y, z ) );
}


#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_ALIGNMENT_IMPL_H__ */

/* end: sixtracklib/common/impl/alignment_impl.h */
