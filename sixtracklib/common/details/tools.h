#ifndef SIXTRACKLIB_COMMON_TOOLS_H__
#define SIXTRACKLIB_COMMON_TOOLS_H__

#if !defined( _GPUCODE )
#include "sixtracklib/_impl/definitions.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>


#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */
    
#endif /* _GPUCODE */

SIXTRL_STATIC SIXTRL_UINT64_T NS(greatest_common_divisor)( 
    SIXTRL_UINT64_T a, SIXTRL_UINT64_T b );

SIXTRL_STATIC SIXTRL_UINT64_T NS(least_common_multiple)( 
    SIXTRL_UINT64_T a, SIXTRL_UINT64_T b );

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
    
#if !defined( _GPUCODE )

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* #if !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_TOOLS_H__ */

/*end: sixtracklib/common/details/tools.h */
