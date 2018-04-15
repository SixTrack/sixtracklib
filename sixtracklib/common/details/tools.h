#ifndef SIXTRACKLIB_COMMON_TOOLS_H__
#define SIXTRACKLIB_COMMON_TOOLS_H__

#include "sixtracklib/_impl/definitions.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */
    
SIXTRL_STATIC uintmax_t NS(greatest_common_divisor)( uintmax_t a, uintmax_t b );
SIXTRL_STATIC uintmax_t NS(least_common_multiple)( uintmax_t a, uintmax_t b );

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE uintmax_t NS(greatest_common_divisor)( uintmax_t a, uintmax_t b )
{
    static uintmax_t const ZERO = ( uintmax_t )0u;
    
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

SIXTRL_INLINE uintmax_t NS(least_common_multiple)( uintmax_t a, uintmax_t b )
{
    static uintmax_t const ZERO = ( uintmax_t )0u;
    uintmax_t const gcd = NS(greatest_common_divisor)( a, b );
    
    return ( gcd != ZERO ) ? ( ( a * b ) / gcd ) : ( ZERO );
}

/* ------------------------------------------------------------------------- */
    
#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_TOOLS_H__ */

/*end: sixtracklib/common/details/tools.h */
