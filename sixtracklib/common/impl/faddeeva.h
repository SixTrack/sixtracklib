#ifndef SIXTRLIB_COMMON_IMPL_FADDEEVA_ERROR_FUNCTION_H__
#define SIXTRLIB_COMMON_IMPL_FADDEEVA_ERROR_FUNCTION_H__

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/_impl/definitions.h"

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

SIXTRL_FN SIXTRL_STATIC int NS(Faddeeva_calculate_w_cern335)( 
    SIXTRL_REAL_T* SIXTRL_RESTRICT re, SIXTRL_REAL_T* SIXTRL_RESTRICT im,
    SIXTRL_REAL_T const in_re, SIXTRL_REAL_T const in_im );

SIXTRL_FN int NS(Faddeeva_calculate_w_mit)( 
    SIXTRL_REAL_T* SIXTRL_RESTRICT re, SIXTRL_REAL_T* SIXTRL_RESTRICT im,
    SIXTRL_REAL_T const in_re, SIXTRL_REAL_T const in_im, 
    SIXTRL_REAL_T rel_error, 
    int const use_continued_fractions );


SIXTRL_INLINE int NS(Faddeeva_calculate_w_cern335)( 
    SIXTRL_REAL_T* SIXTRL_RESTRICT out_re, 
    SIXTRL_REAL_T* SIXTRL_RESTRICT out_im,
    SIXTRL_REAL_T const re, SIXTRL_REAL_T const im )
{
    int ret = ( ( out_re != 0 ) && ( out_im != 0 ) ) ? 0 : -1;
    
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO     = ( SIXTRL_REAL_T )0.0L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE      = ( SIXTRL_REAL_T )1.0L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE_HALF = ( SIXTRL_REAL_T )0.5L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const TWO      = ( SIXTRL_REAL_T )2.0L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const X_MAX    = ( SIXTRL_REAL_T )5.33L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const Y_MAX    = ( SIXTRL_REAL_T )4.29L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const FACTOR   = 
        ( SIXTRL_REAL_T )1.1283791670955099L;
    
    SIXTRL_REAL_T const x = ( re >= ZERO ) ? re : -re;
    SIXTRL_REAL_T const y = ( im >= ZERO ) ? im : -im;
        
    if( ret != 0 ) return ret;
    
    if( ( x < X_MAX ) && ( y < Y_MAX ) )
    {
        SIXTRL_STATIC_VAR SIXTRL_REAL_T const C1 = ( SIXTRL_REAL_T )3.2L;
            
        SIXTRL_REAL_T const x_rel = x / X_MAX;
        SIXTRL_REAL_T const y_rel = y / Y_MAX;
        
        SIXTRL_REAL_T const q  = ( ONE - y_rel ) * sqrt( ONE - ( x_rel * x_rel) );
        SIXTRL_REAL_T const h  = ONE / ( C1 * q );
        
        int const nc =  7 + ( int )( 23.0 * q );
        int const nu = 10 + ( int )( 21.0 * q );
        
        SIXTRL_REAL_T RX[ 33 ] = 
        {
            ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 
            ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 
            ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 
            ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO
        };
        
        SIXTRL_REAL_T RY[ 33 ] = 
        {
            ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 
            ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 
            ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 
            ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO
        };
        
        SIXTRL_REAL_T xl = pow( h, ( double )( ONE - nc ) );
        SIXTRL_REAL_T xh = y + ONE_HALF / h;
        SIXTRL_REAL_T yh = x;
                
        SIXTRL_REAL_T sx = ZERO;
        SIXTRL_REAL_T sy = ZERO;
        
        int n = nu;
        
        for( ; n > 0 ; --n )
        {
            SIXTRL_REAL_T const tx = xh + n * RX[ n ];
            SIXTRL_REAL_T const ty = yh - n * RY[ n ];
            SIXTRL_REAL_T const c = ONE_HALF / ( tx * tx + ty * ty );
            
            RX[ n - 1 ] = tx * c;
            RY[ n - 1 ] = ty * c;            
        }
        
        for( n = nc ; n > 0 ; --n )
        {
            SIXTRL_REAL_T const s_aux = sx + xl;
            sx = RX[ n - 1 ] * s_aux - RY[ n - 1 ] * sy;
            sy = RX[ n - 1 ] * sy    + RY[ n - 1 ] * s_aux;
            xl = h * xl;
        }
        
        *out_re = FACTOR * sx;
        *out_im = FACTOR * sy;
    }
    else
    {
        SIXTRL_REAL_T xh = y;
        SIXTRL_REAL_T yh = x;
        
        SIXTRL_REAL_T rx = ZERO;
        SIXTRL_REAL_T ry = ZERO;
        
        int n = 9;
        
        for( ; n > 0 ; --n )
        {
            SIXTRL_REAL_T const tx = xh + n * rx;
            SIXTRL_REAL_T const ty = yh - n * ry;
            SIXTRL_REAL_T const c = ONE_HALF / ( tx * tx + ty * ty );
            
            rx = tx * c;
            ry = ty * c;            
        }
        
        *out_re = FACTOR * rx;
        *out_im = FACTOR * ry;        
    }
    
    if( im == ZERO )
    {
        *out_re = exp( -x * x );
    }
    
    if( im < ZERO )
    {
        SIXTRL_REAL_T const C1 = TWO * exp( y * y - x * x );            
        SIXTRL_REAL_T const C2 = TWO * x * y;
            
        *out_re =  C1 * cos( C2 ) - *out_re;
        *out_im = -C1 * sin( C2 ) - *out_im;
        
        if( re > ZERO )
        {
            *out_im = -( *out_im );
        }
    }
    else if( re < ZERO )
    {
        *out_im = -( *out_im );
    }
    
    return ret;
}

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* SIXTRLIB_COMMON_IMPL_FADDEEVA_ERROR_FUNCTION_H__ */

/* end: sixtracklib/common/impl/faddeeva.h */
