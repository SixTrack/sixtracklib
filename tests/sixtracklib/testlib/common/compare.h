#ifndef SIXTRACKLIB_TESTLIB_COMMON_COMPARE_C99_H__
#define SIXTRACKLIB_TESTLIB_COMMON_COMPARE_C99_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"        
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_STATIC SIXTRL_FN 
int NS(TestLibCompare_real_attribute)(
    SIXTRL_REAL_T const lhs_attribute, SIXTRL_REAL_T const rhs_attribute );

SIXTRL_STATIC SIXTRL_FN 
int NS(TestLibCompare_real_attribute_with_treshold)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs, 
    SIXTRL_REAL_T const treshold );

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* ************               Inline implementation           ************** */
/* ************************************************************************* */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE int NS(TestLibCompare_real_attribute)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs )
{
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO = ( SIXTRL_REAL_T )0;
    return NS(TestLibCompare_real_attribute_with_treshold)( lhs, rhs, ZERO );
}

SIXTRL_INLINE int NS(TestLibCompare_real_attribute_with_treshold)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs, 
    SIXTRL_REAL_T const treshold )
{
    int cmp_result = ( int )-1;
    
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO = ( SIXTRL_REAL_T )0.0;
    SIXTRL_REAL_T const delta = lhs - rhs;
    
    if( delta == ZERO )
    {
        cmp_result = 0;        
    }
    else if( treshold >= ZERO )
    {
        cmp_result = 0;
        
        if( ( delta >= ZERO ) && ( delta > treshold ) )
        {
            cmp_result = -1;
        }
        else if( ( delta < ZERO ) && ( delta < -treshold ) )
        {
            cmp_result = +1;
        }
    }
    
    return cmp_result;
}

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */
    
#endif /* SIXTRACKLIB_TESTLIB_COMMON_COMPARE_C99_H__ */
/* end: tests/sixtracklib/testlib/common/compare.h */
