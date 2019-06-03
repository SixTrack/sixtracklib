#ifndef SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_LIMIT_RECT_C99_H__
#define SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_LIMIT_RECT_C99_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>
    
    #if !defined( _GPUCODE )
    #include <stdio.h>
    #endif /* !defined( _GPUCODE ) */
    
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_limit/definitions.h"
    #include "sixtracklib/common/be_limit/be_limit_rect.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_STATIC SIXTRL_FN int NS(LimitRect_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT rhs );

SIXTRL_STATIC SIXTRL_FN int NS(LimitRect_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold );

SIXTRL_STATIC SIXTRL_FN void NS(LimitRect_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT limit );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(LimitRect_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT limit );

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/*        Implementation of inline functions for NS(LimitRect)               */
/* ========================================================================= */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/testlib/common/compare.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE int NS(LimitRect_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT rhs )
{
    return NS(LimitRect_compare_values_with_treshold)( 
        lhs, rhs, ( SIXTRL_REAL_T )0 );
}

SIXTRL_INLINE int NS(LimitRect_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold )
{
    int cmp_result = -1;
    
    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;
        
        if( lhs != rhs )
        {
            cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                NS(LimitRect_get_min_x)( lhs ), NS(LimitRect_get_min_x)( rhs ), 
                    treshold );
            
            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                    NS(LimitRect_get_max_x)( lhs ), 
                    NS(LimitRect_get_max_x)( rhs ), treshold );
            }
            
            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                    NS(LimitRect_get_min_y)( lhs ), 
                    NS(LimitRect_get_min_y)( rhs ), treshold );
            }
            
            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                    NS(LimitRect_get_max_y)( lhs ), 
                    NS(LimitRect_get_max_y)( rhs ), treshold );
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }
    
    return cmp_result;
}

#if !defined( _GPUCODE )
    
SIXTRL_INLINE void NS(LimitRect_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT limit )
{
    NS(LimitRect_print)( stdout, limit );
}

#else /* !defined( _GPUCODE ) */

SIXTRL_INLINE void NS(LimitRect_print_out)( 
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT limit )
{
    if( limit != SIXTRL_NULLPTR )
    {
        printf( "|limit_rect       | min_x    = %+16.12f m;\r\n"
                "                  | max_x    = %+16.12f m;\r\n"
                "                  | min_y    = %+16.12f m;\r\n"
                "                  | max_y    = %+16.12f m;\r\n",
                NS(LimitRect_get_min_x)( limit ),
                NS(LimitRect_get_max_x)( limit ),
                NS(LimitRect_get_min_y)( limit ),
                NS(LimitRect_get_max_y)( limit ) );
    }
    
    return;
}

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */
    
#endif /* SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_LIMIT_RECT_C99_H__ */
/* end: tests/sixtracklib/testlib/common/beam_elements/be_limit_rect.h */
