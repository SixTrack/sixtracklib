#ifndef SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_LIMIT_ELLIPSE_C99_H__
#define SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_LIMIT_ELLIPSE_C99_H__

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
    #include "sixtracklib/common/be_limit/be_limit_ellipse.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_STATIC SIXTRL_FN int NS(LimitEllipse_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT rhs );

SIXTRL_STATIC SIXTRL_FN int NS(LimitEllipse_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold );

SIXTRL_STATIC SIXTRL_FN void NS(LimitEllipse_print_out)(
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamMonitor) *const 
        SIXTRL_RESTRICT limit_ellipse );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(LimitEllipse_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const struct NS(LimitEllipse) *const 
        SIXTRL_RESTRICT limit_ellipse );

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/*        Implementation of inline functions for NS(LimitEllipse)            */
/* ========================================================================= */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/testlib/common/compare.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE int NS(LimitEllipse_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT rhs )
{
    return NS(LimitEllipse_compare_values_with_treshold)( 
        lhs, rhs, ( SIXTRL_REAL_T )0 );
}

SIXTRL_INLINE int NS(LimitEllipse_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold )
{
    int cmp_result = -1;
    
    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;
        
        if( lhs != rhs )
        {
            cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                NS(LimitEllipse_get_origin_x)( lhs ), 
                NS(LimitEllipse_get_origin_x)( rhs ), treshold );
            
            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                    NS(LimitEllipse_get_origin_y)( lhs ), 
                    NS(LimitEllipse_get_origin_y)( rhs ), treshold );
            }
            
            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                    NS(LimitEllipse_get_x_half_axis_squ)( lhs ), 
                    NS(LimitEllipse_get_x_half_axis_squ)( rhs ), treshold );
            }
            
            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                    NS(LimitEllipse_get_y_half_axis_squ)( lhs ), 
                    NS(LimitEllipse_get_y_half_axis_squ)( rhs ), treshold );
            }
            
            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                    NS(LimitEllipse_get_half_axis_product_squ)( lhs ), 
                    NS(LimitEllipse_get_half_axis_product_squ)( rhs ), 
                    treshold );
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
    
SIXTRL_INLINE void NS(LimitEllipse_print_out)(
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamMonitor) *const 
        SIXTRL_RESTRICT limit_ellipse )
{
    NS(LimitEllipse_print)( stdout, limit_ellipse );
}

#else /* !defined( _GPUCODE ) */

SIXTRL_INLINE void NS(LimitEllipse_print_out)(
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamMonitor) *const 
        SIXTRL_RESTRICT limit_ellipse )
{
    if( limit_ellipse != SIXTRL_NULLPTR )
    {
        printf( "|limit_ellipse    | origin x    = %+16.12f m;\r\n"
                "                  | origin y    = %+16.12f m;\r\n"
                "                  | half-axis x = %+16.12f m;\r\n"
                "                  | half-axis y = %+16.12f m;\r\n",
                NS(LimitEllipse_get_origin_x)( limit_ellipse ),
                NS(LimitEllipse_get_origin_y)( limit_ellipse ),
                NS(LimitEllipse_get_x_half_axis)( limit_ellipse ),
                NS(LimitEllipse_get_y_half_axis)( limit_ellipse ) );
    }
    
    return;
}

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */
    
#endif /* SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_LIMIT_ELLIPSE_C99_H__ */

/* end: tests/sixtracklib/testlib/common/beam_elements/be_limit_ellipse.h */
