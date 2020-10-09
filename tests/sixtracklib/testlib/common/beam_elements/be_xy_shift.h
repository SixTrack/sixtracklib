#ifndef SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_XY_SHIFT_H__
#define SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_XY_SHIFT_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>

    #if !defined( _GPUCODE )
    #include <stdio.h>
    #endif /* !defined( _GPUCODE ) */

#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_xyshift/be_xyshift.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/internal/type_comparison_helpers.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* C++, Host */

SIXTRL_STATIC SIXTRL_FN void NS(XYShift_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(XYShift) *const SIXTRL_RESTRICT xyshift );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(XYShift_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xyshift );

#endif /* !defined( _GPUCODE ) */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN int NS(XYShift_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT rhs
) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN int NS(XYShift_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const ABS_TOL ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* C++, Host */

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* C++, Host */

SIXTRL_INLINE void NS(XYShift_print_out)( SIXTRL_ARGPTR_DEC const NS(XYShift)
    *const SIXTRL_RESTRICT xyshift )
{
    printf( "|xy_shift         | dx        = %+20.14f m\r\n"
            "                  | dy        = %+20.14f m\r\n",
            NS(XYShift_dx)( xyshift ), NS(XYShift_dy)( xyshift ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE int NS(XYShift_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT rhs
) SIXTRL_NOEXCEPT
{
    int cmp_value = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_value = 0;

        if( lhs != rhs )
        {
            cmp_value = NS(Type_value_comp_result)( lhs->dx, rhs->dx );

            if( cmp_value == 0 )
            {
                cmp_value = NS(Type_value_comp_result)( lhs->dy, rhs->dy );
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_value = +1;
    }

    return cmp_value;
}

SIXTRL_INLINE int NS(XYShift_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const ABS_TOL ) SIXTRL_NOEXCEPT
{
    int cmp_value = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) &&
        ( ABS_TOL >= ( SIXTRL_REAL_T )0 ) )
    {
        cmp_value = 0;

        if( rhs != lhs )
        {
            SIXTRL_REAL_T const REL_TOL = ( SIXTRL_REAL_T )0;

            cmp_value = NS(Type_value_comp_result_with_tolerances)(
                lhs->dx, rhs->dx, REL_TOL, ABS_TOL );

            if( cmp_value == 0 )
            {
                cmp_value = NS(Type_value_comp_result_with_tolerances)(
                    lhs->dy, rhs->dy, REL_TOL, ABS_TOL );
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_value = +1;
    }

    return cmp_value;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* C++, Host */
#endif /* SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_XY_SHIFT_H__ */
