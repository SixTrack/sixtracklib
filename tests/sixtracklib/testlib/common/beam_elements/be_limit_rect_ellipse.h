#ifndef SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_LIMIT_RECT_ELLIPSE_H__
#define SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_LIMIT_RECT_ELLIPSE_H__

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
    #include "sixtracklib/common/be_limit/be_limit_rect_ellipse.h"
    #include "sixtracklib/common/internal/type_comparison_helpers.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_STATIC SIXTRL_FN int NS(LimitRectEllipse_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN int NS(LimitRectEllipse_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(LimitRectEllipse_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(LimitRectEllipse_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit );

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

SIXTRL_INLINE int NS(LimitRectEllipse_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const SIXTRL_RESTRICT rhs
) SIXTRL_NOEXCEPT
{
    int cmp_result = -1;
    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;

        if( lhs != rhs )
        {
            cmp_result = NS(Type_value_comp_result)(
                NS(LimitRectEllipse_max_x)( lhs ),
                NS(LimitRectEllipse_max_x)( rhs ) );

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    NS(LimitRectEllipse_max_y)( lhs ),
                    NS(LimitRectEllipse_max_y)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    NS(LimitRectEllipse_a_squ)( lhs ),
                    NS(LimitRectEllipse_a_squ)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    NS(LimitRectEllipse_b_squ)( lhs ),
                    NS(LimitRectEllipse_b_squ)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    NS(LimitRectEllipse_a_squ_b_squ)( lhs ),
                    NS(LimitRectEllipse_a_squ_b_squ)( rhs ) );
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

SIXTRL_INLINE int NS(LimitRectEllipse_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const SIXTRL_RESTRICT rhs,
    NS(particle_real_t) const ABS_TOL ) SIXTRL_NOEXCEPT
{
    int cmp_result = -1;
    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;
        if( lhs != rhs )
        {
            NS(particle_real_t) const REL_TOL = ( NS(particle_real_t) )0;

            cmp_result = NS(Type_value_comp_result_with_tolerances)(
                NS(LimitRectEllipse_max_x)( lhs ),
                NS(LimitRectEllipse_max_x)( rhs ), REL_TOL, ABS_TOL );

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    NS(LimitRectEllipse_max_y)( lhs ),
                    NS(LimitRectEllipse_max_y)( rhs ), REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    NS(LimitRectEllipse_a_squ)( lhs ),
                    NS(LimitRectEllipse_a_squ)( rhs ),
                    REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    NS(LimitRectEllipse_b_squ)( lhs ),
                    NS(LimitRectEllipse_b_squ)( rhs ),
                    REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    NS(LimitRectEllipse_a_squ_b_squ)( lhs ),
                    NS(LimitRectEllipse_a_squ_b_squ)( rhs ),
                    REL_TOL, ABS_TOL );
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

SIXTRL_INLINE void NS(LimitRectEllipse_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(LimitRectEllipse) *const SIXTRL_RESTRICT limit )
{
    NS(LimitRectEllipse_print)( stdout, limit );
}

#else /* !defined( _GPUCODE ) */

SIXTRL_INLINE void NS(LimitRectEllipse_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(LimitRectEllipse) *const SIXTRL_RESTRICT limit )
{
    if( limit != SIXTRL_NULLPTR )
    {
        printf( "|limit_rect_ellips| half-axis x = %+21.18f m\r\n"
                "                  | half-axis y = %+21.18f m\r\n"
                "                  | x limit     = %+21.18f m\r\n"
                "                  | y limit     = %+21.18f m\r\n",
                NS(LimitRectEllipse_a)( limit ),
                NS(LimitRectEllipse_b)( limit ),
                NS(LimitRectEllipse_max_x)( limit ),
                NS(LimitRectEllipse_max_y)( limit ) );
    }
}

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */
#endif /* SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_LIMIT_RECT_ELLIPSE_H__ */
