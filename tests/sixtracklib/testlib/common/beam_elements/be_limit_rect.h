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
    #include "sixtracklib/common/internal/type_comparison_helpers.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_STATIC SIXTRL_FN int NS(LimitRect_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const
        SIXTRL_RESTRICT rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN int NS(LimitRect_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold ) SIXTRL_NOEXCEPT;

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
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const
        SIXTRL_RESTRICT rhs ) SIXTRL_NOEXCEPT
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;

        if( lhs != rhs )
        {
            cmp_result = NS(Type_value_comp_result)(
                NS(LimitRect_min_x)( lhs ), NS(LimitRect_min_x)( rhs ) );

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    NS(LimitRect_max_x)( lhs ), NS(LimitRect_max_x)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    NS(LimitRect_min_y)( lhs ), NS(LimitRect_min_y)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    NS(LimitRect_max_y)( lhs ), NS(LimitRect_max_y)( rhs ) );
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

SIXTRL_INLINE int NS(LimitRect_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT rhs,
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
                NS(LimitRect_min_x)( lhs ), NS(LimitRect_min_x)( rhs ),
                        REL_TOL, ABS_TOL );

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    NS(LimitRect_max_x)( lhs ), NS(LimitRect_max_x)( rhs ),
                        REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    NS(LimitRect_min_y)( lhs ), NS(LimitRect_min_y)( rhs ),
                        REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    NS(LimitRect_max_y)( lhs ), NS(LimitRect_max_y)( rhs ),
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
        printf( "|limit_rect       | min_x    = %+20.12f m;\r\n"
                "                  | max_x    = %+20.12f m;\r\n"
                "                  | min_y    = %+20.12f m;\r\n"
                "                  | max_y    = %+20.12f m;\r\n",
                NS(LimitRect_min_x)( limit ), NS(LimitRect_max_x)( limit ),
                NS(LimitRect_min_y)( limit ), NS(LimitRect_max_y)( limit ) );
    }
}

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_LIMIT_RECT_C99_H__ */
/* end: tests/sixtracklib/testlib/common/beam_elements/be_limit_rect.h */
