#ifndef SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_MULTIPOLE_H__
#define SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_MULTIPOLE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>

    #if !defined( _GPUCODE )
    #include <stdio.h>
    #endif /* !defined( _GPUCODE ) */

#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/be_multipole/be_multipole.h"
    #include "sixtracklib/common/internal/type_comparison_helpers.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_STATIC SIXTRL_FN int NS(Multipole_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const
        SIXTRL_RESTRICT rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN int NS(Multipole_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT rhs,
    NS(multipole_real_t) const treshold ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(Multipole_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT e );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Multipole_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT mp );

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */


#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE int NS(Multipole_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const
        SIXTRL_RESTRICT rhs ) SIXTRL_NOEXCEPT
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;

        if( lhs != rhs )
        {
            cmp_result = ( lhs->order == rhs->order )
                ? 0 : ( ( lhs->order > rhs->order ) ? +1 : -1 );

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    NS(Multipole_length)( lhs ), NS(Multipole_length)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    NS(Multipole_hxl)( lhs ), NS(Multipole_hxl)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    NS(Multipole_hyl)( lhs ), NS(Multipole_hyl)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_for_range)(
                    NS(Multipole_const_bal_begin)( lhs ),
                    NS(Multipole_const_bal_end)( lhs ),
                    NS(Multipole_const_bal_begin)( rhs ) );
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

SIXTRL_INLINE int NS(Multipole_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT rhs,
    NS(multipole_real_t) const ABS_TOL ) SIXTRL_NOEXCEPT
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;

        if( lhs != rhs )
        {
            NS(multipole_real_t) const REL_TOL = ( NS(multipole_real_t) )0;

            cmp_result = ( lhs->order == rhs->order )
                ? 0 : ( ( lhs->order > rhs->order ) ? +1 : -1 );

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    NS(Multipole_length)( lhs ), NS(Multipole_length)( rhs ),
                        REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    NS(Multipole_hxl)( lhs ), NS(Multipole_hxl)( rhs ),
                        REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    NS(Multipole_hyl)( lhs ), NS(Multipole_hyl)( rhs ),
                        REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result =
                NS(Type_value_comp_result_with_tolerances_for_range)(
                    NS(Multipole_const_bal_begin)( lhs ),
                    NS(Multipole_const_bal_end)( lhs ),
                    NS(Multipole_const_bal_begin)( rhs ), REL_TOL, ABS_TOL );
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

SIXTRL_INLINE void NS(Multipole_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT mp )
{
    if( mp != SIXTRL_NULLPTR )
    {
        NS(multipole_order_t) const order = NS(Multipole_order)( mp );

        printf( "|multipole        | order    = %3ld;\r\n"
                "                  | length   = %+20.12f m;\r\n"
                "                  | hxl      = %+20.12f m;\r\n"
                "                  | hyl      = %+20.12f m;\r\n",
                ( long int )order, NS(Multipole_length)( mp ),
                NS(Multipole_hxl)( mp ), NS(Multipole_hyl)( mp ) );

        printf( "                  |"
                    "    idx"
                    "                  knl" "                  ksl\r\n" );

        if( order >= ( NS(multipole_order_t) )0 )
        {
            NS(multipole_order_t) ii = ( NS(multipole_order_t) )0;
            for( ; ii <= order ; ++ii )
            {
                printf( "                  | %6ld %+20.12f %+20.12f\r\n",
                        ( long int )ii,
                        NS(Multipole_knl)( mp, ii ),
                        NS(Multipole_ksl)( mp, ii ) );
            }
        }
        else
        {
            printf( "                  |"
                    "    ---"
                    "                  n/a" "                  n/a\r\n" );
        }
    }
}

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */
#endif /* SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_MULTIPOLE_H__ */

