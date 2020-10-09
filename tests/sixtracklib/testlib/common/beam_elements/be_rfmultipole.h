#ifndef SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_RF_MULTIPOLE_H__
#define SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_RF_MULTIPOLE_H__

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
    #include "sixtracklib/common/be_rfmultipole/be_rfmultipole.h"
    #include "sixtracklib/common/internal/type_comparison_helpers.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_STATIC SIXTRL_FN int NS(RFMultipole_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN int NS(RFMultipole_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT rhs,
    NS(rf_multipole_real_t) const ABS_TOL ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(RFMultipole_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT e );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(RFMultipole_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT mp );

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */


#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE int NS(RFMultipole_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const
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
                    NS(RFMultipole_voltage)( lhs ),
                    NS(RFMultipole_voltage)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    NS(RFMultipole_frequency)( lhs ),
                    NS(RFMultipole_frequency)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    NS(RFMultipole_lag)( lhs ),
                    NS(RFMultipole_lag)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_for_range)(
                    NS(RFMultipole_const_bal_begin)( lhs ),
                    NS(RFMultipole_const_bal_end)( lhs ),
                    NS(RFMultipole_const_bal_begin)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_for_range)(
                    NS(RFMultipole_const_phase_begin)( lhs ),
                    NS(RFMultipole_const_phase_end)( lhs ),
                    NS(RFMultipole_const_phase_begin)( rhs ) );
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

SIXTRL_INLINE int NS(RFMultipole_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT rhs,
    NS(rf_multipole_real_t) const ABS_TOL ) SIXTRL_NOEXCEPT
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;

        if( lhs != rhs )
        {
            NS(rf_multipole_real_t) const REL_TOL = (
                NS(rf_multipole_real_t) )0;

            cmp_result = ( lhs->order == rhs->order )
                ? 0 : ( ( lhs->order > rhs->order ) ? +1 : -1 );

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    NS(RFMultipole_voltage)( lhs ),
                    NS(RFMultipole_voltage)( rhs ), REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    NS(RFMultipole_frequency)( lhs ),
                    NS(RFMultipole_frequency)( rhs ), REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    NS(RFMultipole_lag)( lhs ),
                    NS(RFMultipole_lag)( rhs ), REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result =
                NS(Type_value_comp_result_with_tolerances_for_range)(
                    NS(RFMultipole_const_bal_begin)( lhs ),
                    NS(RFMultipole_const_bal_end)( lhs ),
                    NS(RFMultipole_const_bal_begin)( rhs ), REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result =
                NS(Type_value_comp_result_with_tolerances_for_range)(
                    NS(RFMultipole_const_phase_begin)( lhs ),
                    NS(RFMultipole_const_phase_end)( lhs ),
                    NS(RFMultipole_const_phase_begin)( rhs ),
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

SIXTRL_INLINE void NS(RFMultipole_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT mp )
{
    if( mp != SIXTRL_NULLPTR )
    {
        NS(rf_multipole_int_t) const order = NS(RFMultipole_order)( mp );

        printf( "|rf-multipole     | order      = %3ld;\r\n"
                "                  | voltage    = %+20.12f m\r\n"
                "                  | frequency  = %+20.12f m\r\n"
                "                  | lag        = %+20.12f m\r\n"
                "                  | bal_addr   = %21lx\r\n"
                "                  | phase_addr = %21lx\r\n",
                ( long int )order,
                NS(RFMultipole_voltage)( mp ),
                NS(RFMultipole_frequency)( mp ),
                NS(RFMultipole_lag)( mp ),
                ( uintptr_t )NS(RFMultipole_bal_addr)( mp ),
                ( uintptr_t )NS(RFMultipole_phase_addr)( mp ) );

        printf( "                  |"
                    "    idx"
                    "                  knl"
                    "                  ksl"
                    "               phase_n"
                    "               phase_s\r\n" );

        if( order >= ( NS(rf_multipole_int_t) )0 )
        {
            NS(rf_multipole_int_t) ii = ( NS(rf_multipole_int_t) )0;
            for( ; ii <= order ; ++ii )
            {
                printf( "                  | %6ld %+20.12f %+20.12f "
                        "%+20.12f %+20.12f\r\n",
                        ( long int )ii,
                        NS(RFMultipole_knl)( mp, ii ),
                        NS(RFMultipole_ksl)( mp, ii ),
                        NS(RFMultipole_phase_n)( mp, ii ),
                        NS(RFMultipole_phase_s)( mp, ii ) );
            }
        }
        else
        {
            printf( "                  |"
                    "    ---"
                    "                  n/a"
                    "                  n/a"
                    "                  n/a"
                    "                  n/a\r\n" );
        }
    }
}

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */
#endif /* SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_RF_MULTIPOLE_H__ */
