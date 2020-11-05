#ifndef SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_CAVITY_H__
#define SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_CAVITY_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>

    #if !defined( _GPUCODE )
    #include <stdio.h>
    #endif /* !defined( _GPUCODE ) */

#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_cavity/be_cavity.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/internal/type_comparison_helpers.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* C++, Host */

SIXTRL_STATIC SIXTRL_FN void NS(Cavity_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(Cavity) *const SIXTRL_RESTRICT cavity ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Cavity_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const struct NS(Cavity) *const SIXTRL_RESTRICT e );

#endif /* !defined( _GPUCODE ) */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN int NS(Cavity_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT rhs
) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN int NS(Cavity_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* C++, Host */

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* C++, Host */

SIXTRL_INLINE void NS(Cavity_print_out)( SIXTRL_ARGPTR_DEC const NS(Cavity)
    *const SIXTRL_RESTRICT cavity ) SIXTRL_NOEXCEPT
{
    printf( "|cavity           | voltage   = %+16.12f V;\r\n"
            "                  | frequency = %+20.12f Hz;\r\n"
            "                  | lag       = %+15.12f deg;\r\n",
            NS(Cavity_voltage)( cavity ),
            NS(Cavity_frequency)( cavity ),
            NS(Cavity_lag)( cavity ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE int NS(Cavity_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT rhs
) SIXTRL_NOEXCEPT
{
    int cmp_value = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_value = 0;

        if( lhs != rhs )
        {
            cmp_value = NS(Type_value_comp_result)( lhs->voltage, rhs->voltage );

            if( cmp_value == 0 )
            {
                cmp_value = NS(Type_value_comp_result)(
                    lhs->frequency, rhs->frequency );
            }

            if( cmp_value == 0 )
            {
                cmp_value = NS(Type_value_comp_result)( lhs->lag, rhs->lag );
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_value = +1;
    }

    return cmp_value;
}

SIXTRL_INLINE int NS(Cavity_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold ) SIXTRL_NOEXCEPT
{
    int cmp_value = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) &&
        ( treshold >= ( SIXTRL_REAL_T )0 ) )
    {
        cmp_value = 0;

        if( rhs != lhs )
        {
            cmp_value = NS(Type_value_comp_result_with_tolerances)(
                lhs->voltage, rhs->voltage, ( SIXTRL_REAL_T )0, treshold );

            if( cmp_value == 0 )
            {
                cmp_value = NS(Type_value_comp_result_with_tolerances)(
                    lhs->frequency, rhs->frequency, ( SIXTRL_REAL_T )0,
                        treshold );
            }

            if( cmp_value == 0 )
            {
                cmp_value = NS(Type_value_comp_result_with_tolerances)(
                    lhs->lag, rhs->lag, ( SIXTRL_REAL_T )0, treshold );
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
#endif /* SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_CAVITY_H__ */
