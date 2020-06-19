#ifndef SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_TRICUB_C99_H__
#define SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_TRICUB_C99_H__

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
    #include "sixtracklib/common/be_tricub/be_tricub.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_STATIC SIXTRL_FN int NS(TriCub_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT rhs );

SIXTRL_STATIC SIXTRL_FN int NS(TriCub_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold );

SIXTRL_STATIC SIXTRL_FN void NS(TriCub_print_out)(
    SIXTRL_BE_ARGPTR_DEC const struct NS(TriCub) *const SIXTRL_RESTRICT e );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TriCub_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const struct NS(TriCub) *const SIXTRL_RESTRICT e );

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====             Implementation of inline functions                ===== */
/* ========================================================================= */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/testlib/common/compare.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE int NS(TriCub_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT rhs )
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;

        if( lhs != rhs )
        {
            cmp_result = NS(TestLibCompare_real_attribute)(
                NS(TriCub_x_shift)( lhs ), NS(TriCub_x_shift)( rhs ) );

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute)(
                    NS(TriCub_y_shift)( lhs ), NS(TriCub_y_shift)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute)(
                    NS(TriCub_tau_shift)( lhs ), NS(TriCub_tau_shift)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute)(
                    NS(TriCub_dipolar_kick_px)( lhs ), NS(TriCub_dipolar_kick_px)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute)(
                    NS(TriCub_dipolar_kick_py)( lhs ), NS(TriCub_dipolar_kick_py)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute)(
                    NS(TriCub_dipolar_kick_ptau)( lhs ), NS(TriCub_dipolar_kick_ptau)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute)(
                    NS(TriCub_length)( lhs ), NS(TriCub_length)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_int64_attribute)(
                    NS(TriCub_data_addr)( lhs ), NS(TriCub_data_addr)( rhs ) );
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

SIXTRL_INLINE int NS(TriCub_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold )
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;

        if( lhs != rhs )
        {
            cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                NS(TriCub_x_shift)( lhs ), NS(TriCub_x_shift)( rhs ), treshold );

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                    NS(TriCub_y_shift)( lhs ), NS(TriCub_y_shift)( rhs ), treshold );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                    NS(TriCub_tau_shift)( lhs ), NS(TriCub_tau_shift)( rhs ), treshold );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                    NS(TriCub_dipolar_kick_px)( lhs ), NS(TriCub_dipolar_kick_px)( rhs ), treshold );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                    NS(TriCub_dipolar_kick_py)( lhs ), NS(TriCub_dipolar_kick_py)( rhs ), treshold );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                    NS(TriCub_dipolar_kick_ptau)( lhs ), NS(TriCub_dipolar_kick_ptau)( rhs ), treshold );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                    NS(TriCub_length)( lhs ), NS(TriCub_length)( rhs ),
                        treshold );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_int64_attribute)(
                    NS(TriCub_data_addr)( lhs ), NS(TriCub_data_addr)( rhs ) );
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

SIXTRL_INLINE void NS(TriCub_print_out)(
    SIXTRL_BE_ARGPTR_DEC const struct NS(TriCub) *const SIXTRL_RESTRICT e )
{
    #if defined( _GPUCODE )

    SIXTRL_ASSERT( e != SIXTRL_NULLPTR );

    printf( "|tricub          | x_shift             = %+20.12f\r\n"
            "                 | y_shift             = %+20.12f\r\n"
            "                 | tau_shift          = %+20.12f\r\n"
            "                 | dipolar_kick_px     = %+20.12f\r\n"
            "                 | dipolar_kick_py     = %+20.12f\r\n"
            "                 | dipolar_kick_ptau   = %+20.12f\r\n"
            "                 | length              = %+20.12f\r\n"
            "                 | table_addr          = %20lu\r\n",
            NS(TriCub_x_shift)( e ), NS(TriCub_y_shift)( e ),
            NS(TriCub_tau_shift)( e ), NS(TriCub_dipolar_kick_px)( e ),
            NS(TriCub_dipolar_kick_py)( e ), NS(TriCub_dipolar_kick_ptau)( e ),
            NS(TriCub_length)( e ),
            ( long unsigned )NS(TriCub_data_addr)( e ) );

    #else

    NS(TriCub_print)( stdout, e );

    #endif /* !defined( _GPUCODE ) */
}

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_TRICUB_C99_H__ */
