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
            typedef NS(be_tricub_int_t)  int_t;

            NS(be_tricub_ptr_const_real_t) lhs_phi =
                NS(TriCub_get_ptr_const_phi)( lhs );

            NS(be_tricub_ptr_const_real_t) rhs_phi =
                NS(TriCub_get_ptr_const_phi)( lhs );

            cmp_result = NS(TestLibCompare_int64_attribute)(
                NS(TriCub_get_nx)( lhs ), NS(TriCub_get_nx)( rhs ) );

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_int64_attribute)(
                    NS(TriCub_get_ny)( lhs ), NS(TriCub_get_ny)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_int64_attribute)(
                    NS(TriCub_get_nz)( lhs ), NS(TriCub_get_nz)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute)(
                    NS(TriCub_get_x0)( lhs ), NS(TriCub_get_x0)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute)(
                    NS(TriCub_get_y0)( lhs ), NS(TriCub_get_y0)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute)(
                    NS(TriCub_get_z0)( lhs ), NS(TriCub_get_z0)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute)(
                    NS(TriCub_get_dx)( lhs ), NS(TriCub_get_dx)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute)(
                    NS(TriCub_get_dy)( lhs ), NS(TriCub_get_dy)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute)(
                    NS(TriCub_get_dz)( lhs ), NS(TriCub_get_dz)( rhs ) );
            }

            if( ( cmp_result == 0 ) && ( lhs_phi != SIXTRL_NULLPTR  ) &&
                ( rhs_phi != SIXTRL_NULLPTR ) )
            {
                int_t const phi_size = NS(TriCub_get_phi_size)( lhs );
                SIXTRL_ASSERT( phi_size == NS(TriCub_get_phi_size)( lhs ) );

                if( ( phi_size > ( int_t )0u ) && ( lhs_phi != rhs_phi ) )
                {
                    NS(be_tricub_int_t) ii = ( int_t )0u;
                    for( ; ii < phi_size ; ++ii )
                    {
                        if( lhs_phi[ ii ] > rhs_phi[ ii ] )
                        {
                            cmp_result = +1;
                            break;
                        }
                        else if( lhs_phi[ ii ] < rhs_phi[ ii ] )
                        {
                            cmp_result = -1;
                            break;
                        }
                    }
                }
            }
            else if( ( cmp_result == 0 ) && ( lhs_phi != SIXTRL_NULLPTR ) )
            {
                SIXTRL_ASSERT( ( rhs_phi == SIXTRL_NULLPTR ) &&
                    ( NS(TriCub_get_phi_size)( rhs ) == ( int_t )0u ) );

                cmp_result = +1;
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
            typedef NS(be_tricub_int_t)  int_t;

            NS(be_tricub_ptr_const_real_t) lhs_phi =
                NS(TriCub_get_ptr_const_phi)( lhs );

            NS(be_tricub_ptr_const_real_t) rhs_phi =
                NS(TriCub_get_ptr_const_phi)( lhs );

            cmp_result = NS(TestLibCompare_int64_attribute)(
                NS(TriCub_get_nx)( lhs ), NS(TriCub_get_nx)( rhs ) );

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_int64_attribute)(
                    NS(TriCub_get_ny)( lhs ), NS(TriCub_get_ny)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_int64_attribute)(
                    NS(TriCub_get_nz)( lhs ), NS(TriCub_get_nz)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                    NS(TriCub_get_x0)( lhs ), NS(TriCub_get_x0)( rhs ),
                    treshold );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                    NS(TriCub_get_y0)( lhs ), NS(TriCub_get_y0)( rhs ),
                    treshold );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                    NS(TriCub_get_z0)( lhs ), NS(TriCub_get_z0)( rhs ),
                    treshold );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                    NS(TriCub_get_dx)( lhs ), NS(TriCub_get_dx)( rhs ),
                    treshold );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                    NS(TriCub_get_dy)( lhs ), NS(TriCub_get_dy)( rhs ),
                    treshold );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(TestLibCompare_real_attribute_with_treshold)(
                    NS(TriCub_get_dz)( lhs ), NS(TriCub_get_dz)( rhs ),
                    treshold );
            }

            if( ( cmp_result == 0 ) && ( lhs_phi != SIXTRL_NULLPTR  ) &&
                ( rhs_phi != SIXTRL_NULLPTR ) )
            {
                NS(be_tricub_int_t) const phi_size =
                    NS(TriCub_get_phi_size)( lhs );

                SIXTRL_ASSERT( phi_size == NS(TriCub_get_phi_size)( lhs ) );

                if( ( phi_size > ( NS(be_tricub_int_t) )0u ) &&
                    ( lhs_phi != rhs_phi ) )
                {
                    typedef NS(be_tricub_real_t) real_t;
                    int_t ii = ( int_t )0u;

                    for( ; ii < phi_size ; ++ii )
                    {
                        real_t const diff     = lhs_phi[ ii ] - rhs_phi[ ii ];
                        real_t const abs_diff = ( diff >= ( real_t )0 )
                            ? diff : -diff;

                        if( abs_diff > treshold )
                        {
                            cmp_result = ( diff > 0 ) ? +1 : -1;
                            break;
                        }
                    }
                }
            }
            else if( ( cmp_result == 0 ) && ( lhs_phi != SIXTRL_NULLPTR ) )
            {
                SIXTRL_ASSERT( ( rhs_phi == SIXTRL_NULLPTR ) &&
                    ( NS(TriCub_get_phi_size)( rhs ) == ( int_t )0u ) );

                cmp_result = +1;
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

    printf( "|tricub          | nx             = %+20ld\r\n"
            "                 | ny             = %+20ld\r\n"
            "                 | nz             = %+20ld\r\n"
            "                 | x0             = %+20.12f\r\n"
            "                 | y0             = %+20.12f\r\n"
            "                 | z0             = %+20.12f\r\n"
            "                 | dx             = %+20.12f\r\n"
            "                 | dy             = %+20.12f\r\n"
            "                 | dz             = %+20.12f\r\n",
            NS(TriCub_get_nx)( e ), NS(TriCub_get_ny)( e ),
            NS(TriCub_get_nz)( e ),
            NS(TriCub_get_x0)( e ), NS(TriCub_get_y0)( e ),
            NS(TriCub_get_z0)( e ),
            NS(TriCub_get_dx)( e ), NS(TriCub_get_dy)( e ),
            NS(TriCub_get_dz)( e ) );

    #else

    NS(TriCub_print)( stdout, e );

    #endif /* !defined( _GPUCODE ) */
}

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_TRICUB_C99_H__ */
