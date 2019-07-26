#ifndef SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BEAMFIELDS_C99_H__
#define SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BEAMFIELDS_C99_H__

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
    #include "sixtracklib/common/be_beamfields/be_beamfields.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************** */
/* BeamBeam4D: */

SIXTRL_STATIC SIXTRL_FN int NS(BeamBeam4D_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT rhs );

SIXTRL_STATIC SIXTRL_FN int NS(BeamBeam4D_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold );

SIXTRL_STATIC SIXTRL_FN void NS(BeamBeam4D_print_out)(
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamBeam4D) *const SIXTRL_RESTRICT e );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(BeamBeam4D_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamBeam4D) *const SIXTRL_RESTRICT e );

#endif /* !defined( _GPUCODE ) */

/* ************************************************************************** */
/* SpaceChargeCoasting: */

SIXTRL_STATIC SIXTRL_FN int NS(SpaceChargeCoasting_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting)
        *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting)
        *const SIXTRL_RESTRICT rhs );

SIXTRL_STATIC SIXTRL_FN int NS(SpaceChargeCoasting_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting)
        *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting)
        *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold );

SIXTRL_STATIC SIXTRL_FN void NS(SpaceChargeCoasting_print_out)(
    SIXTRL_BE_ARGPTR_DEC const struct NS(SpaceChargeCoasting)
        *const SIXTRL_RESTRICT e );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(SpaceChargeCoasting_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const struct NS(SpaceChargeCoasting)
        *const SIXTRL_RESTRICT e );

#endif /* !defined( _GPUCODE ) */

/* ************************************************************************** */
/* SpaceChargeBunched: */

SIXTRL_STATIC SIXTRL_FN int NS(SpaceChargeBunched_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched)
        *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched)
        *const SIXTRL_RESTRICT rhs );

SIXTRL_STATIC SIXTRL_FN int NS(SpaceChargeBunched_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched)
        *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched)
        *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold );

SIXTRL_STATIC SIXTRL_FN void NS(SpaceChargeBunched_print_out)(
    SIXTRL_BE_ARGPTR_DEC const struct NS(SpaceChargeBunched)
        *const SIXTRL_RESTRICT e );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(SpaceChargeBunched_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const struct NS(SpaceChargeBunched)
        *const SIXTRL_RESTRICT e );

#endif /* !defined( _GPUCODE ) */


/* ************************************************************************** */
/* BeamBeam6D: */

SIXTRL_STATIC SIXTRL_FN int NS(BeamBeam6D_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT rhs );

SIXTRL_STATIC SIXTRL_FN int NS(BeamBeam6D_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold );

SIXTRL_STATIC SIXTRL_FN void NS(BeamBeam6D_print_out)(
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamBeam6D) *const SIXTRL_RESTRICT e );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(BeamBeam6D_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamBeam6D) *const SIXTRL_RESTRICT e );

#endif /* !defined( _GPUCODE ) */

/* ************************************************************************* */
/* Helper functions */

SIXTRL_STATIC SIXTRL_FN int NS(BeamBeam_compare_values_generic)(
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT lhs_data,
    NS(buffer_size_t) const lhs_size,
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT rhs_data,
    NS(buffer_size_t) const rhs_size );

SIXTRL_INLINE int NS(BeamBeam_compare_values_generic_with_treshold)(
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT lhs_data,
    NS(buffer_size_t) const lhs_size,
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT rhs_data,
    NS(buffer_size_t) const rhs_size,
    SIXTRL_REAL_T const treshold );

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====             Implementation of inline functions                ===== */
/* ========================================================================= */

/* ************************************************************************* */
/* Helper functions: */

SIXTRL_INLINE int NS(BeamBeam_compare_values_generic)(
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT lhs_data,
    NS(buffer_size_t) const lhs_size,
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT rhs_data,
    NS(buffer_size_t) const rhs_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    int cmp_value = -1;

    if( ( lhs_data == rhs_data ) && ( lhs_data != SIXTRL_NULLPTR ) &&
        ( lhs_size == rhs_size ) && ( lhs_size > ( buf_size_t )0u ) )
    {
        cmp_value = 0;
    }
    else if( ( lhs_data != SIXTRL_NULLPTR  ) && ( rhs_data != SIXTRL_NULLPTR ) )
    {
        if( lhs_size == rhs_size )
        {
            cmp_value = 0;

            if( ( lhs_size > ( buf_size_t )0u ) && ( lhs_data != rhs_data ) )
            {
                buf_size_t ii = ( buf_size_t )0u;

                for( ; ii < lhs_size ; ++ii )
                {
                    if( lhs_data[ ii ] > rhs_data[ ii ] )
                    {
                        cmp_value = +1;
                        break;
                    }
                    else if( lhs_data[ ii ] < rhs_data[ ii ] )
                    {
                        cmp_value = -1;
                        break;
                    }
                }
            }
        }
        else if( lhs_size > rhs_size )
        {
            cmp_value = +1;
        }
        else if( rhs_size < lhs_size )
        {
            cmp_value = -1;
        }
    }
    else if( lhs_data != SIXTRL_NULLPTR )
    {
        SIXTRL_ASSERT( ( rhs_data == SIXTRL_NULLPTR ) &&
                       ( rhs_size == ( buf_size_t )0u ) );
        cmp_value = +1;
    }

    return cmp_value;
}

SIXTRL_INLINE int NS(BeamBeam_compare_values_generic_with_treshold)(
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT lhs_data,
    NS(buffer_size_t) const lhs_size,
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT rhs_data,
    NS(buffer_size_t) const rhs_size,
    SIXTRL_REAL_T const treshold )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_REAL_T real_t;

    int cmp_value = -1;

    if( ( lhs_data != SIXTRL_NULLPTR  ) && ( rhs_data != SIXTRL_NULLPTR ) )
    {
        if( lhs_size == rhs_size )
        {
            cmp_value = 0;

            if( ( lhs_size > ( buf_size_t )0u ) && ( lhs_data != rhs_data ) )
            {
                buf_size_t ii = ( buf_size_t )0u;

                for( ; ii < lhs_size ; ++ii )
                {
                    real_t const diff     = lhs_data[ ii ] - rhs_data[ ii ];
                    real_t const abs_diff = ( diff >= ( real_t )0 )
                        ? diff : -diff;

                    if( abs_diff > treshold )
                    {
                        cmp_value = ( diff > 0 ) ? +1 : -1;
                        break;
                    }
                }
            }
        }
        else if( lhs_size > rhs_size )
        {
            cmp_value = +1;
        }
        else if( rhs_size < lhs_size )
        {
            cmp_value = -1;
        }
    }
    else if( lhs_data != SIXTRL_NULLPTR )
    {
        SIXTRL_ASSERT( ( rhs_data == SIXTRL_NULLPTR ) &&
                       ( rhs_size == ( buf_size_t )0u ) );
        cmp_value = +1;
    }

    return cmp_value;
}

/* ************************************************************************* */
/* BeamBeam4D: */

SIXTRL_INLINE int NS(BeamBeam4D_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT rhs )
{
    return NS(BeamBeam_compare_values_generic)(
       NS(BeamBeam4D_get_const_data)( lhs ), NS(BeamBeam4D_get_data_size)( lhs ),
       NS(BeamBeam4D_get_const_data)( rhs ), NS(BeamBeam4D_get_data_size)( rhs ) );
}

SIXTRL_INLINE int NS(BeamBeam4D_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold )
{
    return NS(BeamBeam_compare_values_generic_with_treshold)(
       NS(BeamBeam4D_get_const_data)( lhs ), NS(BeamBeam4D_get_data_size)( lhs ),
       NS(BeamBeam4D_get_const_data)( rhs ), NS(BeamBeam4D_get_data_size)( rhs ),
       treshold );
}


SIXTRL_INLINE void NS(BeamBeam4D_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT e )
{
    #if defined( _GPUCODE )

    typedef NS(beambeam4d_real_const_ptr_t)  bb_data_ptr_t;
    typedef SIXTRL_BE_DATAPTR_DEC NS(BB4D_data)* data_ptr_t;

    bb_data_ptr_t data = NS(BeamBeam4D_get_const_data)( e );
    data_ptr_t bb4ddata = ( NS(BB4D_data_ptr_t) )data;

    SIXTRL_ASSERT( bb4ddata != SIXTRL_NULLPTR );

    printf( "|beambeam4d      | q_part         = %+20e\r\n"
            "                 | N_part         = %+20e\r\n"
            "                 | sigma_x        = %+20.12f\r\n"
            "                 | sigma_y        = %+20.12f\r\n"
            "                 | beta_s         = %+20.12f\r\n"
            "                 | min_sigma_diff = %+20.12f\r\n"
            "                 | Delta_x        = %+20.12f\r\n"
            "                 | Delta_y        = %+20.12f\r\n"
            "                 | Dpx_sub        = %+20.12f\r\n"
            "                 | Dpy_sub        = %+20.12f\r\n"
            "                 | enabled        = %20ld\r\n",
            bb4ddata->q_part,  bb4ddata->N_part,  bb4ddata->sigma_x,
            bb4ddata->sigma_y, bb4ddata->beta_s,  bb4ddata->min_sigma_diff,
            bb4ddata->Delta_x, bb4ddata->Delta_y, bb4ddata->Dpx_sub,
            bb4ddata->Dpy_sub, ( long int )bb4ddata->enabled );

    #else

    NS(BeamBeam4D_print)( stdout, e );

    #endif /* !defined( _GPUCODE ) */
}

/* ************************************************************************* */
/* SpaceChargeCoasting: */

SIXTRL_INLINE int NS(SpaceChargeCoasting_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT rhs )
{
    return NS(BeamBeam_compare_values_generic)(
       NS(SpaceChargeCoasting_get_const_data)( lhs ),
       NS(SpaceChargeCoasting_get_data_size)( lhs ),
       NS(SpaceChargeCoasting_get_const_data)( rhs ),
       NS(SpaceChargeCoasting_get_data_size)( rhs ) );
}

SIXTRL_INLINE int NS(SpaceChargeCoasting_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold )
{
    return NS(BeamBeam_compare_values_generic_with_treshold)(
       NS(SpaceChargeCoasting_get_const_data)( lhs ),
       NS(SpaceChargeCoasting_get_data_size)( lhs ),
       NS(SpaceChargeCoasting_get_const_data)( rhs ),
       NS(SpaceChargeCoasting_get_data_size)( rhs ),
       treshold );
}

SIXTRL_INLINE void NS(SpaceChargeCoasting_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT e )
{
    #if defined( _GPUCODE )

    typedef NS(beambeam4d_real_const_ptr_t)  bb_data_ptr_t;
    bb_data_ptr_t data = NS(SpaceChargeCoasting_get_const_data)( e );

    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );

    printf( "|sc coasting     | \r\n" );
    ( void )data;

    #else

    NS(SpaceChargeCoasting_print)( stdout, e );

    #endif /* !defined( _GPUCODE ) */
}

/* ************************************************************************* */
/* SpaceChargeBunched: */

SIXTRL_INLINE int NS(SpaceChargeBunched_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const SIXTRL_RESTRICT rhs )
{
    return NS(BeamBeam_compare_values_generic)(
       NS(SpaceChargeBunched_get_const_data)( lhs ),
       NS(SpaceChargeBunched_get_data_size)( lhs ),
       NS(SpaceChargeBunched_get_const_data)( rhs ),
       NS(SpaceChargeBunched_get_data_size)( rhs ) );
}

SIXTRL_INLINE int NS(SpaceChargeBunched_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold )
{
    return NS(BeamBeam_compare_values_generic_with_treshold)(
       NS(SpaceChargeBunched_get_const_data)( lhs ),
       NS(SpaceChargeBunched_get_data_size)( lhs ),
       NS(SpaceChargeBunched_get_const_data)( rhs ),
       NS(SpaceChargeBunched_get_data_size)( rhs ),
       treshold );
}

SIXTRL_INLINE void NS(SpaceChargeBunched_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const SIXTRL_RESTRICT e )
{
    #if defined( _GPUCODE )

    typedef NS(beambeam4d_real_const_ptr_t)  bb_data_ptr_t;
    bb_data_ptr_t data = NS(SpaceChargeBunched_get_const_data)( e );

    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );

    printf( "|sc bunched      | \r\n" );
    ( void )data;

    #else

    NS(SpaceChargeBunched_print)( stdout, e );

    #endif /* !defined( _GPUCODE ) */
}

/* ************************************************************************* */
/* BeamBeam6D: */

SIXTRL_INLINE int NS(BeamBeam6D_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT rhs )
{
    return NS(BeamBeam_compare_values_generic)(
       NS(BeamBeam6D_get_const_data)( lhs ), NS(BeamBeam6D_get_data_size)( lhs ),
       NS(BeamBeam6D_get_const_data)( rhs ), NS(BeamBeam6D_get_data_size)( rhs ) );
}

SIXTRL_INLINE int NS(BeamBeam6D_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold )
{
    return NS(BeamBeam_compare_values_generic_with_treshold)(
       NS(BeamBeam6D_get_const_data)( lhs ), NS(BeamBeam6D_get_data_size)( lhs ),
       NS(BeamBeam6D_get_const_data)( rhs ), NS(BeamBeam6D_get_data_size)( rhs ),
       treshold );
}


SIXTRL_INLINE void NS(BeamBeam6D_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT elem )
{
    #if defined( _GPUCODE )

    typedef SIXTRL_REAL_T                           real_t;
    typedef SIXTRL_BE_DATAPTR_DEC real_t const*     ptr_real_t;
    typedef NS(beambeam6d_real_const_ptr_t)         bb_data_ptr_t;
    typedef SIXTRL_BE_DATAPTR_DEC NS(BB6D_data)*    data_ptr_t;

    data_ptr_t data = NS(BeamBeam6D_get_const_data)( elem );
    NS(BB6D_data_ptr_t) bb6ddata = ( NS(BB6D_data_ptr_t) )data;

    if( ( bb6ddata != SIXTRL_NULLPTR ) && ( bb6ddata->enabled ) )
    {
        int num_slices = (int)(bb6ddata->N_slices);
        int ii = 0;

        ptr_real_t N_part_per_slice =
            SIXTRL_BB_GET_PTR(bb6ddata, N_part_per_slice);

        ptr_real_t x_slices_star =
            SIXTRL_BB_GET_PTR(bb6ddata, x_slices_star);

        ptr_real_t y_slices_star =
            SIXTRL_BB_GET_PTR(bb6ddata, y_slices_star);

        ptr_real_t sigma_slices_star =
            SIXTRL_BB_GET_PTR(bb6ddata, sigma_slices_star);

        SIXTRL_ASSERT( N_part_per_slice  != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( x_slices_star     != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( y_slices_star     != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( sigma_slices_star != SIXTRL_NULLPTR );

        printf( "|beambeam6d      | enabled                = %20ld\r\n"
                "                 | sphi                   = %+20e\r\n"
                "                 | calpha                 = %+20e\r\n"
                "                 | S33                    = %+20.12f\r\n"
                "                 | N_slices               = %+20d\r\n",
                ( long int )bb6ddata->enabled,
                (bb6ddata->parboost).sphi, (bb6ddata->parboost).calpha,
                (bb6ddata->Sigmas_0_star).Sig_33_0, num_slices );

        for( ; ii < num_slices ; ++ii )
        {
            printf( ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . "
                    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . \r\n"
                    "                 | N_part_per_slice[%4d]  = %20e\r\n"
                    "                 | x_slices_star[%4d]     = %20.12f\r\n"
                    "                 | y_slices_star[%4d]     = %20.12f\r\n"
                    "                 | sigma_slices_star[%4d] = %20.12f\r\n",
                    ii, N_part_per_slice[ ii ],
                    ii, x_slices_star[ ii ],
                    ii, y_slices_star[ ii ],
                    ii, sigma_slices_star[ ii ] );
        }
    }
    else
    {
        printf( "|beambeam6d      | enabled                = %20ld\r\n",
                ( long int )0 );
    }

    #else

    NS(BeamBeam6D_print)( stdout, elem );

    #endif /* !defined( _GPUCODE ) */
}

#endif /* SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BEAMFIELDS_C99_H__ */

/* end: tests/sixtracklib/testlib/common/be_beamfields/be_beamfields.h */
