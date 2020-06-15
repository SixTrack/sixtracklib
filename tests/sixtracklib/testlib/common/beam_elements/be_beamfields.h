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
    #include "sixtracklib/common/internal/type_comparison_helpers.hpp"
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
/* SpaceChargeQGaussianProfile: */

SIXTRL_STATIC SIXTRL_FN int NS(SpaceChargeQGaussianProfile_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeQGaussianProfile)
        *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeQGaussianProfile)
        *const SIXTRL_RESTRICT rhs );

SIXTRL_STATIC SIXTRL_FN int
NS(SpaceChargeQGaussianProfile_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeQGaussianProfile)
        *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeQGaussianProfile)
        *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold );

SIXTRL_STATIC SIXTRL_FN void NS(SpaceChargeQGaussianProfile_print_out)(
    SIXTRL_BE_ARGPTR_DEC const struct NS(SpaceChargeQGaussianProfile)
        *const SIXTRL_RESTRICT e );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(SpaceChargeQGaussianProfile_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const struct NS(SpaceChargeQGaussianProfile)
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
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;

        if( lhs != rhs )
        {
            cmp_result = ( NS(Type_comp_all_equal)(
                lhs->num_particles, rhs->num_particles ) )
                ? 0 : ( NS(Type_comp_all_more)(
                    lhs->num_particles, rhs->num_particles ) ) ? +1 : -1;

            if( cmp_result == 0 )
            {
                cmp_result = ( NS(Type_comp_all_equal)(
                    lhs->circumference, rhs->circumference ) )
                    ? 0 : ( NS(Type_comp_all_more)(
                        lhs->circumference, rhs->circumference ) ) ? +1 : -1;
            }

            if( cmp_result == 0 )
            {
                cmp_result = ( NS(Type_comp_all_equal)(
                    lhs->sigma_x, rhs->sigma_x ) )
                    ? 0 : ( NS(Type_comp_all_more)(
                        lhs->sigma_x, rhs->sigma_x ) ) ? +1 : -1;
            }

            if( cmp_result == 0 )
            {
                cmp_result = ( NS(Type_comp_all_equal)(
                    lhs->sigma_y, rhs->sigma_y ) )
                    ? 0 : ( NS(Type_comp_all_more)(
                        lhs->sigma_y, rhs->sigma_y ) ) ? +1 : -1;
            }

            if( cmp_result == 0 )
            {
                cmp_result = ( NS(Type_comp_all_equal)(
                    lhs->length, rhs->length ) )
                    ? 0 : ( NS(Type_comp_all_more)(
                        lhs->length, rhs->length ) ) ? +1 : -1;
            }

            if( cmp_result == 0 )
            {
                cmp_result = ( NS(Type_comp_all_equal)(
                    lhs->x_co, rhs->x_co ) )
                    ? 0 : ( NS(Type_comp_all_more)(
                        lhs->x_co, rhs->x_co ) ) ? +1 : -1;
            }

            if( cmp_result == 0 )
            {
                cmp_result = ( NS(Type_comp_all_equal)(
                    lhs->y_co, rhs->y_co ) )
                    ? 0 : ( NS(Type_comp_all_more)(
                        lhs->y_co, rhs->y_co ) ) ? +1 : -1;
            }

            if( cmp_result == 0 )
            {
                cmp_result = ( NS(Type_comp_all_equal)(
                    lhs->min_sigma_diff, rhs->min_sigma_diff ) )
                    ? 0 : ( NS(Type_comp_all_more)(
                        lhs->min_sigma_diff, rhs->min_sigma_diff ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) &&
                ( lhs->enabled != rhs->enabled ) )
            {
                cmp_result = ( lhs->enabled > rhs->enabled ) ? +1 : -1;
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

SIXTRL_INLINE int NS(SpaceChargeCoasting_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold )
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;

        if( lhs != rhs )
        {
            cmp_result = ( NS(Type_comp_all_are_close)(
                lhs->num_particles, rhs->num_particles, 0.0, treshold ) )
                ? 0 : ( NS(Type_comp_all_more)(
                    lhs->num_particles, rhs->num_particles ) ) ? +1 : -1;

            if( ( cmp_result == 0 ) &&
                ( !NS(Type_comp_all_are_close)(
                    lhs->circumference, rhs->circumference, 0.0, treshold ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->circumference,
                   rhs->circumference ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) &&
                ( !NS(Type_comp_all_are_close)(
                    lhs->sigma_x, rhs->sigma_x, 0.0, treshold ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->sigma_x,
                   rhs->sigma_x ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) &&
                ( !NS(Type_comp_all_are_close)(
                    lhs->sigma_y, rhs->sigma_y, 0.0, treshold ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->sigma_y,
                   rhs->sigma_y ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) &&
                ( !NS(Type_comp_all_are_close)(
                    lhs->length, rhs->length, 0.0, treshold ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->length,
                   rhs->length ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) &&
                ( !NS(Type_comp_all_are_close)(
                    lhs->x_co, rhs->x_co, 0.0, treshold ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->x_co,
                   rhs->x_co ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) &&
                ( !NS(Type_comp_all_are_close)(
                    lhs->y_co, rhs->y_co, 0.0, treshold ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->y_co,
                   rhs->y_co ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) &&
                ( !NS(Type_comp_all_are_close)(
                    lhs->min_sigma_diff, rhs->min_sigma_diff, 0.0, treshold ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->min_sigma_diff,
                   rhs->min_sigma_diff ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) && ( lhs->enabled != rhs->enabled ) )
            {
                cmp_result = ( lhs->enabled > rhs->enabled ) ? +1 : -1;
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

SIXTRL_INLINE void NS(SpaceChargeCoasting_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT e )
{
    #if defined( _GPUCODE )

    if( != SIXTRL_NULLPTR )
    {
        printf( "|sc coasting     | num_particles     = %+20.12f\r\n"
                "                 | circumference     = %+20.12f\r\n"
                "                 | sigma_x           = %+20.12f\r\n"
                "                 | sigma_y           = %+20.12f\r\n"
                "                 | length            = %+20.12f\r\n"
                "                 | x_co              = %+20.12f\r\n"
                "                 | y_co              = %+20.12f\r\n"
                "                 | min_sigma_diff    = %+20.12f\r\n"
                "                 | enabled           = %20lu\r\n",
            e->num_particles, e->circumference, e->sigma_x, e->sigma_y,
            e->length, e->x_co, e->y_co, e->min_sigma_diff,
            ( unsigned long )e->enabled );
    }

    #else

    NS(SpaceChargeCoasting_print)( stdout, e );

    #endif /* !defined( _GPUCODE ) */
}

/* ************************************************************************* */
/* SpaceChargeQGaussianProfile: */

SIXTRL_INLINE int NS(SpaceChargeQGaussianProfile_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeQGaussianProfile) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeQGaussianProfile) *const SIXTRL_RESTRICT rhs )
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;

        if( lhs != rhs )
        {
            cmp_result = ( NS(Type_comp_all_equal)(
                lhs->num_particles, rhs->num_particles ) )
                ? 0 : ( NS(Type_comp_all_more)(
                    lhs->num_particles, rhs->num_particles ) ) ? +1 : -1;

            if( ( cmp_result == 0 ) && ( NS(Type_comp_all_equal)(
                    lhs->bunchlength_rms, rhs->bunchlength_rms ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->bunchlength_rms,
                    rhs->bunchlength_rms ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) && ( NS(Type_comp_all_equal)(
                    lhs->sigma_x, rhs->sigma_x ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->sigma_x,
                    rhs->sigma_x ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) && ( NS(Type_comp_all_equal)(
                    lhs->sigma_y, rhs->sigma_y ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->sigma_y,
                    rhs->sigma_y ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) && ( NS(Type_comp_all_equal)(
                    lhs->length, rhs->length ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->length,
                    rhs->length ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) && ( NS(Type_comp_all_equal)(
                    lhs->x_co, rhs->x_co ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->x_co,
                    rhs->x_co ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) && ( NS(Type_comp_all_equal)(
                    lhs->y_co, rhs->y_co ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->y_co,
                    rhs->y_co ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) && ( NS(Type_comp_all_equal)(
                    lhs->min_sigma_diff, rhs->min_sigma_diff ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->min_sigma_diff,
                    rhs->min_sigma_diff ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) && ( NS(Type_comp_all_equal)(
                    lhs->q_param, rhs->q_param ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->q_param,
                    rhs->q_param ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) && ( NS(Type_comp_all_equal)(
                    lhs->b_param, rhs->b_param ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->b_param,
                    rhs->b_param ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) &&
                ( lhs->enabled != rhs->enabled ) )
            {
                cmp_result = ( lhs->enabled > rhs->enabled ) ? +1 : -1;
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

SIXTRL_INLINE int NS(SpaceChargeQGaussianProfile_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeQGaussianProfile) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeQGaussianProfile) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold )
{
     int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;

        if( lhs != rhs )
        {
            cmp_result = ( NS(Type_comp_all_are_close)( lhs->num_particles,
                rhs->num_particles, 0.0, treshold ) )
                ? 0 : ( NS(Type_comp_all_more)(
                    lhs->num_particles, rhs->num_particles ) ) ? +1 : -1;

            if( ( cmp_result == 0 ) && ( NS(Type_comp_all_are_close)(
                    lhs->bunchlength_rms,
                    rhs->bunchlength_rms, 0.0, treshold ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->bunchlength_rms,
                    rhs->bunchlength_rms ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) && ( NS(Type_comp_all_are_close)(
                    lhs->sigma_x, rhs->sigma_x, 0.0, treshold ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->sigma_x,
                    rhs->sigma_x ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) && ( NS(Type_comp_all_are_close)(
                    lhs->sigma_y, rhs->sigma_y, 0.0, treshold ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->sigma_y,
                    rhs->sigma_y ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) && ( NS(Type_comp_all_are_close)(
                    lhs->length, rhs->length, 0.0, treshold ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->length,
                    rhs->length ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) && ( NS(Type_comp_all_are_close)(
                    lhs->x_co, rhs->x_co, 0.0, treshold ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->x_co,
                    rhs->x_co ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) && ( NS(Type_comp_all_are_close)(
                    lhs->y_co, rhs->y_co, 0.0, treshold ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->y_co,
                    rhs->y_co ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) && ( NS(Type_comp_all_are_close)(
                    lhs->min_sigma_diff,
                    rhs->min_sigma_diff, 0.0, treshold ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->min_sigma_diff,
                    rhs->min_sigma_diff ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) && ( NS(Type_comp_all_are_close)(
                    lhs->q_param, rhs->q_param, 0.0, treshold ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->q_param,
                    rhs->q_param ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) && ( NS(Type_comp_all_are_close)(
                    lhs->b_param, rhs->b_param, 0.0, treshold ) ) )
            {
                cmp_result = ( NS(Type_comp_all_more)( lhs->b_param,
                    rhs->b_param ) ) ? +1 : -1;
            }

            if( ( cmp_result == 0 ) &&
                ( lhs->enabled != rhs->enabled ) )
            {
                cmp_result = ( lhs->enabled > rhs->enabled ) ? +1 : -1;
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

SIXTRL_INLINE void NS(SpaceChargeQGaussianProfile_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeQGaussianProfile) *const SIXTRL_RESTRICT e )
{
    #if defined( _GPUCODE )

    if( ( fp != SIXTRL_NULLPTR ) && ( e != SIXTRL_NULLPTR ) )
    {
        printf( "|sc q-gaussian   | num_particles     = %+20.12f\r\n"
                "                 | circumference     = %+20.12f\r\n"
                "                 | sigma_x           = %+20.12f\r\n"
                "                 | sigma_y           = %+20.12f\r\n"
                "                 | length            = %+20.12f\r\n"
                "                 | x_co              = %+20.12f\r\n"
                "                 | y_co              = %+20.12f\r\n"
                "                 | min_sigma_diff    = %+20.12f\r\n"
                "                 | q_param           = %+20.12f\r\n"
                "                 | b_param           = %+20.12f\r\n"
                "                 | enabled           = %20lu\r\n",
            e->num_particles, e->bunchlength_rms, e->sigma_x, e->sigma_y,
            e->length, e->x_co, e->y_co, e->min_sigma_diff, e->q_param,
            e->b_param, ( unsigned long )e->enabled );
    }

    #else

    NS(SpaceChargeQGaussianProfile_print)( stdout, e );

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
