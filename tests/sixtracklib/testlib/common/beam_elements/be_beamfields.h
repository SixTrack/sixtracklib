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
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const
        SIXTRL_RESTRICT rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN int NS(BeamBeam4D_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(BeamBeam4D_print_out)(
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamBeam4D) *const
        SIXTRL_RESTRICT e );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(BeamBeam4D_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamBeam4D) *const SIXTRL_RESTRICT e );

#endif /* !defined( _GPUCODE ) */

/* ************************************************************************** */
/* BeamBeam6D: */

SIXTRL_STATIC SIXTRL_FN int NS(BeamBeam6D_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const
        SIXTRL_RESTRICT rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN int NS(BeamBeam6D_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(BeamBeam6D_print_out)(
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamBeam6D) *const SIXTRL_RESTRICT e );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(BeamBeam6D_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamBeam6D) *const SIXTRL_RESTRICT e );

#endif /* !defined( _GPUCODE ) */

/* ************************************************************************** */
/* SCCoasting: */

SIXTRL_STATIC SIXTRL_FN int NS(SCCoasting_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN int NS(SCCoasting_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(SCCoasting_print_out)(
    SIXTRL_BE_ARGPTR_DEC const struct NS(SCCoasting)
        *const SIXTRL_RESTRICT e );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(SCCoasting_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT e );

#endif /* !defined( _GPUCODE ) */

/* ************************************************************************** */
/* SCQGaussProfile: */

SIXTRL_STATIC SIXTRL_FN int NS(SCQGaussProfile_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN int
NS(SCQGaussProfile_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(SCQGaussProfile_print_out)(
    SIXTRL_BE_ARGPTR_DEC const struct NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT e );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(SCQGaussProfile_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT e );

#endif /* !defined( _GPUCODE ) */

/* ************************************************************************** */
/* LineDensityProfileData: */

SIXTRL_STATIC SIXTRL_FN int NS(LineDensityProfileData_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN int
NS(LineDensityProfileData_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT rhs, SIXTRL_REAL_T const treshold ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(LineDensityProfileData_print_out)(
    SIXTRL_BE_ARGPTR_DEC const struct NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT e );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(LineDensityProfileData_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT e );

#endif /* !defined( _GPUCODE ) */


/* ************************************************************************** */
/* SCInterpolatedProfile: */

SIXTRL_STATIC SIXTRL_FN int NS(SCInterpolatedProfile_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCInterpolatedProfile) *const
        SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SCInterpolatedProfile) *const
        SIXTRL_RESTRICT rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN int
NS(SCInterpolatedProfile_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCInterpolatedProfile) *const
        SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SCInterpolatedProfile) *const
        SIXTRL_RESTRICT rhs, SIXTRL_REAL_T const treshold ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(SCInterpolatedProfile_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCInterpolatedProfile) *const
        SIXTRL_RESTRICT e );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(SCInterpolatedProfile_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(SCInterpolatedProfile) *const
        SIXTRL_RESTRICT e );

#endif /* !defined( _GPUCODE ) */


/* ************************************************************************* */
/* Helper functions */

SIXTRL_STATIC SIXTRL_FN int NS(BeamBeam_compare_values_generic)(
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT lhs_data,
    NS(buffer_size_t) const lhs_size,
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT rhs_data,
    NS(buffer_size_t) const rhs_size ) SIXTRL_NOEXCEPT;

SIXTRL_INLINE int NS(BeamBeam_compare_values_generic_with_treshold)(
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT lhs_data,
    NS(buffer_size_t) const lhs_size,
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT rhs_data,
    NS(buffer_size_t) const rhs_size,
    SIXTRL_REAL_T const treshold ) SIXTRL_NOEXCEPT;

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
    NS(buffer_size_t) const rhs_size ) SIXTRL_NOEXCEPT
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
    SIXTRL_REAL_T const treshold ) SIXTRL_NOEXCEPT
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
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const
        SIXTRL_RESTRICT rhs ) SIXTRL_NOEXCEPT
{
    return NS(BeamBeam_compare_values_generic)(
       ( SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* )( uintptr_t
           )NS(BeamBeam4D_data_addr)( lhs ), NS(BeamBeam4D_data_size)( lhs ),
       ( SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* )( uintptr_t
           )NS(BeamBeam4D_data_addr)( rhs ), NS(BeamBeam4D_data_size)( rhs ) );
}

SIXTRL_INLINE int NS(BeamBeam4D_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold ) SIXTRL_NOEXCEPT
{
    return NS(BeamBeam_compare_values_generic_with_treshold)(
       ( SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* )( uintptr_t
           )NS(BeamBeam4D_data_addr)( lhs ), NS(BeamBeam4D_data_size)( lhs ),
       ( SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* )( uintptr_t
           )NS(BeamBeam4D_data_addr)( rhs ), NS(BeamBeam4D_data_size)( rhs ),
       treshold );
}

SIXTRL_INLINE void NS(BeamBeam4D_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT e )
{
    #if defined( _GPUCODE )

    typedef NS(beambeam4d_real_const_ptr_t)  bb_data_ptr_t;
    typedef SIXTRL_BE_DATAPTR_DEC NS(BB4D_data)* data_ptr_t;

    bb_data_ptr_t data = NS(BeamBeam4D_const_data)( e );
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
/* BeamBeam6D: */

SIXTRL_INLINE int NS(BeamBeam6D_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const
        SIXTRL_RESTRICT rhs ) SIXTRL_NOEXCEPT
{
    return NS(BeamBeam_compare_values_generic)(
       ( SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* )( uintptr_t
           )NS(BeamBeam6D_data_addr)( lhs ), NS(BeamBeam6D_data_size)( lhs ),
       ( SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* )( uintptr_t
           )NS(BeamBeam6D_data_addr)( rhs ), NS(BeamBeam6D_data_size)( rhs ) );
}

SIXTRL_INLINE int NS(BeamBeam6D_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold ) SIXTRL_NOEXCEPT
{
    return NS(BeamBeam_compare_values_generic_with_treshold)(
       ( SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* )( uintptr_t
           )NS(BeamBeam6D_data_addr)( lhs ), NS(BeamBeam6D_data_size)( lhs ),
       ( SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* )( uintptr_t
           )NS(BeamBeam6D_data_addr)( rhs ), NS(BeamBeam6D_data_size)( rhs ),
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

    data_ptr_t data = NS(BeamBeam6D_const_data)( elem );
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

/* ************************************************************************* */
/* SCCoasting: */

SIXTRL_INLINE int NS(SCCoasting_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting) *const
        SIXTRL_RESTRICT rhs ) SIXTRL_NOEXCEPT
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;

        if( lhs != rhs )
        {
            cmp_result = NS(Type_value_comp_result)(
                lhs->number_of_particles, rhs->number_of_particles );

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    lhs->circumference, rhs->circumference );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    lhs->sigma_x, rhs->sigma_x );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    lhs->sigma_y, rhs->sigma_y );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    lhs->length, rhs->length );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)( lhs->x_co, rhs->x_co );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(lhs->y_co, rhs->y_co );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    lhs->min_sigma_diff, rhs->min_sigma_diff );
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

SIXTRL_INLINE int NS(SCCoasting_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const ABS_TOL ) SIXTRL_NOEXCEPT
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;

        if( lhs != rhs )
        {
            SIXTRL_REAL_T const REL_TOL = ( SIXTRL_REAL_T )0;
            cmp_result = NS(Type_value_comp_result_with_tolerances)(
                lhs->number_of_particles, rhs->number_of_particles, REL_TOL, ABS_TOL );

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->circumference, rhs->circumference, REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->sigma_x, rhs->sigma_x, REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->sigma_y, rhs->sigma_y, REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->length, rhs->length, REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->x_co, rhs->x_co, REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->y_co, rhs->y_co, REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->min_sigma_diff, rhs->min_sigma_diff, REL_TOL, ABS_TOL );
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

SIXTRL_INLINE void NS(SCCoasting_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting) *const SIXTRL_RESTRICT e )
{
    #if defined( _GPUCODE )

    if( != SIXTRL_NULLPTR )
    {
        printf( "|sc coasting     | number_of_particles = %+20.12f\r\n"
                "                 | circumference       = %+20.12f\r\n"
                "                 | sigma_x             = %+20.12f\r\n"
                "                 | sigma_y             = %+20.12f\r\n"
                "                 | length              = %+20.12f\r\n"
                "                 | x_co                = %+20.12f\r\n"
                "                 | y_co                = %+20.12f\r\n"
                "                 | min_sigma_diff      = %+20.12f\r\n"
                "                 | enabled             = %20lu\r\n",
            e->number_of_particles, e->circumference, e->sigma_x, e->sigma_y,
            e->length, e->x_co, e->y_co, e->min_sigma_diff,
            ( unsigned long )e->enabled );
    }

    #else

    NS(SCCoasting_print)( stdout, e );

    #endif /* !defined( _GPUCODE ) */
}

/* ************************************************************************* */
/* SCQGaussProfile: */

SIXTRL_INLINE int NS(SCQGaussProfile_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile) *const
        SIXTRL_RESTRICT rhs ) SIXTRL_NOEXCEPT
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;

        if( lhs != rhs )
        {
            cmp_result = NS(Type_value_comp_result)(
                lhs->number_of_particles, rhs->number_of_particles );

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    lhs->bunchlength_rms, rhs->bunchlength_rms );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    lhs->sigma_x, rhs->sigma_x );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    lhs->sigma_y, rhs->sigma_y );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    lhs->length, rhs->length );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)( lhs->x_co, rhs->x_co );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(lhs->y_co, rhs->y_co );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    lhs->q_param, rhs->q_param );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)( lhs->cq, rhs->cq );
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

SIXTRL_INLINE int NS(SCQGaussProfile_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile) *const
        SIXTRL_RESTRICT rhs, SIXTRL_REAL_T const ABS_TOL ) SIXTRL_NOEXCEPT
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;

        if( lhs != rhs )
        {
            SIXTRL_REAL_T const REL_TOL = ( SIXTRL_REAL_T )0;
            cmp_result = NS(Type_value_comp_result_with_tolerances)(
                lhs->number_of_particles, rhs->number_of_particles, REL_TOL, ABS_TOL );

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->bunchlength_rms, rhs->bunchlength_rms,
                        REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->sigma_x, rhs->sigma_x, REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->sigma_y, rhs->sigma_y, REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->length, rhs->length, REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->x_co, rhs->x_co, REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->y_co, rhs->y_co, REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->min_sigma_diff, rhs->min_sigma_diff, REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->q_param, rhs->q_param, REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->cq, rhs->cq, REL_TOL, ABS_TOL );
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

SIXTRL_INLINE void NS(SCQGaussProfile_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT e )
{
    #if defined( _GPUCODE )

    if( ( fp != SIXTRL_NULLPTR ) && ( e != SIXTRL_NULLPTR ) )
    {
        printf( "|sc q-gaussian   | number_of_particles = %+20.12f\r\n"
                "                 | circumference       = %+20.12f\r\n"
                "                 | sigma_x             = %+20.12f\r\n"
                "                 | sigma_y             = %+20.12f\r\n"
                "                 | length              = %+20.12f\r\n"
                "                 | x_co                = %+20.12f\r\n"
                "                 | y_co                = %+20.12f\r\n"
                "                 | min_sigma_diff      = %+20.12f\r\n"
                "                 | q_param             = %+20.12f\r\n"
                "                 | b_param             = %+20.12f\r\n"
                "                 | enabled             = %20lu\r\n",
            e->number_of_particles, e->bunchlength_rms, e->sigma_x, e->sigma_y,
            e->length, e->x_co, e->y_co, e->min_sigma_diff, e->q_param,
            e->b_param, ( unsigned long )e->enabled );
    }

    #else

    NS(SCQGaussProfile_print)( stdout, e );

    #endif /* !defined( _GPUCODE ) */
}

/* ************************************************************************* */
/* NS(LineDensityProfileData): */

SIXTRL_INLINE int NS(LineDensityProfileData_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT rhs ) SIXTRL_NOEXCEPT
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;

        if( lhs != rhs )
        {
            cmp_result = ( lhs->method == rhs->method )
                ? 0 : ( ( lhs->method > rhs->method ) ? +1 : -1 );

            if( ( cmp_result == 0 ) && ( lhs->num_values != rhs->num_values ) )
            {
                cmp_result = ( lhs->num_values > rhs->num_values ) ? +1 : -1;
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_for_range)(
                    NS(LineDensityProfileData_const_values_begin)( lhs ),
                    NS(LineDensityProfileData_const_values_end)( lhs ),
                    NS(LineDensityProfileData_const_values_begin)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_for_range)(
                    NS(LineDensityProfileData_const_derivatives_begin)( lhs ),
                    NS(LineDensityProfileData_const_derivatives_end)( lhs ),
                    NS(LineDensityProfileData_const_derivatives_begin)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)( lhs->z0, rhs->z0 );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)( lhs->dz, rhs->dz );
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

SIXTRL_INLINE int NS(LineDensityProfileData_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT rhs, SIXTRL_REAL_T const ABS_TOL ) SIXTRL_NOEXCEPT
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;

        if( lhs != rhs )
        {
            SIXTRL_REAL_T const REL_TOL = ( SIXTRL_REAL_T )0;
            cmp_result = ( lhs->method == rhs->method )
                ? 0 : ( ( lhs->method > rhs->method ) ? +1 : -1 );

            if( ( cmp_result == 0 ) && ( lhs->num_values != rhs->num_values ) )
            {
                cmp_result = ( lhs->num_values > rhs->num_values ) ? +1 : -1;
            }

            if( cmp_result == 0 )
            {
                cmp_result =
                NS(Type_value_comp_result_with_tolerances_for_range)(
                    NS(LineDensityProfileData_const_values_begin)( lhs ),
                    NS(LineDensityProfileData_const_values_end)( lhs ),
                    NS(LineDensityProfileData_const_values_begin)( rhs ),
                    REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result =
                NS(Type_value_comp_result_with_tolerances_for_range)(
                    NS(LineDensityProfileData_const_derivatives_begin)( lhs ),
                    NS(LineDensityProfileData_const_derivatives_end)( lhs ),
                    NS(LineDensityProfileData_const_derivatives_begin)( rhs ),
                    REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->z0, rhs->z0, REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->dz, rhs->dz, REL_TOL, ABS_TOL );
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

SIXTRL_INLINE void NS(LineDensityProfileData_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT e )
{
    NS(math_abscissa_idx_t) const nvalues =
        NS(LineDensityProfileData_num_values)( e );

    printf( "|lprof density data  | method            = " );

    switch( NS(LineDensityProfileData_method)( e ) )
    {
        case NS(MATH_INTERPOL_LINEAR):
        {
            printf( "linear" );
            break;
        }

        case NS(MATH_INTERPOL_CUBIC):
        {
            printf( "cubic" );
            break;
        }

        default:
        {
            printf( "n/a (unknown)" );
        }
    };

    printf( "\r\n"
            "                     | num_values        = %21ld\r\n"
            "                     | values_addr       = %21lx\r\n"
            "                     | derivatives_addr  = %21lx\r\n"
            "                     | z0                = %+20.12f\r\n"
            "                     | dz                = %+20.12f\r\n"
            "                     | capacity          = %21ld\r\n",
            ( long int )NS(LineDensityProfileData_num_values)( e ),
            ( uintptr_t )NS(LineDensityProfileData_values_addr)( e ),
            ( uintptr_t )NS(LineDensityProfileData_derivatives_addr)( e ),
            NS(LineDensityProfileData_z0)( e ),
            NS(LineDensityProfileData_dz)( e ),
            ( long int )NS(LineDensityProfileData_capacity)( e ) );

    if( nvalues > ( NS(math_abscissa_idx_t ) )0 )
    {
        printf( "                     |" " ii"
                "                     z"
                "                 value"
                "      derivative value\r\n" );
    }

    if( nvalues <= ( NS(math_abscissa_idx_t) )10 )
    {
        NS(math_abscissa_idx_t) ii = ( NS(math_abscissa_idx_t) )0;
        SIXTRL_REAL_T z = NS(LineDensityProfileData_z0)( e );

        for( ; ii < nvalues ; ++ii )
        {
            printf( "                     |%3ld"
                    "%+20.12f""%+20.12f""%+20.12f\r\n",
                    ( long int )ii, z,
                    NS(LineDensityProfileData_value_at_idx)( e, ii ),
                    NS(LineDensityProfileData_derivatives_at_idx)( e, ii ) );

            z += NS(LineDensityProfileData_dz)( e );
        }
    }
    else
    {
        NS(math_abscissa_idx_t) ii = ( NS(math_abscissa_idx_t) )0;
        SIXTRL_REAL_T z = NS(LineDensityProfileData_z0)( e );

        for( ; ii < 5 ; ++ii )
        {
            printf( "                     |%3ld"
                    "%+20.12f""%+20.12f""%+20.12f\r\n", ( long int )ii, z,
                    NS(LineDensityProfileData_value_at_idx)( e, ii ),
                    NS(LineDensityProfileData_derivatives_at_idx)( e, ii ) );

            z += NS(LineDensityProfileData_dz)( e );
        }

        printf( "                         | .. ..................... "
                "..................... .....................\r\n" );

        ii = nvalues - 6;
        z = NS(LineDensityProfileData_z0)( e ) +
            NS(LineDensityProfileData_dz)( e ) * ii;

        for( ; ii < 5 ; ++ii )
        {
            printf( "                     |%3ld"
                    "%+20.12f""%+20.12f""%+20.12f\r\n", ( long int )ii, z,
                    NS(LineDensityProfileData_value_at_idx)( e, ii ),
                    NS(LineDensityProfileData_derivatives_at_idx)( e, ii ) );

            z += NS(LineDensityProfileData_dz)( e );
        }
    }
}

/* ************************************************************************** */
/* SCInterpolatedProfile: */

SIXTRL_INLINE int NS(SCInterpolatedProfile_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCInterpolatedProfile) *const
        SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SCInterpolatedProfile) *const
        SIXTRL_RESTRICT rhs ) SIXTRL_NOEXCEPT
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;

        if( lhs != rhs )
        {
            cmp_result = NS(Type_value_comp_result)(
                lhs->number_of_particles, rhs->number_of_particles );

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    lhs->sigma_x, rhs->sigma_x );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    lhs->sigma_y, rhs->sigma_y );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    lhs->length, rhs->length );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)( lhs->x_co, rhs->x_co );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)( lhs->y_co, rhs->y_co );
            }

            if( ( cmp_result == 0 ) &&
                ( lhs->interpol_data_addr != rhs->interpol_data_addr ) )
            {
                cmp_result = NS(LineDensityProfileData_compare_values)(
                    NS(SCInterpolatedProfile_const_line_density_profile_data)( lhs ),
                    NS(SCInterpolatedProfile_const_line_density_profile_data)( rhs ) );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    lhs->line_density_prof_fallback,
                    rhs->line_density_prof_fallback );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result)(
                    lhs->min_sigma_diff, rhs->min_sigma_diff );
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

SIXTRL_INLINE int NS(SCInterpolatedProfile_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCInterpolatedProfile) *const
        SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(SCInterpolatedProfile) *const
        SIXTRL_RESTRICT rhs, SIXTRL_REAL_T const ABS_TOL ) SIXTRL_NOEXCEPT
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;

        if( lhs != rhs )
        {
            SIXTRL_REAL_T const REL_TOL = ( SIXTRL_REAL_T )0;
            cmp_result = NS(Type_value_comp_result_with_tolerances)(
                lhs->number_of_particles, rhs->number_of_particles, REL_TOL, ABS_TOL );

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->sigma_x, rhs->sigma_x, REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->sigma_y, rhs->sigma_y, REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->length, rhs->length, REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->x_co, rhs->x_co, REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->y_co, rhs->y_co, REL_TOL, ABS_TOL );
            }

            if( ( cmp_result == 0 ) &&
                ( lhs->interpol_data_addr != rhs->interpol_data_addr ) )
            {
                cmp_result = NS(LineDensityProfileData_compare_values_with_treshold)(
                    NS(SCInterpolatedProfile_const_line_density_profile_data)( lhs ),
                    NS(SCInterpolatedProfile_const_line_density_profile_data)(
                        rhs ), ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->line_density_prof_fallback,
                    rhs->line_density_prof_fallback, REL_TOL, ABS_TOL );
            }

            if( cmp_result == 0 )
            {
                cmp_result = NS(Type_value_comp_result_with_tolerances)(
                    lhs->min_sigma_diff, rhs->min_sigma_diff, REL_TOL, ABS_TOL );
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

SIXTRL_INLINE void NS(SCInterpolatedProfile_print_out)(
    SIXTRL_BE_ARGPTR_DEC const struct NS(SCInterpolatedProfile) *const
        SIXTRL_RESTRICT e )
{
    SIXTRL_ASSERT( e != SIXTRL_NULLPTR );
    printf( "|sc interpolated | number_of_particles   = %+20.12f\r\n"
            "                 | sigma_x               = %+20.12f\r\n"
            "                 | sigma_y               = %+20.12f\r\n"
            "                 | length                = %+20.12f\r\n"
            "                 | x_co                  = %+20.12f\r\n"
            "                 | y_co                  = %+20.12f\r\n"
            "                 | interpol_data_addr    = %21lx\r\n"
            "                 | line_density_fallback = %+20.12f\r\n"
            "                 | min_sigma_diff        = %+20.12f\r\n"
            "                 | enabled               = %20lu\r\n",
            NS(SCInterpolatedProfile_number_of_particles)( e ),
            NS(SCInterpolatedProfile_sigma_x)( e ),
            NS(SCInterpolatedProfile_sigma_y)( e ),
            NS(SCInterpolatedProfile_length)( e ),
            NS(SCInterpolatedProfile_x_co)( e ),
            NS(SCInterpolatedProfile_y_co)( e ),
            ( uintptr_t )NS(SCInterpolatedProfile_interpol_data_addr)(
                e ),
            NS(SCInterpolatedProfile_line_density_prof_fallback)( e ),
            NS(SCInterpolatedProfile_min_sigma_diff)( e ),
            ( unsigned long )e->enabled );
}

#endif /* SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BEAMFIELDS_C99_H__ */
