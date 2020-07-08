#include "sixtracklib/testlib/common/beam_elements/be_beamfields.h"

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "sixtracklib/common/definitions.h"

/* ************************************************************************* */
/* BeamBeam4D:  */

void NS(BeamBeam4D_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT e )
{
    SIXTRL_BE_DATAPTR_DEC NS(BB4D_data) const* bb4data =
        NS(BeamBeam4D_const_data)( e );

    if( ( fp != SIXTRL_NULLPTR ) && ( bb4data != SIXTRL_NULLPTR ) )
    {
        fprintf( fp,
                 "|beambeam4d      | q_part         = %+20e\r\n"
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
                 bb4data->q_part,  bb4data->N_part,  bb4data->sigma_x,
                 bb4data->sigma_y, bb4data->beta_s,  bb4data->min_sigma_diff,
                 bb4data->Delta_x, bb4data->Delta_y, bb4data->Dpx_sub,
                 bb4data->Dpy_sub, ( long int )bb4data->enabled );
    }

    return;
}

/* ************************************************************************* */
/* BeamBeam6D:  */

void NS(BeamBeam6D_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT elem )
{
    SIXTRL_BE_DATAPTR_DEC NS(BB6D_data) const* bb6ddata =
        NS(BeamBeam6D_const_data)( elem );

    if( ( bb6ddata != SIXTRL_NULLPTR ) && ( bb6ddata->enabled ) &&
        ( fp != SIXTRL_NULLPTR ) )
    {
        typedef SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const* ptr_real_t;

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

        fprintf( fp,
                "|beambeam6d      | enabled                = %20ld\r\n"
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

    return;
}

/* ************************************************************************* */
/* SpaceChargeCoasting:  */

void NS(SpaceChargeCoasting_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT e )
{
    if( ( fp != SIXTRL_NULLPTR ) && ( e != SIXTRL_NULLPTR ) )
    {
        fprintf( fp,
                 "|sc coasting     | number_of_particles = %+20.12f\r\n"
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

    return;
}

/* ************************************************************************* */
/* SpaceChargeQGaussianProfile:  */

void NS(SpaceChargeQGaussianProfile_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const
        NS(SpaceChargeQGaussianProfile) *const SIXTRL_RESTRICT e )
{
    if( ( fp != SIXTRL_NULLPTR ) && ( e != SIXTRL_NULLPTR ) )
    {
        fprintf( fp,
                 "|sc q-gaussian   | number_of_particles = %+20.12f\r\n"
                 "                 | circumference       = %+20.12f\r\n"
                 "                 | sigma_x             = %+20.12f\r\n"
                 "                 | sigma_y             = %+20.12f\r\n"
                 "                 | length              = %+20.12f\r\n"
                 "                 | x_co                = %+20.12f\r\n"
                 "                 | y_co                = %+20.12f\r\n"
                 "                 | min_sigma_diff      = %+20.12f\r\n"
                 "                 | q_param             = %+20.12f\r\n"
                 "                 | cq                  = %+20.12f\r\n"
                 "                 | enabled             = %20lu\r\n",
            e->number_of_particles, e->bunchlength_rms, e->sigma_x, e->sigma_y,
            e->length, e->x_co, e->y_co, e->min_sigma_diff, e->q_param,
            e->cq, ( unsigned long )e->enabled );
    }

    return;
}

/* ************************************************************************* */
/* SpaceChargeInterpolatedProfile:  */

void NS(LineDensityProfileData_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BUFFER_DATAPTR_DEC const
        NS(LineDensityProfileData) *const SIXTRL_RESTRICT e )
{
    if( ( fp != SIXTRL_NULLPTR ) && ( e != SIXTRL_NULLPTR ) )
    {
        NS(math_abscissa_idx_t) const nvalues =
            NS(LineDensityProfileData_num_values)( e );

        fprintf( fp, "|lprof density data  | method            = " );

        switch( NS(LineDensityProfileData_method)( e ) )
        {
            case NS(MATH_INTERPOL_LINEAR):
            {
                fprintf( fp, "linear" );
                break;
            }

            case NS(MATH_INTERPOL_CUBIC):
            {
                fprintf( fp, "cubic" );
                break;
            }

            default:
            {
                fprintf( fp, "n/a (unknown)" );
            }
        };

        fprintf( fp, "\r\n"
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
            fprintf( fp, "                     |" " ii"
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
                fprintf( fp, "                     |%3ld"
                        "%+20.12f""%+20.12f""%+20.12f\r\n", ( long int )ii, z,
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
                fprintf( fp, "                     |%3ld"
                        "%+20.12f""%+20.12f""%+20.12f\r\n", ( long int )ii, z,
                        NS(LineDensityProfileData_value_at_idx)( e, ii ),
                        NS(LineDensityProfileData_derivatives_at_idx)( e, ii ) );

                z += NS(LineDensityProfileData_dz)( e );
            }

            fprintf( fp, "                         | .. ..................... "
                    "..................... .....................\r\n" );

            ii = nvalues - 6;
            z = NS(LineDensityProfileData_z0)( e ) +
                NS(LineDensityProfileData_dz)( e ) * ii;

            for( ; ii < 5 ; ++ii )
            {
                fprintf( fp, "                     |%3ld"
                        "%+20.12f""%+20.12f""%+20.12f\r\n", ( long int )ii, z,
                        NS(LineDensityProfileData_value_at_idx)( e, ii ),
                        NS(LineDensityProfileData_derivatives_at_idx)( e, ii ) );

                z += NS(LineDensityProfileData_dz)( e );
            }
        }
    }
}

/* ************************************************************************* */
/* SpaceChargeInterpolatedProfile:  */

void NS(SpaceChargeInterpolatedProfile_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const
        NS(SpaceChargeInterpolatedProfile) *const SIXTRL_RESTRICT e )
{
    if( ( fp != SIXTRL_NULLPTR ) && ( e != SIXTRL_NULLPTR ) )
    {
        fprintf( fp,
                 "|sc interpolated | number_of_particles   = %+20.12f\r\n"
                 "                 | sigma_x               = %+20.12f\r\n"
                 "                 | sigma_y               = %+20.12f\r\n"
                 "                 | length                = %+20.12f\r\n"
                 "                 | x_co                  = %+20.12f\r\n"
                 "                 | y_co                  = %+20.12f\r\n"
                 "                 | interpol_data_addr    = %21lu\r\n"
                 "                 | line_density_fallback = %+20.12f\r\n"
                 "                 | min_sigma_diff        = %+20.12f\r\n"
                 "                 | enabled               = %20lu\r\n",
            e->number_of_particles, e->sigma_x, e->sigma_y, e->length, e->x_co,
            e->y_co, ( unsigned long )e->interpol_data_addr,
            e->line_density_prof_fallback, e->min_sigma_diff,
            ( unsigned long )e->enabled );
    }
}
