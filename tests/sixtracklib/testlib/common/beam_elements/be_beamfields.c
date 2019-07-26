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
    typedef NS(beambeam4d_real_const_ptr_t)  bb_data_ptr_t;
    typedef SIXTRL_BE_DATAPTR_DEC NS(BB4D_data)* data_ptr_t;

    bb_data_ptr_t data = NS(BeamBeam4D_get_const_data)( e );
    data_ptr_t bb4ddata = ( data_ptr_t )data;

    if( ( fp != SIXTRL_NULLPTR ) &&
        ( bb4ddata != SIXTRL_NULLPTR ) )
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
                 bb4ddata->q_part,  bb4ddata->N_part,  bb4ddata->sigma_x,
                 bb4ddata->sigma_y, bb4ddata->beta_s,  bb4ddata->min_sigma_diff,
                 bb4ddata->Delta_x, bb4ddata->Delta_y, bb4ddata->Dpx_sub,
                 bb4ddata->Dpy_sub, ( long int )bb4ddata->enabled );
    }

    return;
}

/* ************************************************************************* */
/* SpaceChargeCoasting:  */

void NS(SpaceChargeCoasting_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT e )
{
    typedef NS(beambeam4d_real_const_ptr_t)  sc_data_ptr_t;
    sc_data_ptr_t data =
        ( sc_data_ptr_t )NS(SpaceChargeCoasting_get_const_data)( e );

    if( ( data != SIXTRL_NULLPTR ) &&
        ( fp != SIXTRL_NULLPTR ) )
    {
        fprintf( fp, "|sc coasting     | \r\n" );
    }

    return;
}


/* ************************************************************************* */
/* SpaceChargeBunched:  */

void NS(SpaceChargeBunched_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const SIXTRL_RESTRICT e )
{
    typedef NS(beambeam4d_real_const_ptr_t)  sc_data_ptr_t;
    sc_data_ptr_t data = NS(SpaceChargeBunched_get_const_data)( e );

    if( ( data != SIXTRL_NULLPTR ) &&
        ( fp != SIXTRL_NULLPTR ) )
    {
        fprintf( fp, "|sc bunched      | \r\n" );
    }

    return;
}


/* ************************************************************************* */
/* BeamBeam6D:  */

void NS(BeamBeam6D_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT elem )
{
    typedef SIXTRL_REAL_T                               real_t;
    typedef SIXTRL_BE_DATAPTR_DEC real_t const*         ptr_real_t;
    typedef SIXTRL_BE_DATAPTR_DEC NS(BB6D_data) const*  data_ptr_t;

    data_ptr_t bb6ddata = ( data_ptr_t )NS(BeamBeam6D_get_const_data)( elem );

    if( ( bb6ddata != SIXTRL_NULLPTR ) && ( bb6ddata->enabled ) &&
        ( fp != SIXTRL_NULLPTR ) )
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

/*end: tests/sixtracklib/testlib/common/be_beamfields/be_beamfields.c */
