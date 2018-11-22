#include "sixtracklib/testlib/common/beam_elements.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/beam_elements.h"

/* ------------------------------------------------------------------------- */

extern SIXTRL_HOST_FN void NS(Drift_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(Drift) *const SIXTRL_RESTRICT drift );

extern SIXTRL_HOST_FN void NS(DriftExact_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(DriftExact) *const SIXTRL_RESTRICT drift );

extern SIXTRL_HOST_FN void NS(MultiPole_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(MultiPole) *const SIXTRL_RESTRICT mp );

extern SIXTRL_HOST_FN void NS(XYShift_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(XYShift) *const SIXTRL_RESTRICT xy_shift );

extern SIXTRL_HOST_FN void NS(SRotation_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(SRotation) *const SIXTRL_RESTRICT srot );

extern SIXTRL_HOST_FN void NS(Cavity_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(Cavity) *const SIXTRL_RESTRICT cav );

extern SIXTRL_HOST_FN void NS(BeamBeam4D_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(BeamBeam4D) *const SIXTRL_RESTRICT bb4d );

extern SIXTRL_HOST_FN void NS(BeamBeam6D_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(BeamBeam6D) *const SIXTRL_RESTRICT bb6d );

extern SIXTRL_HOST_FN void NS(BeamMonitor_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(BeamMonitor) *const SIXTRL_RESTRICT monitor );


extern SIXTRL_HOST_FN void NS(BeamElement_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(Object) *const SIXTRL_RESTRICT be_info );

/* ------------------------------------------------------------------------- */

extern SIXTRL_HOST_FN void NS(Drift_fprint)( FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift );

extern SIXTRL_HOST_FN void NS(DriftExact_fprint)( FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift );

extern SIXTRL_HOST_FN void NS(MultiPole_fprint)( FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT mp );

extern SIXTRL_HOST_FN void NS(XYShift_fprint)( FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift );

extern SIXTRL_HOST_FN void NS(SRotation_fprint)( FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT srot );

extern SIXTRL_HOST_FN void NS(Cavity_fprint)( FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cav );

extern SIXTRL_HOST_FN void NS(BeamBeam4D_fprint)( FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT bb4d );

extern SIXTRL_HOST_FN void NS(BeamBeam6D_fprint)( FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT bb6d );

extern SIXTRL_HOST_FN void NS(BeamMonitor_fprint)( FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor );


extern SIXTRL_HOST_FN void NS(BeamElement_fprint)( FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC  const NS(Object) *const SIXTRL_RESTRICT be_info );

/* ========================================================================= */
/* =====           BEAM_ELEMENTS_TOOLS IMPLEMENTATION                  ===== */
/* ========================================================================= */

void NS(Drift_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(Drift) *const SIXTRL_RESTRICT drift )
{
    NS(Drift_print)( drift );
    return;
}

void NS(DriftExact_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(DriftExact) *const SIXTRL_RESTRICT drift )
{
    NS(DriftExact_print)( drift );
    return;
}

void NS(MultiPole_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(MultiPole) *const SIXTRL_RESTRICT mp )
{
    NS(MultiPole_print)( mp );
    return;
}

void NS(XYShift_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(XYShift) *const SIXTRL_RESTRICT xy_shift )
{
    NS(XYShift_print)( xy_shift );
    return;
}

void NS(SRotation_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(SRotation) *const SIXTRL_RESTRICT srot )
{
    NS(SRotation_print)( srot );
    return;
}

void NS(Cavity_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(Cavity) *const SIXTRL_RESTRICT cav )
{
    NS(Cavity_print)( cav );
    return;
}

void NS(BeamBeam4D_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(BeamBeam4D) *const SIXTRL_RESTRICT bb4d )
{
    NS(BeamBeam4D_print)( bb4d );
    return;
}

void NS(BeamBeam6D_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(BeamBeam6D) *const SIXTRL_RESTRICT bb6d )
{
    NS(BeamBeam6D_print)( bb6d );
    return;
}

void NS(BeamMonitor_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(BeamMonitor) *const SIXTRL_RESTRICT monitor )
{
    NS(BeamMonitor_print)( monitor );
    return;
}


void NS(BeamElement_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(Object) *const SIXTRL_RESTRICT be_info )
{
    NS(BeamElement_print)( be_info );
    return;
}

/* ========================================================================= */

void NS(Drift_fprint)( FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( fp != SIXTRL_NULLPTR );
    fprintf( fp, "|drift            | length   = %+16.12f m;\r\n",
            NS(Drift_get_length)( drift ) );

    return;
}

/* ------------------------------------------------------------------------- */

void NS(DriftExact_fprint)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( fp != SIXTRL_NULLPTR );

    fprintf( fp, "|drift            | length   = %+16.12f m;\r\n",
            NS(DriftExact_get_length)( drift ) );

    return;
}

void NS(MultiPole_fprint)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT mp )
{
    typedef NS(multipole_order_t) mp_order_t;

    mp_order_t const order = NS(MultiPole_get_order)( mp );
    SIXTRL_ASSERT( fp != SIXTRL_NULLPTR );

    fprintf( fp, "|multipole        | order    = %3d;\r\n"
            "                  | length   = %+16.12f m;\r\n"
            "                  | hxl      = %+16.12f m;\r\n"
            "                  | hyl      = %+16.12f m;\r\n",
            ( int )order, NS(MultiPole_get_length)( mp ),
            NS(MultiPole_get_hxl)( mp ), NS(MultiPole_get_hyl)( mp ) );

    if( order >= ( mp_order_t )0 )
    {
        mp_order_t ii = ( mp_order_t )0;
        mp_order_t const num_k_values = order + ( mp_order_t )1;

        for( ; ii < num_k_values ; ++ii )
        {
            fprintf( fp, "                  |"
                    "knl[ %3d ] = %+20.12f ; ksl[ %3d ] = %+20.12f \r\n",
                    ( int )ii, NS(MultiPole_get_knl_value)( mp, ii ),
                    ( int )ii, NS(MultiPole_get_ksl_value)( mp, ii ) );
        }
    }
    else
    {
        fprintf( fp, "                  | knl = n/a ; ksl = n/a\r\n" );
    }

    return;
}

void NS(XYShift_fprint)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift )
{
    SIXTRL_ASSERT( fp != SIXTRL_NULLPTR );
    fprintf( fp,
             "|xy_shift         | dx       = %+16.12f m;\r\n"
             "                  | dy       = %+16.12f m;\r\n",
             NS(XYShift_get_dx)( xy_shift ),
             NS(XYShift_get_dy)( xy_shift ) );
}

void NS(SRotation_fprint)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT srot )
{
    SIXTRL_ASSERT( fp != SIXTRL_NULLPTR );
    fprintf( fp,
             "|srotation        | angle    = %+16.12f deg  ( %+16.12f rad )\r\n"
             "                  | cos_z    = %+13.12f;\r\n"
             "                  | sin_z    = %+13.12f;\r\n",
             NS(SRotation_get_angle_deg)( srot ),
             NS(SRotation_get_angle)( srot ),
             NS(SRotation_get_cos_angle)( srot ),
             NS(SRotation_get_sin_angle)( srot ) );

    return;
}

void NS(Cavity_fprint)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cav )
{
    SIXTRL_ASSERT( fp != SIXTRL_NULLPTR );
    fprintf( fp,
             "|cavity           | voltage   = %+16.12f V   \r\n"
             "                  | frequency = %+20.12f Hz; \r\n"
             "                  | lag       = %+15.12f deg;\r\n",
             NS(Cavity_get_voltage)( cav ),
             NS(Cavity_get_frequency)( cav ),
             NS(Cavity_get_lag)( cav ) );

    return;
}


void NS(BeamBeam4D_fprint)( FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT bb4d )
{
    typedef NS(beambeam4d_real_const_ptr_t)  bb_data_ptr_t;
    typedef SIXTRL_BE_DATAPTR_DEC BB4D_data* BB4D_data_ptr_t;

    bb_data_ptr_t data = NS(BeamBeam4D_get_const_data)( bb4d );
    BB4D_data_ptr_t bb4ddata = (BB4D_data_ptr_t) data;

    SIXTRL_ASSERT( bb4ddata != SIXTRL_NULLPTR );

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

    return;
}

void NS(BeamBeam6D_fprint)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT bb6d )
{
    typedef SIXTRL_REAL_T                       real_t;
    typedef NS(beambeam6d_real_const_ptr_t)     bb_data_ptr_t;
    typedef SIXTRL_BE_DATAPTR_DEC BB6D_data*    BB6D_data_ptr_t;
    typedef SIXTRL_BE_DATAPTR_DEC real_t const* ptr_real_t;

    bb_data_ptr_t data = NS(BeamBeam6D_get_const_data)( bb6d );
    BB6D_data_ptr_t bb6ddata = (BB6D_data_ptr_t) data;

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
            fprintf( fp,
                    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . "
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
        fprintf( fp, "|beambeam6d      | enabled                = %20ld\r\n",
                 ( long int )0 );
    }

    return;
}

void NS(BeamMonitor_fprint)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor )
{
    int const is_rolling = NS(BeamMonitor_is_rolling)( monitor ) ? 1 : 0;

    int const attr_cont =
        NS(BeamMonitor_are_attributes_continous)( monitor ) ? 1 : 0;

    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );

    fprintf( fp,
            "|beam-monitor     | num stores       = %20d \r\n"
            "                  | start turn       = %20d;\r\n"
            "                  | skip turns       = %20d;\r\n"
            "                  | out_address      = %20lu;\r\n"
            "                  | out store stride = %20lu;\r\n"
            "                  | rolling          = %20d;\r\n"
            "                  | cont attributes  = %20d;\r\n",
            ( int )NS(BeamMonitor_get_num_stores)( monitor ),
            ( int )NS(BeamMonitor_get_start)( monitor ),
            ( int )NS(BeamMonitor_get_skip)( monitor ),
            ( unsigned long )NS(BeamMonitor_get_out_address)( monitor ),
            ( unsigned long )monitor->out_store_stride,
            is_rolling, attr_cont );

    return;
}

/* ------------------------------------------------------------------------- */

void NS(BeamElement_fprint)( FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT be_info )
{
    if( ( fp != SIXTRL_NULLPTR ) && ( be_info != SIXTRL_NULLPTR ) )
    {
        NS(object_type_id_t) const type_id = NS(Object_get_type_id)( be_info );
        NS(buffer_addr_t) const addr = NS(Object_get_begin_addr)( be_info );

        switch( type_id )
        {
            case NS(OBJECT_TYPE_DRIFT):
            {
                typedef NS(Drift) beam_element_t;
                typedef SIXTRL_DATAPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t beam_element = ( ptr_belem_t )( uintptr_t )addr;

                SIXTRL_ASSERT( sizeof( beam_element_t ) <=
                               NS(Object_get_size)( be_info ) );

                NS(Drift_fprint)( fp, beam_element );
                break;
            }

            case NS(OBJECT_TYPE_DRIFT_EXACT):
            {
                typedef NS(DriftExact) beam_element_t;
                typedef SIXTRL_DATAPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t beam_element = ( ptr_belem_t )( uintptr_t )addr;

                SIXTRL_ASSERT( sizeof( beam_element_t ) <=
                               NS(Object_get_size)( be_info ) );

                NS(DriftExact_fprint)( fp, beam_element );
                break;
            }

            case NS(OBJECT_TYPE_MULTIPOLE):
            {
                typedef NS(MultiPole) beam_element_t;
                typedef SIXTRL_DATAPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t beam_element = ( ptr_belem_t )( uintptr_t )addr;

                SIXTRL_ASSERT( sizeof( beam_element_t ) <=
                               NS(Object_get_size)( be_info ) );

                NS(MultiPole_fprint)( fp, beam_element );
                break;
            }

            case NS(OBJECT_TYPE_CAVITY):
            {
                typedef NS(Cavity) beam_element_t;
                typedef SIXTRL_DATAPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t beam_element = ( ptr_belem_t )( uintptr_t )addr;

                SIXTRL_ASSERT( sizeof( beam_element_t ) <=
                               NS(Object_get_size)( be_info ) );

                NS(Cavity_fprint)( fp, beam_element );
                break;
            }

            case NS(OBJECT_TYPE_XYSHIFT):
            {
                typedef NS(XYShift) beam_element_t;
                typedef SIXTRL_DATAPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t beam_element = ( ptr_belem_t )( uintptr_t )addr;

                SIXTRL_ASSERT( sizeof( beam_element_t ) <=
                               NS(Object_get_size)( be_info ) );

                NS(XYShift_fprint)( fp, beam_element );
                break;
            }

            case NS(OBJECT_TYPE_SROTATION):
            {
                typedef NS(SRotation) beam_element_t;
                typedef SIXTRL_DATAPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t beam_element = ( ptr_belem_t )( uintptr_t )addr;

                SIXTRL_ASSERT( sizeof( beam_element_t ) <=
                               NS(Object_get_size)( be_info ) );

                NS(SRotation_fprint)( fp, beam_element );

                break;
            }

            case NS(OBJECT_TYPE_BEAM_BEAM_4D):
            {
                typedef SIXTRL_DATAPTR_DEC NS(BeamBeam4D) const* ptr_to_belem_t;
                ptr_to_belem_t beam_element = ( ptr_to_belem_t )( uintptr_t )addr;

                NS(BeamBeam4D_fprint)( fp, beam_element );
                break;
            }

            case NS(OBJECT_TYPE_BEAM_BEAM_6D):
            {
                typedef SIXTRL_DATAPTR_DEC NS(BeamBeam6D) const* ptr_to_belem_t;
                ptr_to_belem_t beam_element = ( ptr_to_belem_t )( uintptr_t )addr;

                NS(BeamBeam6D_fprint)( fp, beam_element );
                break;
            }

            case NS(OBJECT_TYPE_BEAM_MONITOR):
            {
                typedef SIXTRL_DATAPTR_DEC NS(BeamMonitor) const* ptr_to_belem_t;
                ptr_to_belem_t beam_element = ( ptr_to_belem_t )( uintptr_t )addr;

                NS(BeamMonitor_fprint)( fp, beam_element );
                break;
            }

            default:
            {
                printf( "|unknown          | type_id  = %3d;\r\n"
                        "                  | size     = %8lu bytes;\r\n"
                        "                  | addr     = %16p;\r\n",
                        ( int )type_id, NS(Object_get_size)( be_info ),
                        ( void* )( uintptr_t )addr );
            }
        };
    }

    return;
}

/* end: tests/sixtracklib/testlib/common/test_beam_elements_tools.c */
