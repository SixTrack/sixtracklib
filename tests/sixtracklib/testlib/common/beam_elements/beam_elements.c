#include "sixtracklib/testlib/common/beam_elements/beam_elements.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/beam_elements.h"

/* ========================================================================= */
/* =====           BEAM_ELEMENTS_TOOLS IMPLEMENTATION                  ===== */
/* ========================================================================= */

void NS(Drift_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( fp != SIXTRL_NULLPTR );
    fprintf( fp, "|drift            | length   = %+16.12f m;\r\n",
            NS(Drift_get_length)( drift ) );
}

void NS(DriftExact_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( fp != SIXTRL_NULLPTR );

    fprintf( fp, "|drift exact      | length   = %+16.12f m;\r\n",
            NS(DriftExact_get_length)( drift ) );
}

void NS(MultiPole_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT mp )
{
    typedef NS(multipole_order_t) mp_order_t;

    mp_order_t const order = NS(MultiPole_get_order)( mp );
    SIXTRL_ASSERT( fp != SIXTRL_NULLPTR );

    fprintf( fp,
             "|multipole        | order    = %3d;\r\n"
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
}

void NS(XYShift_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift )
{
    SIXTRL_ASSERT( fp != SIXTRL_NULLPTR );
    fprintf( fp,
             "|xy_shift         | dx       = %+16.12f m;\r\n"
             "                  | dy       = %+16.12f m;\r\n",
             NS(XYShift_get_dx)( xy_shift ),
             NS(XYShift_get_dy)( xy_shift ) );
}

void NS(SRotation_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT srot )
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
}

void NS(Cavity_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cav )
{
    SIXTRL_ASSERT( fp != SIXTRL_NULLPTR );
    fprintf( fp,
             "|cavity           | voltage   = %+16.12f V   \r\n"
             "                  | frequency = %+20.12f Hz; \r\n"
             "                  | lag       = %+15.12f deg;\r\n",
             NS(Cavity_get_voltage)( cav ),
             NS(Cavity_get_frequency)( cav ),
             NS(Cavity_get_lag)( cav ) );
}

void NS(BeamMonitor_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor )
{
    int const is_rolling =
        NS(BeamMonitor_is_rolling)( monitor ) ? 1 : 0;

    int const is_turn_ordered =
        NS(BeamMonitor_is_turn_ordered)( monitor ) ? 1 : 0;

    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );

    fprintf( fp,
            "|beam-monitor     | num stores       = %20d \r\n"
            "                  | start turn       = %20d\r\n"
            "                  | skip turns       = %20d\r\n"
            "                  | out_address      = %20lu;\r\n"
            "                  | min_particle_id  = %20d\r\n"
            "                  | max_particle_id  = %20d\r\n"
            "                  | is_rolling       = %20d\r\n"
            "                  | is_turn_ordered  = %20d\r\n",
            ( int )NS(BeamMonitor_get_num_stores)( monitor ),
            ( int )NS(BeamMonitor_get_start)( monitor ),
            ( int )NS(BeamMonitor_get_skip)( monitor ),
            ( unsigned long )NS(BeamMonitor_get_out_address)( monitor ),
            ( int )NS(BeamMonitor_get_min_particle_id)( monitor ),
            ( int )NS(BeamMonitor_get_max_particle_id)( monitor ),
            is_rolling, is_turn_ordered );
}

/* ------------------------------------------------------------------------- */

void NS(BeamElement_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const struct NS(Object) *const SIXTRL_RESTRICT
        be_info )
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

                NS(Drift_print)( fp, beam_element );
                break;
            }

            case NS(OBJECT_TYPE_DRIFT_EXACT):
            {
                typedef NS(DriftExact) beam_element_t;
                typedef SIXTRL_DATAPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t beam_element = ( ptr_belem_t )( uintptr_t )addr;

                SIXTRL_ASSERT( sizeof( beam_element_t ) <=
                               NS(Object_get_size)( be_info ) );

                NS(DriftExact_print)( fp, beam_element );
                break;
            }

            case NS(OBJECT_TYPE_MULTIPOLE):
            {
                typedef NS(MultiPole) beam_element_t;
                typedef SIXTRL_DATAPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t beam_element = ( ptr_belem_t )( uintptr_t )addr;

                SIXTRL_ASSERT( sizeof( beam_element_t ) <=
                               NS(Object_get_size)( be_info ) );

                NS(MultiPole_print)( fp, beam_element );
                break;
            }

            case NS(OBJECT_TYPE_CAVITY):
            {
                typedef NS(Cavity) beam_element_t;
                typedef SIXTRL_DATAPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t beam_element = ( ptr_belem_t )( uintptr_t )addr;

                SIXTRL_ASSERT( sizeof( beam_element_t ) <=
                               NS(Object_get_size)( be_info ) );

                NS(Cavity_print)( fp, beam_element );
                break;
            }

            case NS(OBJECT_TYPE_XYSHIFT):
            {
                typedef NS(XYShift) beam_element_t;
                typedef SIXTRL_DATAPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t beam_element = ( ptr_belem_t )( uintptr_t )addr;

                SIXTRL_ASSERT( sizeof( beam_element_t ) <=
                               NS(Object_get_size)( be_info ) );

                NS(XYShift_print)( fp, beam_element );
                break;
            }

            case NS(OBJECT_TYPE_SROTATION):
            {
                typedef NS(SRotation) beam_element_t;
                typedef SIXTRL_DATAPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t beam_element = ( ptr_belem_t )( uintptr_t )addr;

                SIXTRL_ASSERT( sizeof( beam_element_t ) <=
                               NS(Object_get_size)( be_info ) );

                NS(SRotation_print)( fp, beam_element );

                break;
            }

            case NS(OBJECT_TYPE_BEAM_BEAM_4D):
            {
                typedef SIXTRL_DATAPTR_DEC NS(BeamBeam4D) const* ptr_to_belem_t;
                ptr_to_belem_t beam_element = ( ptr_to_belem_t )( uintptr_t )addr;

                NS(BeamBeam4D_print)( fp, beam_element );
                break;
            }

            case NS(OBJECT_TYPE_SC_COASTING):
            {
                typedef SIXTRL_DATAPTR_DEC NS(SpaceChargeCoasting) const*
                        ptr_to_belem_t;

                ptr_to_belem_t beam_element =
                    ( ptr_to_belem_t )( uintptr_t )addr;

                NS(SpaceChargeCoasting_print)( fp, beam_element );
                break;
            }

            case NS(OBJECT_TYPE_SC_QGAUSSIAN_PROF):
            {
                typedef SIXTRL_DATAPTR_DEC NS(SpaceChargeQGaussianProfile) const*
                        ptr_to_belem_t;

                ptr_to_belem_t beam_element =
                    ( ptr_to_belem_t )( uintptr_t )addr;

                NS(SpaceChargeQGaussianProfile_print)( fp, beam_element );
                break;
            }

            case NS(OBJECT_TYPE_BEAM_BEAM_6D):
            {
                typedef SIXTRL_DATAPTR_DEC NS(BeamBeam6D) const* ptr_to_belem_t;
                ptr_to_belem_t beam_element = ( ptr_to_belem_t )( uintptr_t )addr;

                NS(BeamBeam6D_print)( fp, beam_element );
                break;
            }

            case NS(OBJECT_TYPE_BEAM_MONITOR):
            {
                typedef SIXTRL_DATAPTR_DEC NS(BeamMonitor) const* ptr_to_belem_t;
                ptr_to_belem_t beam_element = ( ptr_to_belem_t )( uintptr_t )addr;

                NS(BeamMonitor_print)( fp, beam_element );
                break;
            }

            case NS(OBJECT_TYPE_LIMIT_RECT):
            {
                typedef SIXTRL_DATAPTR_DEC NS(LimitRect) const* ptr_to_belem_t;

                ptr_to_belem_t beam_element =
                    ( ptr_to_belem_t )( uintptr_t )addr;

                NS(LimitRect_print)( fp, beam_element );
                break;
            }

            case NS(OBJECT_TYPE_LIMIT_ELLIPSE):
            {
                typedef SIXTRL_DATAPTR_DEC NS(LimitEllipse) const*
                        ptr_to_belem_t;

                ptr_to_belem_t beam_element =
                    ( ptr_to_belem_t )( uintptr_t )addr;

                NS(LimitEllipse_print)( fp, beam_element );
                break;
            }

            case NS(OBJECT_TYPE_DIPEDGE):
            {
                typedef SIXTRL_DATAPTR_DEC NS(DipoleEdge) const*
                        ptr_to_belem_t;

                ptr_to_belem_t beam_element =
                    ( ptr_to_belem_t )( uintptr_t )addr;

                NS(DipoleEdge_print)( fp, beam_element );
                break;
            }

            default:
            {
                printf( "|unknown          | type_id  = %3d\r\n"
                        "                  | size     = %8lu bytes\r\n"
                        "                  | addr     = %16p\r\n",
                        ( int )type_id, NS(Object_get_size)( be_info ),
                        ( void* )( uintptr_t )addr );
            }
        };
    }

    return;
}

/* end: tests/sixtracklib/testlib/common/beam_elements/beam_elements.c */
