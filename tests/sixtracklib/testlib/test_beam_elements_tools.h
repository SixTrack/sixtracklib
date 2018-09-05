#ifndef SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_TEST_BEAM_ELEMENTS_TOOLS_H__
#define SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_TEST_BEAM_ELEMENTS_TOOLS_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdio.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

struct NS(Drift);
struct NS(DriftExact);
struct NS(MultiPole);
struct NS(XYShift);
struct NS(SRotation);
struct NS(Cavity);

struct NS(Object);

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC void NS(Drift_print)(
    SIXTRL_ARGPTR_DEC const struct NS(Drift) *const SIXTRL_RESTRICT drift );

#if !defined( _GPUCODE )
SIXTRL_HOST_FN void NS(Drift_fprint)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const struct NS(Drift) *const SIXTRL_RESTRICT drift );
#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC void NS(DriftExact_print)(
    SIXTRL_ARGPTR_DEC const struct NS(DriftExact) *const SIXTRL_RESTRICT drift );

#if !defined( _GPUCODE )
SIXTRL_HOST_FN void NS(DriftExact_fprint)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const struct NS(DriftExact) *const SIXTRL_RESTRICT drift );
#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC void NS(MultiPole_print)(
    SIXTRL_ARGPTR_DEC const struct NS(MultiPole) *const SIXTRL_RESTRICT mp );

#if !defined( _GPUCODE )
SIXTRL_HOST_FN void NS(MultiPole_fprint)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const struct NS(MultiPole) *const SIXTRL_RESTRICT mp );
#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC void NS(XYShift_print)(
    SIXTRL_ARGPTR_DEC const struct NS(XYShift) *const SIXTRL_RESTRICT xy_shift );

#if !defined( _GPUCODE )
SIXTRL_HOST_FN void NS(XYShift_fprint)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const struct NS(XYShift) *const SIXTRL_RESTRICT xy_shift );
#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC void NS(SRotation_print)(
    SIXTRL_ARGPTR_DEC const struct NS(SRotation) *const SIXTRL_RESTRICT srot );

#if !defined( _GPUCODE )
SIXTRL_HOST_FN void NS(SRotation_fprint)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const struct NS(SRotation) *const SIXTRL_RESTRICT srot );
#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC void NS(Cavity_print)(
    SIXTRL_ARGPTR_DEC const struct NS(Cavity) *const SIXTRL_RESTRICT cav );

#if !defined( _GPUCODE )
SIXTRL_HOST_FN void NS(Cavity_fprint)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const struct NS(Cavity) *const SIXTRL_RESTRICT cav );
#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC void NS(BeamElement_print)(
    SIXTRL_ARGPTR_DEC const struct NS(Object) *const SIXTRL_RESTRICT be_info );

#if !defined( _GPUCODE )
SIXTRL_HOST_FN void NS(BeamElement_fprint)( FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const struct NS(Object) *const SIXTRL_RESTRICT be_info );
#endif /* !defined( _GPUCODE ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= *
 * ======== INLINE IMPLEMENTATION                                            *
 * ========================================================================= */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/beam_elements.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE void NS(Drift_print)(
    SIXTRL_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    printf( "|drift            | length   = %+16.12f m;\r\n",
            NS(Drift_get_length)( drift ) );

    return;
}

SIXTRL_INLINE void NS(DriftExact_print)(
    SIXTRL_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift )
{
    printf( "|drift exact      | length   = %+16.12f m;\r\n",
            NS(DriftExact_get_length)( drift ) );

    return;
}

SIXTRL_INLINE void NS(MultiPole_print)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT mp )
{
    typedef NS(multipole_order_t) mp_order_t;

    mp_order_t const order = NS(MultiPole_get_order)( mp );

    printf( "|multipole        | order    = %3d;\r\n"
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
            printf( "                  |"
                    "knl[ %3d ] = %+20.12f ; ksl[ %3d ] = %+20.12f \r\n",
                    ( int )ii, NS(MultiPole_get_knl_value)( mp, ii ),
                    ( int )ii, NS(MultiPole_get_ksl_value)( mp, ii ) );
        }
    }
    else
    {
        printf( "                  | knl = n/a ; ksl = n/a\r\n" );
    }

    return;
}

SIXTRL_INLINE void NS(XYShift_print)(
    SIXTRL_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift )
{
    printf( "|xy_shift         | dx       = %+16.12f m;\r\n"
            "                  | dy       = %+16.12f m;\r\n",
            NS(XYShift_get_dx)( xy_shift ),
            NS(XYShift_get_dy)( xy_shift ) );

    return;
}

SIXTRL_INLINE void NS(SRotation_print)(
    SIXTRL_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT srot )
{
    printf( "|srotation        | angle    = %+16.12f deg  ( %+16.12f rad )\r\n"
            "                  | cos_z    = %+13.12f;\r\n"
            "                  | sin_z    = %+13.12f;\r\n",
            NS(SRotation_get_angle_deg)( srot ),
            NS(SRotation_get_angle)( srot ),
            NS(SRotation_get_cos_angle)( srot ),
            NS(SRotation_get_sin_angle)( srot ) );

    return;
}

SIXTRL_INLINE void NS(Cavity_print)(
    SIXTRL_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cav )
{
    printf( "|cavity           | voltage   = %+16.12f V   \r\n"
            "                  | frequency = %+20.12f Hz; \r\n"
            "                  | lag       = %+15.12f deg;\r\n",
            NS(Cavity_get_voltage)( cav ),
            NS(Cavity_get_frequency)( cav ),
            NS(Cavity_get_lag)( cav ) );

    return;
}

SIXTRL_INLINE void NS(BeamElement_print)(
    SIXTRL_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT be_info )
{
    if( be_info != SIXTRL_NULLPTR )
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

                NS(Drift_print)( beam_element );

                break;
            }

            case NS(OBJECT_TYPE_DRIFT_EXACT):
            {
                typedef NS(DriftExact) beam_element_t;
                typedef SIXTRL_DATAPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t beam_element = ( ptr_belem_t )( uintptr_t )addr;

                SIXTRL_ASSERT( sizeof( beam_element_t ) <=
                               NS(Object_get_size)( be_info ) );

                NS(DriftExact_print)( beam_element );

                break;
            }

            case NS(OBJECT_TYPE_MULTIPOLE):
            {
                typedef NS(MultiPole) beam_element_t;
                typedef SIXTRL_DATAPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t beam_element = ( ptr_belem_t )( uintptr_t )addr;

                SIXTRL_ASSERT( sizeof( beam_element_t ) <=
                               NS(Object_get_size)( be_info ) );

                NS(MultiPole_print)( beam_element );

                break;
            }

            case NS(OBJECT_TYPE_CAVITY):
            {
                typedef NS(Cavity) beam_element_t;
                typedef SIXTRL_DATAPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t beam_element = ( ptr_belem_t )( uintptr_t )addr;

                SIXTRL_ASSERT( sizeof( beam_element_t ) <=
                               NS(Object_get_size)( be_info ) );

                NS(Cavity_print)( beam_element );

                break;
            }

            case NS(OBJECT_TYPE_XYSHIFT):
            {
                typedef NS(XYShift) beam_element_t;
                typedef SIXTRL_DATAPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t beam_element = ( ptr_belem_t )( uintptr_t )addr;

                SIXTRL_ASSERT( sizeof( beam_element_t ) <=
                               NS(Object_get_size)( be_info ) );

                NS(XYShift_print)( beam_element );

                break;
            }

            case NS(OBJECT_TYPE_SROTATION):
            {
                typedef NS(SRotation) beam_element_t;
                typedef SIXTRL_DATAPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t beam_element = ( ptr_belem_t )( uintptr_t )addr;

                SIXTRL_ASSERT( sizeof( beam_element_t ) <=
                               NS(Object_get_size)( be_info ) );

                NS(SRotation_print)( beam_element );

                break;
            }

            /*
            case NS(BLOCK_TYPE_BEAM_BEAM):
            {
                typedef SIXTRL_DATAPTR_DEC NS(BeamBeam) const* ptr_to_belem_t;
                ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

                for( ; index < index_end ; ++index )
                {
                    ret |= NS(Track_particle_beam_beam)( p, index, belem );
                }

                break;
            }
            */

            default:
            {
                printf( "|unknown          | type_id  = %3d;\r\n"
                        "                  | size     = %8lu bytes;\r\n"
                        "                  | addr     = %16p;\r\n",
                        ( int )type_id, NS(Object_get_size)( be_info ),
                        ( void const* )( uintptr_t )addr );
            }
        };
    }

    return;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_TEST_BEAM_ELEMENTS_TOOLS_H__ */

/* end: tests/sixtracklib/testlib/test_beam_elements_tools.h */
