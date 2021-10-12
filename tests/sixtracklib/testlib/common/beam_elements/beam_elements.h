#ifndef SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_HEADER_H__
#define SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_HEADER_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdio.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

struct NS(XYShift);
struct NS(Object);

SIXTRL_STATIC SIXTRL_FN void NS(BeamElement_print_out)(
    SIXTRL_BE_ARGPTR_DEC const struct NS(Object) *const SIXTRL_RESTRICT be_info );

SIXTRL_STATIC SIXTRL_FN int NS(BeamElements_compare_objects)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT rhs );

SIXTRL_STATIC SIXTRL_FN int NS(BeamElements_compare_objects_with_treshold)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold );

SIXTRL_STATIC SIXTRL_FN int NS(BeamElements_compare_lines)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT rhs_begin );

SIXTRL_STATIC SIXTRL_FN int NS(BeamElements_compare_lines_with_treshold)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT rhs_begin,
    SIXTRL_REAL_T const treshold );

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(BeamElement_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const struct NS(Object) *const SIXTRL_RESTRICT info );

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

    #include "sixtracklib/testlib/common/beam_elements/be_beamfields.h"
    #include "sixtracklib/testlib/common/beam_elements/be_cavity.h"
    #include "sixtracklib/testlib/common/beam_elements/be_dipedge.h"
    #include "sixtracklib/testlib/common/beam_elements/be_drift.h"
    #include "sixtracklib/testlib/common/beam_elements/be_limit_rect.h"
    #include "sixtracklib/testlib/common/beam_elements/be_limit_ellipse.h"
    #include "sixtracklib/testlib/common/beam_elements/be_limit_rect_ellipse.h"
    #include "sixtracklib/testlib/common/beam_elements/be_monitor.h"
    #include "sixtracklib/testlib/common/beam_elements/be_multipole.h"
    #include "sixtracklib/testlib/common/beam_elements/be_rfmultipole.h"
    #include "sixtracklib/testlib/common/beam_elements/be_srotation.h"
    #include "sixtracklib/testlib/common/beam_elements/be_tricub.h"
    #include "sixtracklib/testlib/common/beam_elements/be_xy_shift.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE void NS(BeamElement_print_out)(
    SIXTRL_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT be_info )
{
    #if !defined( _GPUCODE )

    if( be_info != SIXTRL_NULLPTR )
    {
        NS(object_type_id_t) const type_id = NS(Object_get_type_id)( be_info );
        NS(buffer_addr_t) const addr = NS(Object_get_begin_addr)( be_info );

        switch( type_id )
        {
            case NS(OBJECT_TYPE_DRIFT):
            {
                typedef SIXTRL_DATAPTR_DEC NS(Drift) const* ptr_elem_t;
                ptr_elem_t beam_element = ( ptr_elem_t )( uintptr_t )addr;
                NS(Drift_print_out)( beam_element );
                break;
            }

            case NS(OBJECT_TYPE_DRIFT_EXACT):
            {
                typedef SIXTRL_DATAPTR_DEC NS(DriftExact) const* ptr_elem_t;
                ptr_elem_t beam_element = ( ptr_elem_t )( uintptr_t )addr;
                NS(DriftExact_print_out)( beam_element );
                break;
            }

            case NS(OBJECT_TYPE_MULTIPOLE):
            {
                typedef SIXTRL_DATAPTR_DEC NS(Multipole) const* ptr_elem_t;
                ptr_elem_t beam_element = ( ptr_elem_t )( uintptr_t )addr;
                NS(Multipole_print_out)( beam_element );
                break;
            }

            case NS(OBJECT_TYPE_RF_MULTIPOLE):
            {
                typedef SIXTRL_DATAPTR_DEC NS(RFMultipole) const* ptr_elem_t;
                ptr_elem_t beam_element = ( ptr_elem_t )( uintptr_t )addr;
                NS(RFMultipole_print_out)( beam_element );
                break;
            }

            case NS(OBJECT_TYPE_CAVITY):
            {
                typedef SIXTRL_DATAPTR_DEC NS(Cavity) const* ptr_elem_t;
                ptr_elem_t beam_element = ( ptr_elem_t )( uintptr_t )addr;
                NS(Cavity_print_out)( beam_element );
                break;
            }

            case NS(OBJECT_TYPE_XYSHIFT):
            {
                typedef SIXTRL_DATAPTR_DEC NS(XYShift) const* ptr_elem_t;
                ptr_elem_t beam_element = ( ptr_elem_t )( uintptr_t )addr;
                NS(XYShift_print_out)( beam_element );
                break;
            }

            case NS(OBJECT_TYPE_SROTATION):
            {
                typedef SIXTRL_DATAPTR_DEC NS(SRotation) const* ptr_elem_t;
                ptr_elem_t beam_element = ( ptr_elem_t )( uintptr_t )addr;
                NS(SRotation_print_out)( beam_element );
                break;
            }

            case NS(OBJECT_TYPE_BEAM_BEAM_4D):
            {
                typedef SIXTRL_DATAPTR_DEC NS(BeamBeam4D) const* ptr_elem_t;
                ptr_elem_t beam_element = ( ptr_elem_t )( uintptr_t )addr;
                NS(BeamBeam4D_print_out)( beam_element );
                break;
            }

            case NS(OBJECT_TYPE_BEAM_BEAM_6D):
            {
                typedef SIXTRL_DATAPTR_DEC NS(BeamBeam6D) const* ptr_elem_t;
                ptr_elem_t beam_element = ( ptr_elem_t )( uintptr_t )addr;
                NS(BeamBeam6D_print_out)( beam_element );
                break;
            }

            case NS(OBJECT_TYPE_SC_COASTING):
            {
                typedef NS(SCCoasting) beam_element_t;
                typedef SIXTRL_DATAPTR_DEC beam_element_t const* ptr_elem_t;
                ptr_elem_t beam_element = ( ptr_elem_t )( uintptr_t )addr;
                NS(SCCoasting_print_out)( beam_element );
                break;
            }

            case NS(OBJECT_TYPE_SC_QGAUSSIAN_PROF):
            {
                typedef NS(SCQGaussProfile) beam_element_t;
                typedef SIXTRL_DATAPTR_DEC beam_element_t const* ptr_elem_t;
                ptr_elem_t beam_element = ( ptr_elem_t )( uintptr_t )addr;
                NS(SCQGaussProfile_print_out)( beam_element );
                break;
            }

            case NS(OBJECT_TYPE_SC_INTERPOLATED_PROF):
            {
                typedef NS(SCInterpolatedProfile) beam_element_t;
                typedef SIXTRL_DATAPTR_DEC beam_element_t const* ptr_elem_t;
                ptr_elem_t beam_element = ( ptr_elem_t )( uintptr_t )addr;
                NS(SCInterpolatedProfile_print_out)( beam_element );
                break;
            }

            case NS(OBJECT_TYPE_BEAM_MONITOR):
            {
                typedef SIXTRL_DATAPTR_DEC NS(BeamMonitor) const* ptr_elem_t;
                ptr_elem_t beam_element = ( ptr_elem_t )( uintptr_t )addr;
                NS(BeamMonitor_print_out)( beam_element );
                break;
            }

            case NS(OBJECT_TYPE_LIMIT_RECT):
            {
                typedef SIXTRL_DATAPTR_DEC NS(LimitRect) const* ptr_elem_t;
                ptr_elem_t elem = ( ptr_elem_t )( uintptr_t )addr;
                NS(LimitRect_print_out)( elem );
                break;
            }

            case NS(OBJECT_TYPE_LIMIT_ELLIPSE):
            {
                typedef SIXTRL_DATAPTR_DEC NS(LimitEllipse) const* ptr_elem_t;
                ptr_elem_t elem = ( ptr_elem_t )( uintptr_t )addr;
                NS(LimitEllipse_print_out)( elem );
                break;
            }

            case NS(OBJECT_TYPE_LIMIT_RECT_ELLIPSE):
            {
                typedef SIXTRL_DATAPTR_DEC NS(LimitRectEllipse) const* ptr_t;
                ptr_t elem = ( ptr_t )( uintptr_t )addr;
                NS(LimitRectEllipse_print_out)( elem );
                break;
            }

            case NS(OBJECT_TYPE_DIPEDGE):
            {
                typedef SIXTRL_DATAPTR_DEC NS(DipoleEdge) const* ptr_elem_t;
                ptr_elem_t beam_element = ( ptr_elem_t )( uintptr_t )addr;
                NS(DipoleEdge_print_out)( beam_element );
                break;
            }

            case NS(OBJECT_TYPE_TRICUB):
            {
                typedef SIXTRL_DATAPTR_DEC NS(TriCub) const*
                    ptr_to_belem_t;

                ptr_to_belem_t beam_element =
                    ( ptr_to_belem_t )( uintptr_t )addr;

                NS(TriCub_print_out)( beam_element );
                break;
            }

            default:
            {
                printf( "|unknown          | type_id  = %3d;\r\n"
                        "                  | size     = %8lu bytes;\r\n"
                        "                  | addr     = %16p;\r\n",
                        ( int )type_id,
                        ( long unsigned )NS(Object_get_size)( be_info ),
                        ( void const* )( uintptr_t )addr );
            }
        };
    }

    #else

    NS(BeamElement_print( stdout, beam_element );

    #endif /* NS(BeamElement_print( stdout, beam_element ); */

    return;
}


SIXTRL_INLINE int NS(BeamElements_compare_objects)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT rhs )
{
    int compare_value = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        typedef NS(buffer_addr_t)       address_t;
        typedef NS(object_type_id_t)    type_id_t;

        type_id_t const lhs_type_id = NS(Object_get_type_id)( lhs );

        type_id_t const rhs_type_id = NS(Object_get_type_id)( rhs );

        if( lhs_type_id == rhs_type_id )
        {
            address_t const lhs_addr = NS(Object_get_begin_addr)( lhs );
            address_t const rhs_addr = NS(Object_get_begin_addr)( rhs );

            if( ( lhs_addr != ( address_t)0u ) &&
                ( rhs_addr != ( address_t)0u ) && ( lhs_addr != rhs_addr ) )
            {
                switch( lhs_type_id )
                {
                    case NS(OBJECT_TYPE_DRIFT):
                    {
                        typedef NS(Drift)                           belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(Drift_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_DRIFT_EXACT):
                    {
                        typedef NS(DriftExact)                       belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const*  ptr_belem_t;

                        compare_value = NS(DriftExact_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_MULTIPOLE):
                    {
                        typedef NS(Multipole)                       belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(Multipole_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_RF_MULTIPOLE):
                    {
                        typedef NS(RFMultipole)                     belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(RFMultipole_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_XYSHIFT):
                    {
                        typedef NS(XYShift)                          belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const*  ptr_belem_t;

                        compare_value = NS(XYShift_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_SROTATION):
                    {
                        typedef NS(SRotation)                        belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const*  ptr_belem_t;

                        compare_value = NS(SRotation_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_CAVITY):
                    {
                        typedef NS(Cavity)                           belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const*  ptr_belem_t;

                        compare_value = NS(Cavity_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_BEAM_BEAM_4D):
                    {
                        typedef NS(BeamBeam4D)                      belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(BeamBeam4D_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_BEAM_BEAM_6D):
                    {
                        typedef NS(BeamBeam6D)                      belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(BeamBeam6D_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_SC_COASTING):
                    {
                        typedef NS(SCCoasting) belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(SCCoasting_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_SC_QGAUSSIAN_PROF):
                    {
                        typedef NS(SCQGaussProfile) belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(SCQGaussProfile_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_SC_INTERPOLATED_PROF):
                    {
                        typedef NS(SCInterpolatedProfile) belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(SCInterpolatedProfile_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_BEAM_MONITOR):
                    {
                        typedef NS(BeamMonitor) belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const*  ptr_belem_t;

                        compare_value = NS(BeamMonitor_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_LIMIT_RECT):
                    {
                        typedef NS(LimitRect) belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const*  ptr_belem_t;

                        compare_value = NS(LimitRect_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_LIMIT_ELLIPSE):
                    {
                        typedef NS(LimitEllipse) belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const*  ptr_belem_t;

                        compare_value = NS(LimitEllipse_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_DIPEDGE):
                    {
                        typedef NS(DipoleEdge) belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const*  ptr_belem_t;

                        compare_value = NS(DipoleEdge_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_TRICUB):
                    {
                        typedef NS(TriCub) belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const*  ptr_belem_t;

                        compare_value = NS(TriCub_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    default:
                    {
                        compare_value = -1;
                    }
                };
            }
            else if( lhs_addr != ( address_t )0u )
            {
                compare_value = ( rhs_addr != lhs_addr ) ? +1 : 0;
            }
            else if( rhs_addr != ( address_t )0u )
            {
                compare_value = ( rhs_addr != lhs_addr ) ? -1 : 0;
            }
        }
        else if( lhs_type_id > rhs_type_id )
        {
            compare_value = +1;
        }
    }

    return compare_value;
}

SIXTRL_INLINE int NS(BeamElements_compare_objects_with_treshold)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold )
{
    int compare_value = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        typedef NS(buffer_addr_t)       address_t;
        typedef NS(object_type_id_t)    type_id_t;

        type_id_t const lhs_type_id = NS(Object_get_type_id)( lhs );

        type_id_t const rhs_type_id = NS(Object_get_type_id)( rhs );

        if( lhs_type_id == rhs_type_id )
        {
            address_t const lhs_addr = NS(Object_get_begin_addr)( lhs );
            address_t const rhs_addr = NS(Object_get_begin_addr)( rhs );

            if( ( lhs_addr != ( address_t)0u ) &&
                ( rhs_addr != ( address_t)0u ) && ( lhs_addr != rhs_addr ) )
            {
                switch( lhs_type_id )
                {
                    case NS(OBJECT_TYPE_DRIFT):
                    {
                        typedef NS(Drift)                            belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const*  ptr_belem_t;

                        compare_value = NS(Drift_compare_values_with_treshold)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr, treshold );

                        break;
                    }

                    case NS(OBJECT_TYPE_DRIFT_EXACT):
                    {
                        typedef NS(DriftExact)                      belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(DriftExact_compare_values_with_treshold)(
                            ( ptr_belem_t )( uintptr_t )rhs_addr,
                            ( ptr_belem_t )( uintptr_t )lhs_addr, treshold  );

                        break;
                    }

                    case NS(OBJECT_TYPE_MULTIPOLE):
                    {
                        typedef NS(Multipole)                       belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(Multipole_compare_values_with_treshold)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr, treshold  );

                        break;
                    }

                    case NS(OBJECT_TYPE_RF_MULTIPOLE):
                    {
                        typedef NS(RFMultipole)                     belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value =
                        NS(RFMultipole_compare_values_with_treshold)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr, treshold  );

                        break;
                    }

                    case NS(OBJECT_TYPE_XYSHIFT):
                    {
                        typedef NS(XYShift)                         belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(XYShift_compare_values_with_treshold)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr, treshold  );

                        break;
                    }

                    case NS(OBJECT_TYPE_SROTATION):
                    {
                        typedef NS(SRotation)                       belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(SRotation_compare_values_with_treshold)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr, treshold  );

                        break;
                    }

                    case NS(OBJECT_TYPE_CAVITY):
                    {
                        typedef NS(Cavity)                          belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(Cavity_compare_values_with_treshold)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr, treshold  );

                        break;
                    }

                    case NS(OBJECT_TYPE_BEAM_BEAM_4D):
                    {
                        typedef NS(BeamBeam4D)                      belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(BeamBeam4D_compare_values_with_treshold)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr, treshold );

                        break;
                    }

                    case NS(OBJECT_TYPE_SC_COASTING):
                    {
                        typedef NS(SCCoasting) belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(SCCoasting_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_SC_QGAUSSIAN_PROF):
                    {
                        typedef NS(SCQGaussProfile) belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(SCQGaussProfile_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_BEAM_BEAM_6D):
                    {
                        typedef NS(BeamBeam6D)                      belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(BeamBeam6D_compare_values_with_treshold)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr, treshold );

                        break;
                    }

                    case NS(OBJECT_TYPE_BEAM_MONITOR):
                    {
                        typedef NS(BeamMonitor) belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(BeamMonitor_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr  );

                        break;
                    }

                    case NS(OBJECT_TYPE_LIMIT_RECT):
                    {
                        typedef NS(LimitRect) belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value =
                        NS(LimitRect_compare_values_with_treshold)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr, treshold  );

                        break;
                    }

                    case NS(OBJECT_TYPE_LIMIT_ELLIPSE):
                    {
                        typedef NS(LimitEllipse) belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value =
                        NS(LimitEllipse_compare_values_with_treshold)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr, treshold  );

                        break;
                    }

                    case NS(OBJECT_TYPE_DIPEDGE):
                    {
                        typedef NS(DipoleEdge) belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value =
                        NS(DipoleEdge_compare_values_with_treshold)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr, treshold );

                        break;
                    }

                    case NS(OBJECT_TYPE_TRICUB):
                    {
                        typedef NS(TriCub) belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value =
                        NS(TriCub_compare_values_with_treshold)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr, treshold );

                        break;
                    }

                    default:
                    {
                        compare_value = -1;
                    }
                };
            }
            else if( lhs_addr != ( address_t )0u )
            {
                compare_value = ( rhs_addr != lhs_addr ) ? +1 : 0;
            }
            else if( rhs_addr != ( address_t )0u )
            {
                compare_value = ( rhs_addr != lhs_addr ) ? -1 : 0;
            }
        }
        else if( lhs_type_id > rhs_type_id )
        {
            compare_value = +1;
        }
    }

    return compare_value;
}


SIXTRL_INLINE int NS(BeamElements_compare_lines)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT lhs_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT lhs_end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT rhs_it )
{
    int compare_value = -1;

    if( ( lhs_it != SIXTRL_NULLPTR ) && ( lhs_end != SIXTRL_NULLPTR ) &&
        ( rhs_it != SIXTRL_NULLPTR ) )
    {
        SIXTRL_ASSERT( ( ptrdiff_t )( lhs_end - lhs_it) > 0 );

        for( ; lhs_it != lhs_end ; ++lhs_it, ++rhs_it )
        {
            compare_value = NS(BeamElements_compare_objects)( lhs_it, rhs_it );

            if( 0 != compare_value )
            {
                break;
            }
        }
    }

    return compare_value;
}

SIXTRL_INLINE int NS(BeamElements_compare_lines_with_treshold)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT lhs_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT lhs_end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT rhs_it,
    SIXTRL_REAL_T const treshold )
{
    int compare_value = -1;

    if( ( lhs_it != SIXTRL_NULLPTR ) && ( lhs_end != SIXTRL_NULLPTR ) &&
        ( rhs_it != SIXTRL_NULLPTR ) )
    {
        SIXTRL_ASSERT( ( ptrdiff_t )( lhs_end - lhs_it) > 0 );

        for( ; lhs_it != lhs_end ; ++lhs_it, ++rhs_it )
        {
            compare_value = NS(BeamElements_compare_objects_with_treshold)(
                lhs_it, rhs_it, treshold );

            if( 0 != compare_value )
            {
                break;
            }
        }
    }

    return compare_value;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_HEADER_H__ */

/* end: tests/sixtracklib/testlib/common/beam_elements/beam_elements.h */
