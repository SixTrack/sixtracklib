#ifndef SIXTRACKLIB_COMMON_BEAM_ELEMENTS_H__
#define SIXTRACKLIB_COMMON_BEAM_ELEMENTS_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/be_drift/be_drift.h"
    #include "sixtracklib/common/be_cavity/be_cavity.h"
    #include "sixtracklib/common/be_multipole/be_multipole.h"
    #include "sixtracklib/common/be_rfmultipole/be_rfmultipole.h"
    #include "sixtracklib/common/be_srotation/be_srotation.h"
    #include "sixtracklib/common/be_xyshift/be_xyshift.h"
    #include "sixtracklib/common/be_monitor/be_monitor.h"
    #include "sixtracklib/common/be_limit/be_limit_rect.h"
    #include "sixtracklib/common/be_limit/be_limit_ellipse.h"
    #include "sixtracklib/common/be_limit/be_limit_rect_ellipse.h"
    #include "sixtracklib/common/be_dipedge/be_dipedge.h"
    #include "sixtracklib/common/be_tricub/be_tricub.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */

SIXTRL_STATIC SIXTRL_FN bool NS(BeamElements_is_beam_element_obj)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj );

SIXTRL_STATIC SIXTRL_FN bool
NS(BeamElements_objects_range_are_all_beam_elements)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_end );

SIXTRL_STATIC SIXTRL_FN bool
NS(BeamElements_managed_buffer_is_beam_elements_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN int NS(BeamElements_calc_buffer_parameters_for_object)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT num_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT num_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT num_dataptrs,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN int NS(BeamElements_copy_object)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT destination,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT src );

SIXTRL_STATIC SIXTRL_FN void NS(BeamElements_clear_object)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT obj );

/* ------------------------------------------------------------------------ */

SIXTRL_STATIC SIXTRL_FN int NS(BeamElements_calc_buffer_parameters_for_line)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT num_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT num_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT num_dataptrs,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN int NS(BeamElements_copy_line)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT destination_begin );

/* ========================================================================= */

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_FN int NS(BeamElements_add_single_new_to_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj );

SIXTRL_STATIC SIXTRL_FN int NS(BeamElements_copy_single_to_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj );

SIXTRL_STATIC SIXTRL_FN int NS(BeamElements_add_new_to_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end );

SIXTRL_STATIC SIXTRL_FN int NS(BeamElements_copy_to_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end );

SIXTRL_STATIC SIXTRL_FN void NS(BeamElements_clear_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_STATIC SIXTRL_FN bool NS(BeamElements_is_beam_elements_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer );

#endif /* !defined( _GPUCODE ) */


/* ========================================================================= */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====            Implementation of inline functions                ====== */
/* ========================================================================= */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"

    #if !defined( _GPUCODE )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */

    #if !defined( SIXTRL_DISABLE_BEAM_BEAM )
        #include "sixtracklib/common/be_beamfields/be_beamfields.h"
    #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM )  */

#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE bool NS(BeamElements_is_beam_element_obj)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj )
{
    bool is_beam_element = false;

    if( obj!= SIXTRL_NULLPTR )
    {
        typedef NS(object_type_id_t) type_id_t;
        type_id_t const type_id = NS(Object_get_type_id)( obj );

        switch( type_id )
        {
            case NS(OBJECT_TYPE_DRIFT):
            case NS(OBJECT_TYPE_DRIFT_EXACT):
            case NS(OBJECT_TYPE_MULTIPOLE):
            case NS(OBJECT_TYPE_XYSHIFT):
            case NS(OBJECT_TYPE_SROTATION):
            case NS(OBJECT_TYPE_CAVITY):
            case NS(OBJECT_TYPE_BEAM_MONITOR):
            case NS(OBJECT_TYPE_LIMIT_RECT):
            case NS(OBJECT_TYPE_LIMIT_ELLIPSE):
            case NS(OBJECT_TYPE_LIMIT_RECT_ELLIPSE):
            case NS(OBJECT_TYPE_DIPEDGE):
            case NS(OBJECT_TYPE_RF_MULTIPOLE):
            {
                is_beam_element = true;
                break;
            }

            #if !defined( SIXTRL_DISABLE_BEAM_BEAM )

            case NS(OBJECT_TYPE_BEAM_BEAM_4D):
            case NS(OBJECT_TYPE_BEAM_BEAM_6D):
            case NS(OBJECT_TYPE_SC_COASTING):
            case NS(OBJECT_TYPE_SC_QGAUSSIAN_PROF):
            case NS(OBJECT_TYPE_SC_INTERPOLATED_PROF):
            {
                is_beam_element = true;
                break;
            }

            #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */

            #if !defined( SIXTRL_DISABLE_TRICUB )

            case NS(OBJECT_TYPE_TRICUB):
            {
                is_beam_element = true;
                break;
            }

            #endif /* !defined( SIXTRL_DISABLE_TRICUB ) */

            default:
            {
                is_beam_element = false;
            }
        };
    }

    return is_beam_element;
}

SIXTRL_INLINE bool NS(BeamElements_objects_range_are_all_beam_elements)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_end )
{
    bool are_all_beam_elements = false;

    if( ( obj_it != SIXTRL_NULLPTR ) && ( obj_end != SIXTRL_NULLPTR ) &&
        ( ( ( uintptr_t )obj_it ) <= ( uintptr_t )obj_end ) )
    {
        /* NOTE: An empty range evaluates as true, this is to allow use
         * in context of NS(Track_*particle*_line*) to finish empty/zero
         * length lines*/

        are_all_beam_elements = true;

        while( ( are_all_beam_elements ) && ( obj_it != obj_end ) )
        {
            are_all_beam_elements = NS(BeamElements_is_beam_element_obj)(
                obj_it++ );
        }
    }

    return are_all_beam_elements;
}

SIXTRL_INLINE int NS(BeamElements_calc_buffer_parameters_for_object)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_dataptrs,
    NS(buffer_size_t) const slot_size )
{
    int success = -1;

    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(object_type_id_t)    type_id_t;
    typedef NS(buffer_addr_t)       address_t;

    if( ( obj!= SIXTRL_NULLPTR ) && ( slot_size > ( buf_size_t )0u ) )
    {
        address_t const begin_addr = NS(Object_get_begin_addr)( obj );
        type_id_t const type_id    = NS(Object_get_type_id)( obj );

        buf_size_t requ_num_objects  = ( buf_size_t )0u;
        buf_size_t requ_num_slots    = ( buf_size_t )0u;
        buf_size_t requ_num_dataptrs = ( buf_size_t )0u;

        success = 0;

        switch( type_id )
        {
            case NS(OBJECT_TYPE_DRIFT):
            {
                typedef NS(Drift) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t elem = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_dataptrs = NS(Drift_num_dataptrs)( elem );
                requ_num_slots    = NS(Drift_num_slots)( elem, slot_size );
                break;
            }

            case NS(OBJECT_TYPE_DRIFT_EXACT):
            {
                typedef NS(DriftExact) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;
                ptr_belem_t elem = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_dataptrs = NS(DriftExact_num_dataptrs)( elem );
                requ_num_slots = NS(DriftExact_num_slots)( elem, slot_size );
                break;
            }

            case NS(OBJECT_TYPE_MULTIPOLE):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(Multipole) const* ptr_elem_t;
                ptr_elem_t elem = ( ptr_elem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_slots = NS(Multipole_num_slots)( elem, slot_size );
                requ_num_dataptrs = NS(Multipole_num_dataptrs)( elem );
                break;
            }

            case NS(OBJECT_TYPE_RF_MULTIPOLE):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(RFMultipole) const* ptr_t;
                ptr_t elem = ( ptr_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_slots = NS(RFMultipole_num_slots)( elem, slot_size );
                requ_num_dataptrs = NS(RFMultipole_num_dataptrs)( elem );
                break;
            }

            case NS(OBJECT_TYPE_XYSHIFT):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(XYShift) const* ptr_elem_t;
                ptr_elem_t elem = ( ptr_elem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_slots = NS(XYShift_num_slots)( elem, slot_size );
                requ_num_dataptrs = NS(XYShift_num_dataptrs)( elem );
                break;
            }

            case NS(OBJECT_TYPE_SROTATION):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(SRotation) const* ptr_elem_t;
                ptr_elem_t elem = ( ptr_elem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_slots = NS(SRotation_num_slots)( elem, slot_size );
                requ_num_dataptrs = NS(SRotation_num_dataptrs)( elem );
                break;
            }

            case NS(OBJECT_TYPE_CAVITY):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(Cavity) const* ptr_elem_t;
                ptr_elem_t elem = ( ptr_elem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_dataptrs = NS(Cavity_num_dataptrs)( elem );
                requ_num_slots = NS(Cavity_num_slots)( elem, slot_size );
                break;
            }

            #if !defined( SIXTRL_DISABLE_BEAM_BEAM )

            case NS(OBJECT_TYPE_BEAM_BEAM_4D):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D) const* ptr_elem_t;
                ptr_elem_t elem = ( ptr_elem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_dataptrs = NS(BeamBeam4D_num_dataptrs)( elem );
                requ_num_slots = NS(BeamBeam4D_num_slots)( elem, slot_size );

                break;
            }

            case NS(OBJECT_TYPE_BEAM_BEAM_6D):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D) const* ptr_elem_t;
                ptr_elem_t elem = ( ptr_elem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_dataptrs = NS(BeamBeam6D_num_dataptrs)( elem );
                requ_num_slots = NS(BeamBeam6D_num_slots)( elem, slot_size );

                break;
            }

            case NS(OBJECT_TYPE_SC_COASTING):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(SCCoasting) const*
                        ptr_belem_t;

                ptr_belem_t sc_elem = ( ptr_belem_t )( uintptr_t )begin_addr;
                ++requ_num_objects;

                requ_num_slots =
                    NS(SCCoasting_num_slots)( sc_elem, slot_size );

                requ_num_dataptrs =
                    NS(SCCoasting_num_dataptrs)( sc_elem );

                break;
            }

            case NS(OBJECT_TYPE_SC_QGAUSSIAN_PROF):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)
                        const* ptr_belem_t;

                ptr_belem_t sc_elem = ( ptr_belem_t )( uintptr_t )begin_addr;
                ++requ_num_objects;

                requ_num_slots = NS(SCQGaussProfile_num_slots)(
                    sc_elem, slot_size );

                requ_num_dataptrs =
                    NS(SCQGaussProfile_num_dataptrs)( sc_elem );

                break;
            }

            case NS(OBJECT_TYPE_SC_INTERPOLATED_PROF):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(SCInterpolatedProfile)
                        const* ptr_belem_t;

                ptr_belem_t sc_elem = ( ptr_belem_t )( uintptr_t )begin_addr;
                ++requ_num_objects;

                requ_num_slots = NS(SCInterpolatedProfile_num_slots)(
                    sc_elem, slot_size );

                requ_num_dataptrs =
                    NS(SCInterpolatedProfile_num_dataptrs)( sc_elem );

                break;
            }

            #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */

            case NS(OBJECT_TYPE_BEAM_MONITOR):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor) const* ptr_elem_t;
                ptr_elem_t elem = ( ptr_elem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_slots = NS(BeamMonitor_num_slots)( elem, slot_size );
                requ_num_dataptrs = NS(BeamMonitor_num_dataptrs)( elem );
                break;
            }

            case NS(OBJECT_TYPE_LIMIT_RECT):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(LimitRect) const* ptr_elem_t;
                ptr_elem_t elem = ( ptr_elem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_slots = NS(LimitRect_num_slots)( elem, slot_size );
                requ_num_dataptrs = NS(LimitRect_num_dataptrs)( elem );
                break;
            }

            case NS(OBJECT_TYPE_LIMIT_ELLIPSE):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse) const* ptr_elem_t;
                ptr_elem_t elem = ( ptr_elem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_slots = NS(LimitEllipse_num_slots)( elem, slot_size );
                requ_num_dataptrs = NS(LimitEllipse_num_dataptrs)( elem );
                break;
            }

            case NS(OBJECT_TYPE_LIMIT_RECT_ELLIPSE):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse) const*
                        ptr_elem_t;
                ptr_elem_t elem = ( ptr_elem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_dataptrs = NS(LimitRectEllipse_num_dataptrs)( elem );
                requ_num_slots = NS(LimitRectEllipse_num_slots)(
                    elem, slot_size );
                break;
            }

            case NS(OBJECT_TYPE_DIPEDGE):
            {
                typedef SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge) const* ptr_belem_t;
                ptr_belem_t elem = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_slots = NS(DipoleEdge_num_slots)( elem, slot_size );
                requ_num_dataptrs = NS(DipoleEdge_num_dataptrs)( elem );
                break;
            }

            case NS(OBJECT_TYPE_TRICUB):
            {
                ++requ_num_objects;
                requ_num_slots = NS(TriCub_num_slots)(
                    SIXTRL_NULLPTR, slot_size );

                requ_num_dataptrs = NS(TriCub_num_dataptrs)( SIXTRL_NULLPTR );

                break;
            }

            case NS(OBJECT_TYPE_TRICUB_DATA):
            {
                ++requ_num_objects;

                requ_num_slots = NS(TriCubData_num_slots)(
                    SIXTRL_NULLPTR, slot_size );

                requ_num_dataptrs =
                    NS(TriCubData_num_dataptrs)( SIXTRL_NULLPTR );

                break;
            }

            default:
            {
                success = -1;
            }
        };

        if( success == 0 )
        {
            if(  ptr_num_objects != SIXTRL_NULLPTR )
            {
                *ptr_num_objects += requ_num_objects;
            }

            if(  ptr_num_slots != SIXTRL_NULLPTR )
            {
                *ptr_num_slots += requ_num_slots;
            }

            if(  ptr_num_dataptrs != SIXTRL_NULLPTR )
            {
                *ptr_num_dataptrs += requ_num_dataptrs;
            }
        }
    }

    return success;
}

SIXTRL_INLINE int NS(BeamElements_copy_object)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT destination,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT src )
{
    int success = -1;

    if( ( destination != SIXTRL_NULLPTR ) && ( src != SIXTRL_NULLPTR ) )
    {
        typedef NS(buffer_size_t)       buf_size_t;
        typedef NS(buffer_addr_t)       address_t;
        typedef NS(object_type_id_t)    type_id_t;

        type_id_t  const    type_id = NS(Object_get_type_id)( src );
        address_t  const   src_addr = NS(Object_get_begin_addr)( src );
        buf_size_t const   src_size = NS(Object_get_size)( src );

        address_t  const  dest_addr = NS(Object_get_begin_addr)( destination );
        buf_size_t const  dest_size = NS(Object_get_size)( destination );

        if( ( type_id   == NS(Object_get_type_id)( destination ) ) &&
            ( src_addr  != ( address_t )0u ) && ( src_addr  != dest_addr ) &&
            ( dest_addr != ( address_t )0u ) && ( src_size  <= dest_size ) )
        {
            switch( type_id )
            {
                case NS(OBJECT_TYPE_DRIFT):
                {
                    typedef NS(Drift)                               belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*           ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const*     ptr_src_t;

                    success = NS(Drift_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_DRIFT_EXACT):
                {
                    typedef NS(DriftExact)                          belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*           ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const*     ptr_src_t;

                    success = NS(DriftExact_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_MULTIPOLE):
                {
                    typedef NS(Multipole)                           belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*           ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const*     ptr_src_t;

                    success = NS(Multipole_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_RF_MULTIPOLE):
                {
                    typedef NS(RFMultipole) beam_element_t;
                    typedef SIXTRL_BE_ARGPTR_DEC beam_element_t* ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_src_t;

                    success = NS(RFMultipole_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_XYSHIFT):
                {
                    typedef NS(XYShift)                             belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*           ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const*     ptr_src_t;

                    success = NS(XYShift_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_SROTATION):
                {
                    typedef NS(SRotation)                           belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*           ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const*     ptr_src_t;

                    success = NS(SRotation_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_CAVITY):
                {
                    typedef NS(Cavity)                              belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*           ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const*     ptr_src_t;

                    success = NS(Cavity_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                #if !defined( SIXTRL_DISABLE_BEAM_BEAM )

                case NS(OBJECT_TYPE_BEAM_BEAM_4D):
                {
                    success = ( ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS
                        == NS(BeamBeam4D_copy)(
                        ( SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* )(
                            uintptr_t )dest_addr,
                        ( SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D) const* )(
                            uintptr_t )src_addr ) );

                    break;
                }

                case NS(OBJECT_TYPE_BEAM_BEAM_6D):
                {
                    success = ( ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS
                        == NS(BeamBeam6D_copy)(
                        ( SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* )(
                            uintptr_t )dest_addr,
                        ( SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D) const* )(
                            uintptr_t )src_addr ) );

                    break;
                }

                case NS(OBJECT_TYPE_SC_COASTING):
                {
                    typedef NS(SCCoasting)             belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*       ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_src_t;

                    success = NS(SCCoasting_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_SC_QGAUSSIAN_PROF):
                {
                    typedef NS(SCQGaussProfile)              belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*       ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_src_t;

                    success = NS(SCQGaussProfile_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */

                case NS(OBJECT_TYPE_BEAM_MONITOR):
                {
                    typedef NS(BeamMonitor)                     belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*       ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_src_t;

                    success = NS(BeamMonitor_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_LIMIT_RECT):
                {
                    typedef NS(LimitRect) belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_src_t;

                    success = NS(LimitRect_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_LIMIT_ELLIPSE):
                {
                    typedef NS(LimitEllipse) belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_src_t;

                    success = NS(LimitEllipse_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_LIMIT_RECT_ELLIPSE):
                {
                    typedef NS(LimitRectEllipse) belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_src_t;

                    success = NS(LimitRectEllipse_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_DIPEDGE):
                {
                    typedef NS(DipoleEdge) belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_src_t;

                    success = NS(DipoleEdge_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_TRICUB):
                {
                    typedef NS(TriCub) belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_src_t;

                    success = NS(TriCub_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_TRICUB_DATA):
                {
                    typedef NS(TriCubData) belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_src_t;

                    success = NS(TriCubData_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                default:
                {
                    success = -1;
                }
            };
        }
    }

    return success;
}

SIXTRL_STATIC SIXTRL_FN void NS(BeamElements_clear_object)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT obj )
{
    if( obj != SIXTRL_NULLPTR )
    {
        typedef NS(buffer_addr_t)       address_t;
        typedef NS(object_type_id_t)    type_id_t;

        type_id_t const type_id = NS(Object_get_type_id)( obj );
        address_t const obj_addr = NS(Object_get_begin_addr)( obj );

        if( obj_addr != ( address_t)0u )
        {
            switch( type_id )
            {
                case NS(OBJECT_TYPE_DRIFT):
                {
                    typedef NS(Drift)                      belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*  ptr_belem_t;
                    NS(Drift_clear)( ( ptr_belem_t )( uintptr_t )obj_addr );
                    break;
                }

                case NS(OBJECT_TYPE_DRIFT_EXACT):
                {
                    typedef NS(DriftExact)                belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;
                    NS(DriftExact_clear)( ( ptr_belem_t )( uintptr_t )obj_addr );
                    break;
                }

                case NS(OBJECT_TYPE_MULTIPOLE):
                {
                    typedef NS(Multipole)                 belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;
                    NS(Multipole_clear)( ( ptr_belem_t )( uintptr_t )obj_addr );
                    break;
                }

                case NS(OBJECT_TYPE_RF_MULTIPOLE):
                {
                    typedef NS(RFMultipole) belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;
                    NS(RFMultipole_clear)( ( ptr_belem_t )( uintptr_t )obj_addr );
                    break;
                }

                case NS(OBJECT_TYPE_XYSHIFT):
                {
                    typedef NS(XYShift)                   belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;
                    NS(XYShift_clear)( ( ptr_belem_t )( uintptr_t )obj_addr );
                    break;
                }

                case NS(OBJECT_TYPE_SROTATION):
                {
                    typedef NS(SRotation)                 belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;
                    NS(SRotation_clear)( ( ptr_belem_t )( uintptr_t )obj_addr );
                    break;
                }

                case NS(OBJECT_TYPE_CAVITY):
                {
                    typedef NS(Cavity)                    belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;
                    NS(Cavity_clear)( ( ptr_belem_t )( uintptr_t )obj_addr );
                    break;
                }

                #if !defined( SIXTRL_DISABLE_BEAM_BEAM )

                case NS(OBJECT_TYPE_BEAM_BEAM_4D):
                {
                    NS(BeamBeam4D_clear)( ( SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)*
                        )( uintptr_t )obj_addr );
                    break;
                }

                case NS(OBJECT_TYPE_BEAM_BEAM_6D):
                {
                    NS(BeamBeam6D_clear)( ( SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)*
                        )( uintptr_t )obj_addr );
                    break;
                }

                case NS(OBJECT_TYPE_SC_COASTING):
                {
                    typedef NS(SCCoasting)       belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;

                    NS(SCCoasting_clear)(
                        ( ptr_belem_t )( uintptr_t )obj_addr );
                    break;
                }

                case NS(OBJECT_TYPE_SC_QGAUSSIAN_PROF):
                {
                    typedef NS(SCQGaussProfile)       belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;

                    NS(SCQGaussProfile_clear)(
                        ( ptr_belem_t )( uintptr_t )obj_addr );
                    break;
                }

                #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */

                case NS(OBJECT_TYPE_BEAM_MONITOR):
                {
                    typedef NS(BeamMonitor)                    belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;
                    NS(BeamMonitor_clear)( ( ptr_belem_t )( uintptr_t )obj_addr );
                    break;
                }

                case NS(OBJECT_TYPE_LIMIT_RECT):
                {
                    typedef NS(LimitRect) belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;
                    NS(LimitRect_clear)(
                        ( ptr_belem_t )( uintptr_t )obj_addr );

                    break;
                }

                case NS(OBJECT_TYPE_LIMIT_ELLIPSE):
                {
                    typedef NS(LimitEllipse) belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;
                    NS(LimitEllipse_clear)(
                        ( ptr_belem_t )( uintptr_t )obj_addr );

                    break;
                }

                case NS(OBJECT_TYPE_LIMIT_RECT_ELLIPSE):
                {
                    typedef NS(LimitRectEllipse) belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;
                    NS(LimitRectEllipse_clear)(
                        ( ptr_belem_t )( uintptr_t )obj_addr );

                    break;
                }

                case NS(OBJECT_TYPE_DIPEDGE):
                {
                    typedef NS(DipoleEdge) belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;
                    NS(DipoleEdge_clear)( ( ptr_belem_t )( uintptr_t )obj_addr );
                    break;
                }

                case NS(OBJECT_TYPE_TRICUB):
                {
                    typedef NS(TriCub) belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;
                    NS(TriCub_clear)( ( ptr_belem_t )( uintptr_t )obj_addr );
                    break;
                }

                case NS(OBJECT_TYPE_TRICUB_DATA):
                {
                    typedef NS(TriCubData) belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;
                    NS(TriCubData_clear)( ( ptr_belem_t )( uintptr_t )obj_addr );
                    break;
                }

                default: {} /* To satisfy compilers that complain if no
                               default section is available */
            };
        }
    }

    return;
}

/* ------------------------------------------------------------------------ */

SIXTRL_INLINE int NS(BeamElements_calc_buffer_parameters_for_line)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT src_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT src_end,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT num_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT num_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT num_dataptrs,
    NS(buffer_size_t) const slot_size )
{
    int success = -1;

    if( ( src_it != SIXTRL_NULLPTR ) && ( src_end != SIXTRL_NULLPTR ) )
    {
        success = 0;

        for( ; src_it != src_end ; ++src_it )
        {
            int const ret = NS(BeamElements_calc_buffer_parameters_for_object)(
                    src_it, num_objects, num_slots, num_dataptrs, slot_size );

            if( 0 != ret )
            {
                success |= ret;
                break;
            }
        }
    }

    return success;
}

SIXTRL_INLINE int NS(BeamElements_copy_line)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT src_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT src_end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT dest_it )
{
    int success = -1;

    if( ( src_it  != SIXTRL_NULLPTR ) && ( src_end != SIXTRL_NULLPTR ) &&
        ( dest_it != SIXTRL_NULLPTR ) )
    {
        SIXTRL_ASSERT( ( ptrdiff_t )( src_end - src_it) > 0 );

        success = 0;

        for( ; src_it != src_end ; ++src_it, ++dest_it )
        {
            int const ret = NS(BeamElements_copy_object)( dest_it, src_it );

            if( 0 != ret )
            {
                success |= ret;
                break;
            }
        }
    }

    return success;
}

/* ========================================================================= */

#if !defined( _GPUCODE )

SIXTRL_INLINE int NS(BeamElements_add_single_new_to_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj )
{
    int success = -1;

    if( obj != SIXTRL_NULLPTR )
    {
        typedef NS(buffer_addr_t)       address_t;
        typedef NS(object_type_id_t)    type_id_t;

        type_id_t  const    type_id = NS(Object_get_type_id)( obj );
        address_t  const begin_addr = NS(Object_get_begin_addr)( obj );

        switch( type_id )
        {
            case NS(OBJECT_TYPE_DRIFT):
            {
                success = ( SIXTRL_NULLPTR !=
                    NS(Drift_new)( buffer ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_DRIFT_EXACT):
            {
                success = ( SIXTRL_NULLPTR !=
                    NS(DriftExact_new)( buffer ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_MULTIPOLE):
            {
                typedef NS(Multipole)                    mp_t;
                typedef NS(multipole_order_t)            mp_order_t;
                typedef SIXTRL_BE_ARGPTR_DEC mp_t const* ptr_mp_t;

                ptr_mp_t ptr_mp = ( ptr_mp_t )( uintptr_t )begin_addr;
                mp_order_t const order = NS(Multipole_order)( ptr_mp );

                success = ( SIXTRL_NULLPTR !=
                    NS(Multipole_new)( buffer, order ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_RF_MULTIPOLE):
            {
                typedef NS(RFMultipole)                  mp_t;
                typedef NS(rf_multipole_int_t)           mp_order_t;
                typedef SIXTRL_BE_ARGPTR_DEC mp_t const* ptr_mp_t;

                ptr_mp_t ptr_mp = ( ptr_mp_t )( uintptr_t )begin_addr;
                mp_order_t const order = NS(RFMultipole_order)( ptr_mp );

                success = ( SIXTRL_NULLPTR !=
                    NS(RFMultipole_new)( buffer, order ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_XYSHIFT):
            {
                success = ( SIXTRL_NULLPTR !=
                    NS(XYShift_new)( buffer ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_SROTATION):
            {
                success = ( SIXTRL_NULLPTR !=
                    NS(SRotation_new)( buffer ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_CAVITY):
            {
                success = ( SIXTRL_NULLPTR !=
                    NS(Cavity_new)( buffer ) ) ? 0 : -1;

                break;
            }

            #if !defined( SIXTRL_DISABLE_BEAM_BEAM )

            case NS(OBJECT_TYPE_BEAM_BEAM_4D):
            {
                success = ( SIXTRL_NULLPTR != NS(BeamBeam4D_new)(
                    buffer, NS(BeamBeam4D_data_size)( ( SIXTRL_BE_ARGPTR_DEC
                        NS(BeamBeam4D) const* )( uintptr_t )begin_addr ) ) );
                break;
            }

            case NS(OBJECT_TYPE_BEAM_BEAM_6D):
            {
                success = ( SIXTRL_NULLPTR != NS(BeamBeam6D_new)(
                    buffer, NS(BeamBeam6D_data_size)( ( SIXTRL_BE_ARGPTR_DEC
                        NS(BeamBeam6D) const* )( uintptr_t )begin_addr ) ) );
                break;
            }

            case NS(OBJECT_TYPE_SC_COASTING):
            {
                success = ( SIXTRL_NULLPTR !=
                    NS(SCCoasting_new)( buffer ) );

                break;
            }

            case NS(OBJECT_TYPE_SC_QGAUSSIAN_PROF):
            {
                success = ( SIXTRL_NULLPTR !=
                    NS(SCQGaussProfile_new)( buffer ) );

                break;
            }

            case NS(OBJECT_TYPE_SC_INTERPOLATED_PROF):
            {
                success = ( SIXTRL_NULLPTR !=
                    NS(SCInterpolatedProfile_new)( buffer ) );

                break;
            }

            #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */

            case NS(OBJECT_TYPE_BEAM_MONITOR):
            {
                success = ( SIXTRL_NULLPTR != NS(BeamMonitor_new)( buffer ) );
                break;
            }

            case NS(OBJECT_TYPE_LIMIT_RECT):
            {
                success = ( SIXTRL_NULLPTR != NS(LimitRect_new)( buffer ) );
                break;
            }

            case NS(OBJECT_TYPE_LIMIT_ELLIPSE):
            {
                success = ( SIXTRL_NULLPTR != NS(LimitEllipse_new)( buffer ) );
                break;
            }

            case NS(OBJECT_TYPE_LIMIT_RECT_ELLIPSE):
            {
                success = ( SIXTRL_NULLPTR != NS(LimitRectEllipse_new)(
                    buffer ) );
                break;
            }

            case NS(OBJECT_TYPE_DIPEDGE):
            {
                success = ( SIXTRL_NULLPTR != NS(DipoleEdge_new)( buffer ) );
                break;
            }

            case NS(OBJECT_TYPE_TRICUB):
            {
                success = ( SIXTRL_NULLPTR != NS(TriCub_new)( buffer ) )
                    ? 0 : -1;
                break;
            }

            case NS(OBJECT_TYPE_TRICUB_DATA):
            {
                typedef NS(TriCubData) tricub_data_t;
                typedef SIXTRL_BUFFER_DATAPTR_DEC
                    tricub_data_t const* ptr_tricub_data_t;

                ptr_tricub_data_t ptr_tricub = ( ptr_tricub_data_t )(
                    uintptr_t )begin_addr;

                NS(be_tricub_int_t) const nx = NS(TriCubData_nx)( ptr_tricub );
                NS(be_tricub_int_t) const ny = NS(TriCubData_ny)( ptr_tricub );
                NS(be_tricub_int_t) const nz = NS(TriCubData_nz)( ptr_tricub );

                success = ( SIXTRL_NULLPTR !=
                    NS(TriCubData_new)( buffer, nx, ny, nz ) ) ? 0 : -1;

                break;
            }

            default:
            {
                success = -1;
            }
        };
    }

    return success;
}

SIXTRL_INLINE int NS(BeamElements_copy_single_to_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj )
{
    int success = -1;

    if( obj != SIXTRL_NULLPTR )
    {
        typedef NS(buffer_addr_t)       address_t;
        typedef NS(object_type_id_t)    type_id_t;

        type_id_t  const    type_id = NS(Object_get_type_id)( obj );
        address_t  const begin_addr = NS(Object_get_begin_addr)( obj );

        switch( type_id )
        {
            case NS(OBJECT_TYPE_DRIFT):
            {
                typedef NS(Drift)             beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC  beam_element_t const* ptr_belem_t;

                ptr_belem_t orig = ( ptr_belem_t )( uintptr_t )begin_addr;
                success = ( SIXTRL_NULLPTR !=
                    NS(Drift_add_copy)( buffer, orig ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_DRIFT_EXACT):
            {
                typedef NS(DriftExact)        beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC  beam_element_t const* ptr_belem_t;

                ptr_belem_t orig = ( ptr_belem_t )( uintptr_t )begin_addr;
                success = ( SIXTRL_NULLPTR !=
                    NS(DriftExact_add_copy)( buffer, orig ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_MULTIPOLE):
            {
                typedef NS(Multipole)                    mp_t;
                typedef SIXTRL_BE_ARGPTR_DEC mp_t const* ptr_mp_t;

                ptr_mp_t ptr_mp = ( ptr_mp_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(Multipole_add_copy)( buffer, ptr_mp ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_RF_MULTIPOLE):
            {
                typedef NS(RFMultipole) mp_t;
                typedef SIXTRL_BE_ARGPTR_DEC mp_t const* ptr_mp_t;

                ptr_mp_t ptr_mp = ( ptr_mp_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(RFMultipole_add_copy)( buffer, ptr_mp ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_XYSHIFT):
            {
                typedef  NS(XYShift)           beam_element_t;
                typedef  SIXTRL_BE_ARGPTR_DEC  beam_element_t const* ptr_belem_t;

                ptr_belem_t orig = ( ptr_belem_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(XYShift_add_copy)( buffer, orig ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_SROTATION):
            {
                typedef  NS(SRotation)           beam_element_t;
                typedef  SIXTRL_BE_ARGPTR_DEC  beam_element_t const* ptr_belem_t;

                ptr_belem_t orig = ( ptr_belem_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(SRotation_add_copy)( buffer, orig ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_CAVITY):
            {
                typedef  NS(Cavity)            beam_element_t;
                typedef  SIXTRL_BE_ARGPTR_DEC  beam_element_t const* ptr_belem_t;

                ptr_belem_t orig = ( ptr_belem_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(Cavity_add_copy)( buffer, orig ) ) ? 0 : -1;

                break;
            }

            #if !defined( SIXTRL_DISABLE_BEAM_BEAM )

            case NS(OBJECT_TYPE_BEAM_BEAM_4D):
            {
                success = ( SIXTRL_NULLPTR !=
                    NS(BeamBeam4D_add_copy)( buffer, ( SIXTRL_BE_ARGPTR_DEC
                        NS(BeamBeam4D) const* )( uintptr_t )begin_addr ) );

                break;
            }

            case NS(OBJECT_TYPE_BEAM_BEAM_6D):
            {
                success = ( SIXTRL_NULLPTR !=
                    NS(BeamBeam6D_add_copy)( buffer, ( SIXTRL_BE_ARGPTR_DEC
                        NS(BeamBeam6D) const* )( uintptr_t )begin_addr ) );

                break;
            }

            case NS(OBJECT_TYPE_SC_COASTING):
            {
                typedef NS(SCCoasting) sc_coasting_t;
                typedef SIXTRL_BE_ARGPTR_DEC sc_coasting_t const*
                        ptr_sc_coasting_t;

                ptr_sc_coasting_t orig =
                    ( ptr_sc_coasting_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(SCCoasting_add_copy)( buffer, orig ) );

                break;
            }

            case NS(OBJECT_TYPE_SC_QGAUSSIAN_PROF):
            {
                typedef NS(SCQGaussProfile) sc_bunched_t;
                typedef SIXTRL_BE_ARGPTR_DEC sc_bunched_t const*
                        ptr_sc_bunched_t;

                ptr_sc_bunched_t orig =
                    ( ptr_sc_bunched_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(SCQGaussProfile_add_copy)( buffer, orig ) );

                break;
            }

            #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */

            case NS(OBJECT_TYPE_BEAM_MONITOR):
            {
                typedef  NS(BeamMonitor)       beam_element_t;
                typedef  SIXTRL_BE_ARGPTR_DEC  beam_element_t const* ptr_belem_t;

                ptr_belem_t orig = ( ptr_belem_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(BeamMonitor_add_copy)( buffer, orig ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_LIMIT_RECT):
            {
                typedef  NS(LimitRect) beam_element_t;
                typedef  SIXTRL_BE_ARGPTR_DEC  beam_element_t const* ptr_belem_t;

                ptr_belem_t orig = ( ptr_belem_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(LimitRect_add_copy)( buffer, orig ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_LIMIT_ELLIPSE):
            {
                typedef  NS(LimitEllipse) beam_element_t;
                typedef  SIXTRL_BE_ARGPTR_DEC  beam_element_t const* ptr_belem_t;

                ptr_belem_t orig = ( ptr_belem_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(LimitEllipse_add_copy)( buffer, orig ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_LIMIT_RECT_ELLIPSE):
            {
                typedef  NS(LimitRectEllipse) beam_element_t;
                typedef  SIXTRL_BE_ARGPTR_DEC  beam_element_t const* ptr_belem_t;

                ptr_belem_t orig = ( ptr_belem_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(LimitRectEllipse_add_copy)( buffer, orig ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_DIPEDGE):
            {
                typedef  NS(DipoleEdge) beam_element_t;
                typedef  SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t orig = ( ptr_belem_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(DipoleEdge_add_copy)( buffer, orig ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_TRICUB):
            {
                typedef  NS(TriCub) beam_element_t;
                typedef  SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t orig = ( ptr_belem_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(TriCub_add_copy)( buffer, orig ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_TRICUB_DATA):
            {
                typedef  NS(TriCubData) beam_element_t;
                typedef  SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t orig = ( ptr_belem_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(TriCubData_add_copy)( buffer, orig ) ) ? 0 : -1;

                break;
            }

            default:
            {
                success = -1;
            }
        };
    }

    return success;
}

SIXTRL_INLINE int NS(BeamElements_add_new_to_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end )
{
    int success = -1;

    if( ( it != SIXTRL_NULLPTR ) && ( end != SIXTRL_NULLPTR ) )
    {
        success = ( ( ( ptrdiff_t )( end - it ) ) > 0 ) ? 0 : -1;

        for( ; it != end ; ++it )
        {
            int const ret = NS(BeamElements_add_single_new_to_buffer)( buffer, it );

            if( 0 != ret )
            {
                success |= ret;
                break;
            }
        }
    }

    return success;
}

SIXTRL_INLINE int NS(BeamElements_copy_to_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*  SIXTRL_RESTRICT it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*  SIXTRL_RESTRICT end )
{
    int success = -1;

    if( ( it != SIXTRL_NULLPTR ) && ( end != SIXTRL_NULLPTR ) )
    {
        success = ( ( ( ptrdiff_t )( end - it ) ) > 0 ) ? 0 : -1;

        for( ; it != end ; ++it )
        {
            int const ret = NS(BeamElements_copy_single_to_buffer)( buffer, it );

            if( 0 != ret )
            {
                success |= ret;
                break;
            }
        }
    }

    return success;
}

SIXTRL_INLINE void NS(BeamElements_clear_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* obj_iter_t;

    obj_iter_t it  = NS(Buffer_get_objects_begin)( buffer );
    obj_iter_t end = NS(Buffer_get_objects_end)( buffer );

    if( it != end )
    {
        SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( buffer ) );

        for( ; it != end ; ++it )
        {
           NS(BeamElements_clear_object)( it );
        }
    }

    return;
}

SIXTRL_INLINE bool NS(BeamElements_is_beam_elements_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(BeamElements_managed_buffer_is_beam_elements_buffer)(
        ( unsigned char const* )( uintptr_t )NS(Buffer_get_data_begin_addr)(
            buffer ), NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE bool NS(BeamElements_managed_buffer_is_beam_elements_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const slot_size )
{
    SIXTRL_ASSERT( buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( slot_size > ( NS(buffer_size_t) )0u );
    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)( buffer, slot_size ) );

    return NS(BeamElements_objects_range_are_all_beam_elements)(
        NS(ManagedBuffer_get_const_objects_index_begin)( buffer, slot_size ),
        NS(ManagedBuffer_get_const_objects_index_end)( buffer, slot_size ) );
}

#endif /* !defined( _GPUCODE ) */


#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_BEAM_ELEMENTS_H__ */

/* end: sixtracklib/sixtracklib/common/beam_elements.h */
