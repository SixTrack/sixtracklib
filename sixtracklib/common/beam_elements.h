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
            case NS(OBJECT_TYPE_SPACE_CHARGE_COASTING):
            case NS(OBJECT_TYPE_SPACE_CHARGE_BUNCHED):
            {
                is_beam_element = true;
                break;
            }

            #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */

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

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_dataptrs =
                    NS(Drift_get_num_dataptrs)( ptr_begin );

                requ_num_slots    =
                    NS(Drift_get_num_slots)( ptr_begin, slot_size );
                break;
            }

            case NS(OBJECT_TYPE_DRIFT_EXACT):
            {
                typedef NS(DriftExact) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;

                requ_num_dataptrs =
                    NS(DriftExact_get_num_dataptrs)( ptr_begin );

                requ_num_slots    =
                    NS(DriftExact_get_num_slots)( ptr_begin, slot_size );
                break;
            }

            case NS(OBJECT_TYPE_MULTIPOLE):
            {
                typedef NS(MultiPole) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_slots =
                    NS(MultiPole_get_num_slots)( ptr_begin, slot_size );

                requ_num_dataptrs =
                    NS(MultiPole_get_num_dataptrs)( ptr_begin );
                break;
            }

            case NS(OBJECT_TYPE_RF_MULTIPOLE):
            {
                typedef NS(RFMultiPole) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;
                ++requ_num_objects;
                requ_num_slots = NS(RFMultiPole_num_slots)( ptr_begin, slot_size );
                requ_num_dataptrs = NS(RFMultiPole_num_dataptrs)( ptr_begin );
                break;
            }

            case NS(OBJECT_TYPE_XYSHIFT):
            {
                typedef NS(XYShift) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_slots =
                    NS(XYShift_get_num_slots)( ptr_begin, slot_size );

                requ_num_dataptrs =
                    NS(XYShift_get_num_dataptrs)( ptr_begin );

                break;
            }

            case NS(OBJECT_TYPE_SROTATION):
            {
                typedef NS(SRotation) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_slots =
                    NS(SRotation_get_num_slots)( ptr_begin, slot_size );

                requ_num_dataptrs =
                    NS(SRotation_get_num_dataptrs)( ptr_begin );
                break;
            }

            case NS(OBJECT_TYPE_CAVITY):
            {
                typedef NS(Cavity) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_slots =
                    NS(Cavity_get_num_slots)( ptr_begin, slot_size );

                requ_num_dataptrs =
                    NS(Cavity_get_num_dataptrs)( ptr_begin );
                break;
            }

            #if !defined( SIXTRL_DISABLE_BEAM_BEAM )

            case NS(OBJECT_TYPE_BEAM_BEAM_4D):
            {
                typedef NS(BeamBeam4D) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;

                requ_num_slots =
                NS(BeamBeam4D_get_required_num_slots_on_managed_buffer)(
                    SIXTRL_NULLPTR, ptr_begin, slot_size );

                requ_num_dataptrs =
                NS(BeamBeam4D_get_required_num_dataptrs_on_managed_buffer)(
                    SIXTRL_NULLPTR, ptr_begin, slot_size );

                break;
            }

            case NS(OBJECT_TYPE_SPACE_CHARGE_COASTING):
            {
                typedef NS(SpaceChargeCoasting) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;

                requ_num_slots =
                NS(SpaceChargeCoasting_get_required_num_slots_on_managed_buffer)(
                    SIXTRL_NULLPTR, ptr_begin, slot_size );

                requ_num_dataptrs =
                NS(SpaceChargeCoasting_get_required_num_dataptrs_on_managed_buffer)(
                    SIXTRL_NULLPTR, ptr_begin, slot_size );

                break;
            }

            case NS(OBJECT_TYPE_SPACE_CHARGE_BUNCHED):
            {
                typedef NS(SpaceChargeBunched) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;

                requ_num_slots =
                NS(SpaceChargeBunched_get_required_num_slots_on_managed_buffer)(
                    SIXTRL_NULLPTR, ptr_begin, slot_size );

                requ_num_dataptrs =
                NS(SpaceChargeBunched_get_required_num_dataptrs_on_managed_buffer)(
                    SIXTRL_NULLPTR, ptr_begin, slot_size );

                break;
            }

            case NS(OBJECT_TYPE_BEAM_BEAM_6D):
            {
                typedef NS(BeamBeam6D) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;

                requ_num_slots =
                NS(BeamBeam6D_get_required_num_slots_on_managed_buffer)(
                    SIXTRL_NULLPTR, ptr_begin, slot_size );

                requ_num_dataptrs =
                NS(BeamBeam6D_get_required_num_dataptrs_on_managed_buffer)(
                    SIXTRL_NULLPTR, ptr_begin, slot_size );

                break;
            }

            #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */

            case NS(OBJECT_TYPE_BEAM_MONITOR):
            {
                typedef NS(BeamMonitor) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;

                requ_num_slots = NS(BeamMonitor_get_num_slots)(
                    ptr_begin, slot_size );

                requ_num_dataptrs = NS(BeamMonitor_get_num_dataptrs)(
                    ptr_begin );

                break;
            }

            case NS(OBJECT_TYPE_LIMIT_RECT):
            {
                typedef NS(LimitRect) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;

                requ_num_slots =
                NS(LimitRect_get_required_num_slots_on_managed_buffer)(
                    SIXTRL_NULLPTR, ptr_begin, slot_size );

                requ_num_dataptrs =
                NS(LimitRect_get_required_num_dataptrs_on_managed_buffer)(
                    SIXTRL_NULLPTR, ptr_begin, slot_size );

                break;
            }

            case NS(OBJECT_TYPE_LIMIT_ELLIPSE):
            {
                typedef NS(LimitEllipse) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;

                requ_num_slots =
                NS(LimitEllipse_get_required_num_slots_on_managed_buffer)(
                    SIXTRL_NULLPTR, ptr_begin, slot_size );

                requ_num_dataptrs =
                NS(LimitEllipse_get_required_num_dataptrs_on_managed_buffer)(
                    SIXTRL_NULLPTR, ptr_begin, slot_size );

                break;
            }

            case NS(OBJECT_TYPE_LIMIT_RECT_ELLIPSE):
            {
                typedef NS(LimitRectEllipse) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;

                requ_num_slots = NS(LimitRectEllipse_num_slots)(
                    ptr_begin, slot_size );

                requ_num_dataptrs =
                    NS(LimitRectEllipse_num_dataptrs)( ptr_begin );

                break;
            }

            case NS(OBJECT_TYPE_DIPEDGE):
            {
                typedef NS(DipoleEdge) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;

                requ_num_slots =
                NS(DipoleEdge_get_required_num_slots_on_managed_buffer)(
                    SIXTRL_NULLPTR, ptr_begin, slot_size );

                requ_num_dataptrs =
                NS(DipoleEdge_get_required_num_dataptrs_on_managed_buffer)(
                    SIXTRL_NULLPTR, ptr_begin, slot_size );

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
                    typedef NS(MultiPole)                           belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*           ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const*     ptr_src_t;

                    success = NS(MultiPole_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_RF_MULTIPOLE):
                {
                    typedef NS(RFMultiPole) beam_element_t;
                    typedef SIXTRL_BE_ARGPTR_DEC beam_element_t* ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_src_t;

                    success = NS(RFMultiPole_copy)(
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
                    typedef NS(BeamBeam4D)                      belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*       ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_src_t;

                    success = NS(BeamBeam4D_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_SPACE_CHARGE_COASTING):
                {
                    typedef NS(SpaceChargeCoasting)             belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*       ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_src_t;

                    success = NS(SpaceChargeCoasting_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_SPACE_CHARGE_BUNCHED):
                {
                    typedef NS(SpaceChargeBunched)              belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*       ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_src_t;

                    success = NS(SpaceChargeBunched_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_BEAM_BEAM_6D):
                {
                    typedef NS(BeamBeam6D)                     belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*       ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_src_t;

                    success = NS(BeamBeam6D_copy)(
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
                    typedef NS(MultiPole)                 belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;
                    NS(MultiPole_clear)( ( ptr_belem_t )( uintptr_t )obj_addr );
                    break;
                }

                case NS(OBJECT_TYPE_RF_MULTIPOLE):
                {
                    typedef NS(RFMultiPole) belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;
                    NS(RFMultiPole_clear)( ( ptr_belem_t )( uintptr_t )obj_addr );
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
                    typedef NS(BeamBeam4D)                belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;

                    NS(BeamBeam4D_clear)( ( ptr_belem_t )( uintptr_t )obj_addr );
                    break;
                }

                case NS(OBJECT_TYPE_SPACE_CHARGE_COASTING):
                {
                    typedef NS(SpaceChargeCoasting)       belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;

                    NS(SpaceChargeCoasting_clear)(
                        ( ptr_belem_t )( uintptr_t )obj_addr );
                    break;
                }

                case NS(OBJECT_TYPE_SPACE_CHARGE_BUNCHED):
                {
                    typedef NS(SpaceChargeBunched)       belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;

                    NS(SpaceChargeBunched_clear)(
                        ( ptr_belem_t )( uintptr_t )obj_addr );
                    break;
                }

                case NS(OBJECT_TYPE_BEAM_BEAM_6D):
                {
                    typedef NS(BeamBeam6D)                belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;

                    NS(BeamBeam6D_clear)( ( ptr_belem_t )( uintptr_t )obj_addr );
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
                typedef NS(MultiPole)                    mp_t;
                typedef NS(multipole_order_t)            mp_order_t;
                typedef SIXTRL_BE_ARGPTR_DEC mp_t const* ptr_mp_t;

                ptr_mp_t ptr_mp = ( ptr_mp_t )( uintptr_t )begin_addr;
                mp_order_t const order = NS(MultiPole_get_order)( ptr_mp );

                success = ( SIXTRL_NULLPTR !=
                    NS(MultiPole_new)( buffer, order ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_RF_MULTIPOLE):
            {
                typedef NS(RFMultiPole)                  mp_t;
                typedef NS(rf_multipole_int_t)           mp_order_t;
                typedef SIXTRL_BE_ARGPTR_DEC mp_t const* ptr_mp_t;

                ptr_mp_t ptr_mp = ( ptr_mp_t )( uintptr_t )begin_addr;
                mp_order_t const order = NS(RFMultiPole_order)( ptr_mp );

                success = ( SIXTRL_NULLPTR !=
                    NS(RFMultiPole_new)( buffer, order ) ) ? 0 : -1;

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
                typedef NS(BeamBeam4D)                          beam_beam_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_beam_t const* ptr_beam_beam_t;

                ptr_beam_beam_t ptr_beam_beam =
                    ( ptr_beam_beam_t )( uintptr_t )begin_addr;

                NS(buffer_size_t) const data_size =
                    NS(BeamBeam4D_get_data_size)( ptr_beam_beam );

                SIXTRL_ASSERT( ptr_beam_beam != SIXTRL_NULLPTR );

                success = ( SIXTRL_NULLPTR !=
                    NS(BeamBeam4D_new)( buffer, data_size ) );

                break;
            }

            case NS(OBJECT_TYPE_SPACE_CHARGE_COASTING):
            {
                typedef NS(SpaceChargeCoasting) sc_coasting_t;
                typedef SIXTRL_BE_ARGPTR_DEC sc_coasting_t const*
                        ptr_sc_coasting_t;

                ptr_sc_coasting_t ptr_sc_coasting =
                    ( ptr_sc_coasting_t )( uintptr_t )begin_addr;

                NS(buffer_size_t) const data_size =
                    NS(SpaceChargeCoasting_get_data_size)( ptr_sc_coasting );

                SIXTRL_ASSERT( ptr_sc_coasting != SIXTRL_NULLPTR );

                success = ( SIXTRL_NULLPTR !=
                    NS(SpaceChargeCoasting_new)( buffer, data_size ) );

                break;
            }

            case NS(OBJECT_TYPE_SPACE_CHARGE_BUNCHED):
            {
                typedef NS(SpaceChargeBunched) sc_bunched_t;
                typedef SIXTRL_BE_ARGPTR_DEC sc_bunched_t const*
                        ptr_sc_bunched_t;

                ptr_sc_bunched_t ptr_sc_bunched =
                    ( ptr_sc_bunched_t )( uintptr_t )begin_addr;

                NS(buffer_size_t) const data_size =
                    NS(SpaceChargeBunched_get_data_size)( ptr_sc_bunched );

                SIXTRL_ASSERT( ptr_sc_bunched != SIXTRL_NULLPTR );

                success = ( SIXTRL_NULLPTR !=
                    NS(SpaceChargeBunched_new)( buffer, data_size ) );

                break;
            }

            case NS(OBJECT_TYPE_BEAM_BEAM_6D):
            {
                typedef NS(BeamBeam6D)                          beam_beam_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_beam_t const* ptr_beam_beam_t;

                ptr_beam_beam_t ptr_beam_beam =
                    ( ptr_beam_beam_t )( uintptr_t )begin_addr;

                NS(buffer_size_t) const data_size =
                    NS(BeamBeam6D_get_data_size)( ptr_beam_beam );

                SIXTRL_ASSERT( ptr_beam_beam != SIXTRL_NULLPTR );

                success = ( SIXTRL_NULLPTR !=
                    NS(BeamBeam4D_new)( buffer, data_size ) );

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
                typedef NS(MultiPole)                    mp_t;
                typedef SIXTRL_BE_ARGPTR_DEC mp_t const* ptr_mp_t;

                ptr_mp_t ptr_mp = ( ptr_mp_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(MultiPole_add_copy)( buffer, ptr_mp ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_RF_MULTIPOLE):
            {
                typedef NS(RFMultiPole) mp_t;
                typedef SIXTRL_BE_ARGPTR_DEC mp_t const* ptr_mp_t;

                ptr_mp_t ptr_mp = ( ptr_mp_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(RFMultiPole_add_copy)( buffer, ptr_mp ) ) ? 0 : -1;

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
                typedef NS(BeamBeam4D)                          beam_beam_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_beam_t const* ptr_beam_beam_t;

                ptr_beam_beam_t orig = ( ptr_beam_beam_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(BeamBeam4D_add_copy)( buffer, orig ) );

                break;
            }

            case NS(OBJECT_TYPE_SPACE_CHARGE_COASTING):
            {
                typedef NS(SpaceChargeCoasting) sc_coasting_t;
                typedef SIXTRL_BE_ARGPTR_DEC sc_coasting_t const*
                        ptr_sc_coasting_t;

                ptr_sc_coasting_t orig =
                    ( ptr_sc_coasting_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(SpaceChargeCoasting_add_copy)( buffer, orig ) );

                break;
            }

            case NS(OBJECT_TYPE_SPACE_CHARGE_BUNCHED):
            {
                typedef NS(SpaceChargeBunched) sc_bunched_t;
                typedef SIXTRL_BE_ARGPTR_DEC sc_bunched_t const*
                        ptr_sc_bunched_t;

                ptr_sc_bunched_t orig =
                    ( ptr_sc_bunched_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(SpaceChargeBunched_add_copy)( buffer, orig ) );

                break;
            }

            case NS(OBJECT_TYPE_BEAM_BEAM_6D):
            {
                typedef NS(BeamBeam6D)                          beam_beam_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_beam_t const* ptr_beam_beam_t;

                ptr_beam_beam_t orig = ( ptr_beam_beam_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(BeamBeam6D_add_copy)( buffer, orig ) );
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
