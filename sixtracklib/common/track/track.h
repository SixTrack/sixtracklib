#ifndef SIXTRACKLIB_COMMON_TRACK_TRACK_C99_H__
#define SIXTRACKLIB_COMMON_TRACK_TRACK_C99_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/particles/definitions.h"
    #include "sixtracklib/common/track/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/elem_by_elem_config_defines.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

struct NS(Particles);
struct NS(ElemByElemConfig);
struct NS(Object);

SIXTRL_STATIC SIXTRL_FN NS(track_status_t)
NS(Track_particle_beam_element_obj_dispatcher)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC struct NS(Object) const*
        SIXTRL_RESTRICT be_info );

SIXTRL_STATIC SIXTRL_FN NS(track_status_t)
NS(Track_particle_beam_element_obj_dispatcher_aperture_check)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC struct NS(Object) const*
        SIXTRL_RESTRICT be_info,
    bool const perform_global_aperture_check );

SIXTRL_STATIC SIXTRL_FN NS(track_status_t)
NS(Track_particle_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const struct NS(Object)
        *const SIXTRL_RESTRICT be_obj );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(track_status_t) NS(Track_particle_until_turn_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC struct NS(Object) const*
        SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC struct NS(Object) const*
        SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn );

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_STATIC SIXTRL_FN NS(track_status_t)
NS(Track_particle_element_by_element_until_turn_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const index,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        struct NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC struct NS(Object) const*
        SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC struct NS(Object) const*
        SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn );

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_STATIC SIXTRL_FN NS(track_status_t) NS(Track_particle_line_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC struct NS(Object) const*
        SIXTRL_RESTRICT line_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC struct NS(Object) const*
        SIXTRL_RESTRICT line_end,
    bool const finish_turn );

#if !defined( _GPUCODE )

struct NS(Buffer);

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t) NS(Track_particle_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements_buffer,
    NS(particle_index_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(Track_all_particles_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC const struct
        NS(Buffer) *const SIXTRL_RESTRICT be_buffer,
    NS(particle_index_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(Track_particle_element_by_element_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const struct NS(ElemByElemConfig)
        *const SIXTRL_RESTRICT elem_by_elem_config,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements_buffer,
    NS(particle_index_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(Track_all_particles_element_by_element_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        struct NS(ElemByElemConfig) *const SIXTRL_RESTRICT elem_by_elem_config,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements_buffer,
    NS(particle_index_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t) NS(Track_particle_line)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements_buffer,
    NS(buffer_size_t) const line_begin_idx,
    NS(buffer_size_t) const line_end_idx, bool const finish_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t) NS(Track_all_particles_line)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements_buffer,
    NS(buffer_size_t) const line_begin_idx,
    NS(buffer_size_t) const line_end_idx, bool const finish_turn );

#endif /* !defined( _GPUCODE ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* ****              Inline methods and implementations                ***** */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* #if !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/generated/config.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/beam_elements.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"

    #include "sixtracklib/common/be_drift/track.h"
    #include "sixtracklib/common/be_cavity/track.h"
    #include "sixtracklib/common/be_multipole/track.h"
    #include "sixtracklib/common/be_monitor/track.h"
    #include "sixtracklib/common/be_srotation/track.h"
    #include "sixtracklib/common/be_xyshift/track.h"
    #include "sixtracklib/common/be_limit/track.h"
    #include "sixtracklib/common/be_dipedge/track.h"

    #if !defined( SIXTRL_DISABLE_BEAM_BEAM )
        #include "sixtracklib/common/be_beamfields/track.h"
    #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */


SIXTRL_INLINE NS(track_status_t) NS(Track_particle_beam_element_obj_dispatcher)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info )
{
    #if defined( SIXTRL_ENABLE_APERTURE_CHECK ) && \
        ( defined( SIXTRL_GLOBAL_APERTURE_CHECK_CONDITIONAL ) || \
          defined( SIXTRL_GLOBAL_APERTURE_CHECK_ALWAYS ) || \
          defined( SIXTRL_GLOBAL_APERTURE_CHECK_NEVER ) )

        #if ( SIXTRL_ENABLE_APERTURE_CHECK == \
              SIXTRL_GLOBAL_APERTURE_CHECK_CONDITIONAL ) || \
            ( SIXTRL_ENABLE_APERTURE_CHECK == \
              SIXTRL_GLOBAL_APERTURE_CHECK_ALWAYS )

        return NS(Track_particle_beam_element_obj_dispatcher_aperture_check)(
            particles, index, be_info, true );

        #elif SIXTRL_ENABLE_APERTURE_CHECK == \
              SIXTRL_GLOBAL_APERTURE_CHECK_NEVER

        return NS(Track_particle_beam_element_obj_dispatcher_aperture_check)(
            particles, index, be_info, false );

        #endif /* SIXTRL_ENABLE_APERTURE_CHECK */

    #else  /* !defined( SIXTRL_ENABLE_APERTURE_CHECK ) */

        return NS(Track_particle_beam_element_obj_dispatcher_aperture_check)(
            particles, index, be_info, true );

    #endif /*  defined( SIXTRL_ENABLE_APERTURE_CHECK ) */
}

SIXTRL_INLINE NS(track_status_t)
NS(Track_particle_beam_element_obj_dispatcher_aperture_check)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info,
    bool const perform_global_aperture_check )
{
    typedef NS(buffer_addr_t) address_t;
    typedef NS(object_type_id_t) type_id_t;

    int ret = SIXTRL_TRACK_SUCCESS;
    type_id_t const type_id = NS(Object_get_type_id)( be_info );
    address_t const begin_addr = NS(Object_get_begin_addr)( be_info );

    SIXTRL_ASSERT( begin_addr != ( address_t )0u );
    SIXTRL_ASSERT( particles  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( index >= ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( index < NS(Particles_get_num_of_particles)( particles ) );
    SIXTRL_ASSERT( NS(Particles_is_not_lost_value)( particles, index ) );

    switch( type_id )
    {
        case NS(OBJECT_TYPE_DRIFT):
        {
            typedef NS(Drift)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_drift)( particles, index, belem );

            if( perform_global_aperture_check )
            {
                ret |= NS(Track_particle_limit_global)( particles, index );
            }

            break;
        }

        case NS(OBJECT_TYPE_DRIFT_EXACT):
        {
            typedef NS(DriftExact)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_drift_exact)( particles, index, belem );

            if( perform_global_aperture_check )
            {
                ret |= NS(Track_particle_limit_global)( particles, index );
            }

            break;
        }

        case NS(OBJECT_TYPE_MULTIPOLE):
        {
            typedef NS(MultiPole)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_multipole)( particles, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_CAVITY):
        {
            typedef NS(Cavity)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_cavity)( particles, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_XYSHIFT):
        {
            typedef NS(XYShift)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_xy_shift)( particles, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_SROTATION):
        {
            typedef NS(SRotation)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_srotation)( particles, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_BEAM_MONITOR):
        {
            typedef NS(BeamMonitor)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_beam_monitor)( particles, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_LIMIT_RECT):
        {
            typedef NS(LimitRect) belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_limit_rect)( particles, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_LIMIT_ELLIPSE):
        {
            typedef NS(LimitEllipse) belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_limit_ellipse)( particles, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_DIPEDGE):
        {
            typedef NS(DipoleEdge) belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_dipedge)( particles, index, belem );
            break;
        }

        #if !defined( SIXTRL_DISABLE_BEAM_BEAM )

        case NS(OBJECT_TYPE_BEAM_BEAM_4D):
        {
            typedef NS(BeamBeam4D)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_beam_beam_4d)( particles, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_SPACE_CHARGE_COASTING):
        {
            typedef NS(SpaceChargeCoasting)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_space_charge_coasting)(
                particles, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_SPACE_CHARGE_BUNCHED):
        {
            typedef NS(SpaceChargeBunched)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_space_charge_bunched)(
                particles, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_BEAM_BEAM_6D):
        {
            typedef NS(BeamBeam6D)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_beam_beam_6d)( particles, index, belem );
            break;
        }

        #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */

        default:
        {
            NS(Particles_mark_as_lost_value)( particles, index );
            ret = SIXTRL_TRACK_STATUS_GENERAL_FAILURE;
        }
    };

    return ret;
}

SIXTRL_INLINE NS(track_status_t) NS(Track_particle_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT be_obj )
{
    bool const was_not_lost = NS(Particles_is_not_lost_value)(
        particles, particle_index );

    NS(track_status_t) const status = ( was_not_lost )
        ? NS(Track_particle_beam_element_obj_dispatcher)(
            particles, particle_index, be_obj )
        : SIXTRL_TRACK_SUCCESS;

    if( ( was_not_lost ) && ( status == SIXTRL_TRACK_SUCCESS ) &&
        ( NS(Particles_is_not_lost_value)( particles, particle_index ) ) )
    {
        NS(Particles_increment_at_element_id_value)(
            particles, particle_index );
    }

    return status;
}

SIXTRL_INLINE NS(track_status_t) NS(Track_particle_until_turn_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p,
    NS(particle_num_elements_t) const idx,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn )
{
    typedef NS(particle_index_t) index_t;
    NS(track_status_t) success = (
        ( ( uintptr_t )be_begin ) < ( ( uintptr_t )be_end ) )
        ? SIXTRL_TRACK_SUCCESS : SIXTRL_TRACK_STATUS_GENERAL_FAILURE;

    index_t const start_at_element =
        NS(Particles_get_at_element_id_value)( p, idx );

    bool continue_tracking = ( ( success == SIXTRL_TRACK_SUCCESS ) &&
        ( until_turn > NS(Particles_get_at_turn_value)( p, idx ) ) &&
        ( NS(Particles_is_not_lost_value)( p, idx ) ) );

    SIXTRL_ASSERT( be_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( p != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(Particles_get_num_of_particles)( p ) > idx );
    SIXTRL_ASSERT( ( NS(Particles_is_not_lost_value)( p, idx ) ) ||
                   ( !continue_tracking ) );

    while( continue_tracking )
    {
        SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* be_it = be_begin;

        while( ( continue_tracking ) && ( be_it != be_end ) )
        {
            success = NS(Track_particle_beam_element_obj_dispatcher)(
                p, idx, be_it++ );

            continue_tracking = ( ( success == SIXTRL_TRACK_SUCCESS ) &&
                ( NS(Particles_is_not_lost_value)( p, idx ) ) );

            if( continue_tracking )
            {
                NS(Particles_increment_at_element_id_value)( p, idx );
            }
        }

        if( continue_tracking )
        {
            SIXTRL_ASSERT( be_it == be_end );
            NS(Particles_set_at_element_id_value)( p, idx, start_at_element );
            NS(Particles_increment_at_turn_value)( p, idx );
            NS(Particles_set_s_value)( p, idx, 0.0 );

            continue_tracking =
                ( NS(Particles_get_at_turn_value)( p, idx ) < until_turn );
        }
        else
        {
            NS(Particles_mark_as_lost_value)( p, idx );
        }
    }

    return success;
}

SIXTRL_INLINE NS(track_status_t)
NS(Track_particle_element_by_element_until_turn_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p,
    NS(particle_num_elements_t) const idx,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig)
        *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn )
{
    NS(track_status_t) success = SIXTRL_TRACK_SUCCESS;

    typedef NS(particle_index_t) index_t;
    typedef NS(ParticlesGenericAddr) out_particle_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC out_particle_t* ptr_out_particles_t;

    ptr_out_particles_t out_particles = ( ptr_out_particles_t )(
            uintptr_t )NS(ElemByElemConfig_get_output_store_address)( config );

    SIXTRL_ASSERT( p != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(Particles_get_num_of_particles)( p ) > idx );

    if( out_particles != SIXTRL_NULLPTR )
    {
        typedef NS(particle_num_elements_t) nelem_t;

        index_t const part_id = NS(Particles_get_particle_id_value)( p, idx );
        index_t const start_at_element_id =
            NS(Particles_get_at_element_id_value)( p, idx );

        index_t at_element_id = start_at_element_id;
        index_t at_turn = NS(Particles_get_at_turn_value)( p, idx );

        nelem_t const out_nn = ( out_particles != SIXTRL_NULLPTR )
            ? out_particles->num_particles : ( nelem_t )0u;

        bool continue_tracking = ( ( be_begin != be_end ) &&
            ( until_turn > at_turn ) && ( at_turn >= ( index_t )0u ) &&
            ( start_at_element_id >= ( index_t )0u ) &&
            ( part_id >= ( index_t )0u ) );

        SIXTRL_ASSERT( be_begin != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( ( ( uintptr_t )be_end ) >= ( ( uintptr_t )be_begin ) );
        SIXTRL_ASSERT( ( NS(Particles_is_not_lost_value)( p, idx ) ) ||
                       ( !continue_tracking ) );

        while( continue_tracking )
        {
            SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* be_it = be_begin;

            while( ( continue_tracking ) && ( be_it != be_end ) )
            {
                nelem_t const out_idx =
                    NS(ElemByElemConfig_get_particles_store_index_details)(
                        config, part_id, at_element_id, at_turn );

                if( ( out_idx >= ( nelem_t )0u ) && ( out_idx < out_nn ) )
                {
                    continue_tracking = ( SIXTRL_ARCH_STATUS_SUCCESS ==
                        NS(Particles_copy_to_generic_addr_data)(
                            out_particles, out_idx, p, idx ) );
                }

                if( continue_tracking )
                {
                    success = NS(Track_particle_beam_element_obj_dispatcher)(
                        p, idx, be_it );

                    continue_tracking = (
                        ( success == SIXTRL_TRACK_SUCCESS ) &&
                        ( NS(Particles_is_not_lost_value)( p, idx ) ) );

                    if( continue_tracking )
                    {
                        ++be_it;
                        if( be_it != be_end )
                        {
                            ++at_element_id;
                            NS(Particles_increment_at_element_id_value)(
                                p, idx );
                        }
                        else
                        {
                            NS(Particles_set_at_element_id_value)(
                                p, idx, start_at_element_id );

                            NS(Particles_set_s_value)( p, idx, 0.0 );
                            NS(Particles_increment_at_turn_value)( p, idx );
                            ++at_turn;

                            at_element_id = start_at_element_id;
                            continue_tracking = ( at_turn < until_turn );
                        }
                    }
                }
            }
        }
    }
    else
    {
        success = NS(Track_particle_until_turn_objs)(
            p, idx, be_begin, be_end, until_turn );
    }

    return success;
}

SIXTRL_INLINE NS(track_status_t) NS(Track_particle_line_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT line_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT line_end,
    bool const finish_turn )
{
    NS(track_status_t) success = SIXTRL_TRACK_SUCCESS;
    bool continue_tracking = ( line_it != line_end );

    SIXTRL_ASSERT( line_it != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( ( uintptr_t )line_it ) <= ( ( uintptr_t )line_end ) );
    SIXTRL_ASSERT( NS(Particles_get_num_of_particles)( particles ) > index );
    SIXTRL_ASSERT( ( NS(Particles_is_not_lost_value)( particles, index ) ) ||
                   ( !continue_tracking ) );

    while( continue_tracking )
    {
        success = NS(Track_particle_beam_element_obj_dispatcher)(
            particles, index, line_it );

        continue_tracking = ( ( success == SIXTRL_TRACK_SUCCESS ) &&
            ( NS(Particles_is_not_lost_value)( particles, index ) ) );

        if( continue_tracking )
        {
            NS(Particles_increment_at_element_id_value)( particles, index );
            ++line_it;
            continue_tracking = ( line_it != line_end );
        }
    }

    if( ( finish_turn ) && ( success == SIXTRL_TRACK_SUCCESS ) &&
        ( NS(Particles_is_not_lost_value)( particles, index ) ) )
    {
        NS(Particles_set_at_element_id_value)( particles, index, 0 );
        NS(Particles_increment_at_turn_value)( particles, index );
        NS(Particles_set_s_value)( particles, index, 0.0 );
    }

    return success;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_TRACK_TRACK_C99_H__ */

/* end: sixtracklib/common/track/track.h */
