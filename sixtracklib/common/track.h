#ifndef SIXTRACKLIB_COMMON_TRACK_H__
#define SIXTRACKLIB_COMMON_TRACK_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_FN SIXTRL_STATIC void NS(Track_particle_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    NS(particle_index_t) const start_beam_element_id );

SIXTRL_FN SIXTRL_STATIC void NS(Track_subset_particles_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_begin_index,
    NS(particle_num_elements_t) const particle_end_index,
    NS(particle_num_elements_t) const particle_index_stride,
    NS(particle_index_t) const start_beam_element_id );

SIXTRL_FN SIXTRL_STATIC void NS(Track_all_particles_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const start_beam_element_id );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info );


SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info );


SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_beam_elements_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end );


SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_beam_elements_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end );


SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_beam_elements_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_until_turn_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const end_turn );

SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_until_turn_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const end_turn );

SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_until_turn_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const end_turn );

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_element_by_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    NS(particle_index_t) start_beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT io_obj_begin );

SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_element_by_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT io_obj_begin );

SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_element_by_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT io_obj_begin );

/* ------------------------------------------------------------------------- */

struct NS(Buffer);

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const be_idx );


SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const be_idx );


SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const be_idx );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_idx_begin,
    NS(buffer_size_t) const be_idx_end );


SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_idx_begin,
    NS(buffer_size_t) const be_idx_end );


SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_idx_begin,
    NS(buffer_size_t) const be_idx_end );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements );


SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements );


SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const end_turn );

SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const end_turn );

SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const end_turn );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    NS(particle_index_t) start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT io_particle_buffer,
    NS(buffer_size_t) const io_particle_blocks_offset );

SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT io_particle_buffer,
    NS(buffer_size_t) const io_particle_blocks_offset );

SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT io_particle_buffer,
    NS(buffer_size_t) const io_particle_blocks_offset );

SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_append_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT io_particle_buffer );

#endif /* defined( _GPUCODE ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====        Implementation of Inline functions and methods         ===== */
/* ========================================================================= */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
#endif /* #if !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/generated/config.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/beam_elements.h"
    #include "sixtracklib/common/buffer/buffer_type.h"

    #include "sixtracklib/common/be_drift/track.h"
    #include "sixtracklib/common/be_cavity/track.h"
    #include "sixtracklib/common/be_multipole/track.h"
    #include "sixtracklib/common/be_monitor/track.h"
    #include "sixtracklib/common/be_srotation/track.h"
    #include "sixtracklib/common/be_xyshift/track.h"

    #if !defined( SIXTRL_DISABLE_BEAM_BEAM )
        #include "sixtracklib/common/be_beambeam/track.h"
    #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */


SIXTRL_INLINE void NS(Track_particle_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    NS(particle_index_t) const start_beam_element_id )
{
    if( NS(Particles_get_state_value)( particles, particle_index ) )
    {
        NS(Particles_increment_at_turn_value)(
            particles, particle_index );

        NS(Particles_set_at_element_id_value)(
            particles, particle_index, start_beam_element_id );
    }

    return;
}

SIXTRL_INLINE void NS(Track_subset_particles_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_index,
    NS(particle_num_elements_t) const particle_end_index,
    NS(particle_num_elements_t) const particle_index_stride,
    NS(particle_index_t) const start_beam_element_id )
{
    SIXTRL_ASSERT( particle_index_stride >  ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( particle_index  >= ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( particle_index  <= particle_end_index );

    for( ; particle_index < particle_end_index ;
           particle_index += particle_index_stride )
    {
        if( NS(Particles_get_state_value)( particles, particle_index ) )
        {
            NS(Particles_increment_at_turn_value)(
                particles, particle_index );

            NS(Particles_set_at_element_id_value)(
                particles, particle_index, start_beam_element_id );
        }
    }

    return;
}

SIXTRL_INLINE void NS(Track_all_particles_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const start_beam_element_id )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t ii = ( num_elem_t )0u;
    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    for( ; ii < num_particles ; ++ii )
    {
        if( NS(Particles_get_state_value)( particles, ii ) )
        {
            NS(Particles_increment_at_turn_value)( particles, ii );

            NS(Particles_set_at_element_id_value)(
                particles, ii, start_beam_element_id );
        }
    }

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Track_particle_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info )
{
    typedef NS(object_type_id_t) type_id_t;
    typedef NS(buffer_addr_t)    address_t;
    typedef NS(particle_index_t) index_t;

    int ret = 0;

    type_id_t const    type_id = NS(Object_get_type_id)( be_info );
    address_t const begin_addr = NS(Object_get_begin_addr)( be_info );

    SIXTRL_ASSERT( begin_addr != ( address_t )0u );
    SIXTRL_ASSERT( particle_index >= ( NS(particle_num_elements_t) )0 );

    SIXTRL_ASSERT( particle_index <
                   NS(Particles_get_num_of_particles)( particles ) );

    SIXTRL_ASSERT( NS(Particles_get_state_value)( particles, particle_index ) ==
                   ( index_t )1 );

    ( void )index_t;

    switch( type_id )
    {
        case NS(OBJECT_TYPE_DRIFT):
        {
            typedef NS(Drift)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            #if defined( SIXTRL_ENABLE_APERATURE_CHECK ) && \
                       ( SIXTRL_ENABLE_APERATURE_CHECK == 1 )

            NS(Particles_set_at_element_id_value)(
                particles, particle_index, beam_element_id );
            #endif /* SIXTRL_ENABLE_APERATURE_CHECK  */

            ret = NS(Track_particle_drift)( particles, particle_index, belem );
            break;
        }

        case NS(OBJECT_TYPE_DRIFT_EXACT):
        {
            typedef NS(DriftExact)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            #if defined( SIXTRL_ENABLE_APERATURE_CHECK ) && \
                       ( SIXTRL_ENABLE_APERATURE_CHECK == 1 )

            NS(Particles_set_at_element_id_value)(
                particles, particle_index, beam_element_id );
            #endif /* SIXTRL_ENABLE_APERATURE_CHECK  */

            ret = NS(Track_particle_drift_exact)(
                particles, particle_index, belem );

            break;
        }

        case NS(OBJECT_TYPE_MULTIPOLE):
        {
            typedef NS(MultiPole)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_multipole)( particles, particle_index, belem );
            break;
        }

        case NS(OBJECT_TYPE_CAVITY):
        {
            typedef NS(Cavity)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_cavity)( particles, particle_index, belem );
            break;
        }

        case NS(OBJECT_TYPE_XYSHIFT):
        {
            typedef NS(XYShift)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_xy_shift)( particles, particle_index, belem );
            break;
        }

        case NS(OBJECT_TYPE_SROTATION):
        {
            typedef NS(SRotation)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_srotation)( particles, particle_index, belem );
            break;
        }

        case NS(OBJECT_TYPE_BEAM_MONITOR):
        {
            typedef NS(BeamMonitor)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            NS(Particles_set_at_element_id_value)(
                particles, particle_index, beam_element_id );

            ret = NS(Track_particle_beam_monitor)(
                particles, particle_index, belem );

            break;
        }

        #if !defined( SIXTRL_DISABLE_BEAM_BEAM )

        case NS(OBJECT_TYPE_BEAM_BEAM_4D):
        {
            typedef NS(BeamBeam4D)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_beam_beam_4d)(
                particles, particle_index, belem );
            break;
        }

        case NS(OBJECT_TYPE_BEAM_BEAM_6D):
        {
            typedef NS(BeamBeam6D)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_beam_beam_6d)(
                particles, particle_index, belem );
            break;
        }

        #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */

        default:
        {
            NS(Particles_set_state_value)( particles, particle_index, 0 );
            NS(Particles_set_at_element_id_value)(
                particles, particle_index, beam_element_id );

            ret = -8;
        }
    };

    return ret;
}


SIXTRL_INLINE int NS(Track_subset_of_particles_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info )
{
    typedef NS(particle_index_t) index_t;

    SIXTRL_ASSERT( particle_idx_stride >  ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( particle_idx        >= ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( particle_idx_end    >= particle_idx );

    for( ; particle_idx < particle_idx_end ; particle_idx += particle_idx_stride )
    {
        if( NS(Particles_get_state_value)( particles, particle_idx ) == ( index_t )1 )
        {
            NS(Track_particle_beam_element_obj)(
                particles, particle_idx, beam_element_id, be_info );
        }
    }

    return 0;
}

SIXTRL_INLINE int NS(Track_all_particles_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info )
{
    return NS(Track_subset_of_particles_beam_element_obj)(
        particles, 0u, NS(Particles_get_num_of_particles)( particles ), 1u,
            beam_element_id, be_info );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Track_particle_beam_elements_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end )
{
    typedef NS(particle_index_t)  index_t;

    SIXTRL_STATIC_VAR index_t const LOST_STATE = ( index_t )0u;

    SIXTRL_ASSERT( be_it  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_end != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( ( uintptr_t )be_it ) <= ( ( uintptr_t )be_end ) );

    if( NS(Particles_get_state_value)( particles, particle_idx ) != LOST_STATE )
    {
        int ret = 0;

        NS(particle_index_t) beam_element_id =
            NS(Particles_get_at_element_id_value)( particles, particle_idx );

        while( ( ret == 0 ) && ( be_it != be_end ) )
        {
            ret = NS(Track_particle_beam_element_obj)(
                particles, particle_idx, beam_element_id++, be_it++ );
        }

        if( NS(Particles_get_state_value)(
                particles, particle_idx ) != LOST_STATE )
        {
            NS(Particles_set_at_element_id_value)(
                particles, particle_idx, beam_element_id );
        }

        SIXTRL_ASSERT( ( ret == 0 ) ||
            ( NS(Particles_get_state_value)( particles, particle_idx ) !=
              ( index_t )1 ) );
    }

    return 0;
}


SIXTRL_INLINE int NS(Track_subset_of_particles_beam_elements_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) index,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end )
{
    SIXTRL_ASSERT( begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( end   != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( ( uintptr_t )begin ) <= ( ( uintptr_t )end ) );

    SIXTRL_ASSERT( particle_idx_stride > ( NS(particle_num_elements_t) )0u );
    SIXTRL_ASSERT( index >= ( NS(particle_num_elements_t) )0u );
    SIXTRL_ASSERT( index <= particle_idx_end );

    for( ; index < particle_idx_end ; index += particle_idx_stride )
    {
        NS(Track_particle_beam_elements_obj)( particles, index, begin, end );
    }

    return 0;
}


SIXTRL_INLINE int NS(Track_all_particles_beam_elements_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end )
{
    return NS(Track_subset_of_particles_beam_elements_obj)( particles, 0u,
        NS(Particles_get_num_of_particles)( particles ), 1u, begin, end );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Track_particle_until_turn_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const end_turn )
{
    typedef NS(particle_index_t) index_t;

    SIXTRL_STATIC_VAR index_t const LOST_STATE = ( index_t )0;

    index_t state = NS(Particles_get_state_value)( particles, index );
    index_t turn  = NS(Particles_get_at_turn_value)( particles, index );

    index_t const start_beam_element_id =
        NS(Particles_get_at_element_id_value)( particles, index );

    SIXTRL_ASSERT( be_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_end   != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( ( uintptr_t )be_begin ) <= ( ( uintptr_t )be_end ) );

    while( ( state != LOST_STATE ) && ( turn < end_turn ) )
    {
        SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* be_it = be_begin;
        index_t beam_element_id = start_beam_element_id;

        for( ; be_it != be_end ; ++be_it, ++beam_element_id )
        {
            int const ret = NS(Track_particle_beam_element_obj)(
                    particles, index, beam_element_id, be_it );

            if( 0 != ret )
            {
                state = NS(Particles_get_state_value)( particles, index );
                SIXTRL_ASSERT( state == LOST_STATE );
                break;
            }
        }

        if( state != LOST_STATE )
        {
            ++turn;

            NS(Particles_set_at_element_id_value)(
                particles, index, start_beam_element_id );

            NS(Particles_set_at_turn_value)( particles, index, turn );
        }
    }

    return 0;
}

SIXTRL_INLINE int NS(Track_subset_of_particles_until_turn_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) index,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const end_turn )
{
    SIXTRL_ASSERT( index >= ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( particle_idx_stride > ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( index <= particle_idx_end );

    for( ; index < particle_idx_end ; index += particle_idx_stride )
    {
        NS(Track_particle_until_turn_obj)(
            particles, index, be_begin, be_end, end_turn );
    }

    return 0;
}

SIXTRL_INLINE int NS(Track_all_particles_until_turn_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const end_turn )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const end_idx = NS(Particles_get_num_of_particles)( particles );
    num_elem_t ii = ( num_elem_t )0u;

    for( ; ii < end_idx ; ++ii )
    {
        NS(Track_particle_until_turn_obj)(
            particles, ii, be_begin, be_end, end_turn );
    }

    return 0;
}

#if !defined( _GPUCODE )

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Track_particle_element_by_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    NS(particle_index_t) beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT io_obj_it )
{
    typedef NS(particle_index_t) index_t;

    int ret = 0;

    SIXTRL_ASSERT( be_it     != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_end    != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( ( uintptr_t )be_it ) <= ( ( uintptr_t )be_end ) );
    SIXTRL_ASSERT( io_obj_it != SIXTRL_NULLPTR );

    for( ; be_it != be_end ; ++be_it, ++io_obj_it )
    {
        SIXTRL_BUFFER_OBJ_DATAPTR_DEC NS(Particles)* io_particles =
            NS(BufferIndex_get_particles)( io_obj_it );

        if( io_particles != SIXTRL_NULLPTR )

        {
            NS(Particles_copy_single)( io_particles, particle_index,
                                       particles, particle_index );

            if( NS(Particles_get_state_value)( io_particles, particle_index ) ==
                ( index_t )1 )
            {
                NS(Particles_set_at_element_id_value)(
                    io_particles, particle_index, beam_element_id );
            }
        }
        else
        {
            NS(Particles_set_at_element_id_value)(
                particles, particle_index, beam_element_id );

            NS(Particles_set_state_value)(
                particles, particle_index, ( NS(particle_index_t) )0u );

            ret = -1;
            break;
        }

        if( 0 != NS(Track_particle_beam_element_obj)(
                particles, particle_index, beam_element_id++, be_it ) )
        {
            break;
        }
    }

    return ret;
}

SIXTRL_INLINE int NS(Track_subset_of_particles_element_by_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT io_obj_begin )
{
    SIXTRL_ASSERT( particle_idx_stride > ( NS(particle_num_elements_t) )0u );
    SIXTRL_ASSERT( particle_idx >= ( NS(particle_num_elements_t) )0u );
    SIXTRL_ASSERT( particle_idx <= particle_idx_end );

    for( ; particle_idx < particle_idx_end ; particle_idx += particle_idx_stride )
    {
        NS(Track_particle_element_by_element_obj)(
            particles, particle_idx, start_beam_element_id,
            be_begin, be_end, io_obj_begin );
    }

    return 0;
}

SIXTRL_INLINE int NS(Track_all_particles_element_by_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT io_obj_begin )
{
    return NS(Track_subset_of_particles_element_by_element_obj)(
        particles, 0u, NS(Particles_get_num_of_particles)( particles ), 1u,
        start_beam_element_id, be_begin, be_end, io_obj_begin );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Track_particle_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const be_idx )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;
    ptr_obj_t ptr_be_obj = NS(Buffer_get_const_objects_begin)( beam_elements );

    SIXTRL_ASSERT( ptr_be_obj != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( NS(buffer_size_t) )be_idx <
                     NS(Buffer_get_num_of_objects)( beam_elements ) );

    return NS(Track_particle_beam_element_obj)(
        particles, particle_idx, beam_element_id, ptr_be_obj + be_idx );
}


SIXTRL_INLINE int
NS(Track_subset_of_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const be_idx )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;
    ptr_obj_t ptr_be_obj = NS(Buffer_get_const_objects_begin)( beam_elements );

    SIXTRL_ASSERT( ptr_be_obj != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( NS(buffer_size_t) )be_idx <
                   NS(Buffer_get_num_of_objects)( beam_elements ) );

    return NS(Track_subset_of_particles_beam_element_obj)(
        particles, particle_idx, particle_idx_end, particle_idx_stride,
            beam_element_id, ptr_be_obj + be_idx );
}


SIXTRL_INLINE int
NS(Track_all_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const be_idx )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;
    ptr_obj_t ptr_be_obj = NS(Buffer_get_const_objects_begin)( beam_elements );

    SIXTRL_ASSERT( ptr_be_obj != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( NS(buffer_size_t) )be_idx <
                   NS(Buffer_get_num_of_objects)( beam_elements ) );

    return NS(Track_all_particles_beam_element_obj)(
        particles, beam_element_id, ptr_be_obj + be_idx );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int
NS(Track_particle_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_begin_idx, NS(buffer_size_t) const be_end_idx )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;
    ptr_obj_t ptr_be_obj = NS(Buffer_get_const_objects_begin)( beam_elements );

    SIXTRL_ASSERT( ptr_be_obj != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_begin_idx <= be_end_idx );
    SIXTRL_ASSERT( be_end_idx < NS(Buffer_get_num_of_objects)( beam_elements ) );

    return NS(Track_particle_beam_elements_obj)( particles, particle_idx,
        ptr_be_obj + be_begin_idx, ptr_be_obj + be_end_idx );
}


SIXTRL_INLINE int
NS(Track_subset_of_particles_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_begin_idx, NS(buffer_size_t) const be_end_idx )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;
    ptr_obj_t ptr_be_obj = NS(Buffer_get_const_objects_begin)( beam_elements );

    SIXTRL_ASSERT( ptr_be_obj != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_begin_idx <= be_end_idx );
    SIXTRL_ASSERT( be_end_idx < NS(Buffer_get_num_of_objects)( beam_elements ) );

    return NS(Track_subset_of_particles_beam_elements_obj)(
        particles, particle_idx_begin, particle_idx_end, particle_idx_stride,
        ptr_be_obj + be_begin_idx, ptr_be_obj + be_end_idx );
}


SIXTRL_INLINE int
NS(Track_all_particles_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_begin_idx, NS(buffer_size_t) const be_end_idx )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;
    ptr_obj_t ptr_be_obj = NS(Buffer_get_const_objects_begin)( beam_elements );

    SIXTRL_ASSERT( ptr_be_obj != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_begin_idx <= be_end_idx );
    SIXTRL_ASSERT( be_end_idx < NS(Buffer_get_num_of_objects)( beam_elements ) );

    return NS(Track_all_particles_beam_elements_obj)( particles,
        ptr_be_obj + be_begin_idx, ptr_be_obj + be_end_idx );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Track_particle_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements )
{
    return NS(Track_particle_beam_elements_obj)(
        particles, particle_index,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ) );
}


SIXTRL_INLINE int NS(Track_subset_of_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements )
{
    return NS(Track_subset_of_particles_beam_elements_obj)(
        particles, particle_idx_begin, particle_idx_end, particle_idx_stride,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ) );
}


SIXTRL_INLINE int NS(Track_all_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements )
{
    return NS(Track_all_particles_beam_elements_obj)( particles,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ) );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Track_particle_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const end_turn )
{
    return NS(Track_particle_until_turn_obj)(
        particles, particle_index,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ), end_turn );
}

SIXTRL_INLINE int NS(Track_subset_of_particles_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const end_turn )
{
    return NS(Track_subset_of_particles_until_turn_obj)(
        particles, particle_idx_begin, particle_idx_end, particle_idx_stride,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ), end_turn );
}

SIXTRL_INLINE int NS(Track_all_particles_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const end_turn )
{
    return NS(Track_all_particles_until_turn_obj)( particles,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ), end_turn );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Track_particle_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT io_particle_buffer,
    NS(buffer_size_t) const io_particle_blocks_offset )
{
    int ret = -1;

    if( ( io_particle_blocks_offset +
          NS(Buffer_get_num_of_objects)( io_particle_buffer ) ) >=
        NS(Particles_buffer_get_num_of_particle_blocks)( io_particle_buffer ) )
    {
        ret = NS(Track_particle_element_by_element_obj)(
            particles, particle_idx, start_beam_element_id,
            NS(Buffer_get_const_objects_begin)( beam_elements ),
            NS(Buffer_get_const_objects_end)( beam_elements ),
            NS(Buffer_get_object)( io_particle_buffer, io_particle_blocks_offset ) );
    }

    return ret;
}

SIXTRL_INLINE int NS(Track_subset_of_particles_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT io_particle_buffer,
    NS(buffer_size_t) const io_particle_blocks_offset )
{
    return NS(Track_subset_of_particles_element_by_element_obj)(
        particles, particle_idx_begin, particle_idx_end, particle_idx_stride,
        start_beam_element_id,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ),
        NS(Buffer_get_object)( io_particle_buffer, io_particle_blocks_offset ) );
}

SIXTRL_INLINE int NS(Track_all_particles_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT io_particle_buffer,
    NS(buffer_size_t) const io_particle_blocks_offset )
{
    return NS(Track_subset_of_particles_element_by_element)(
        particles, 0u, NS(Particles_get_num_of_particles)( particles ), 1u,
            start_beam_element_id, beam_elements,
                io_particle_buffer, io_particle_blocks_offset );
}

SIXTRL_INLINE int NS(Track_all_particles_append_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT io_particle_buffer )
{
    typedef NS(particle_num_elements_t) num_elements_t;
    typedef NS(particle_index_t)        index_t;

    int ret = 0;

    NS(Object) const* obj_it  = NS(Buffer_get_const_objects_begin)( beam_elements );
    NS(Object) const* obj_end = NS(Buffer_get_const_objects_end)( beam_elements );

    for( ; obj_it != obj_end ; ++obj_it, ++beam_element_id )
    {
        NS(Particles)* elem_by_elem_dump = NS(Particles_add_copy)(
                io_particle_buffer, particles );

        if( elem_by_elem_dump != SIXTRL_NULLPTR )
        {
            num_elements_t const NUM_PARTICLES =
                NS(Particles_get_num_of_particles)( particles );

            num_elements_t ii = ( num_elements_t )0u;

            for( ; ii < NUM_PARTICLES ; ++ii )
            {
                if( NS(Particles_get_state_value)( particles, ii ) == ( index_t )1 )
                {
                    NS(Particles_set_at_element_id_value)(
                        particles, ii, beam_element_id );
                }
            }
        }
        else
        {
            ret = -1;
            break;
        }

        if( 0 != NS(Track_all_particles_beam_element_obj)(
                particles, beam_element_id, obj_it ) )
        {
            break;
        }
    }

    return ret;
}

/* ------------------------------------------------------------------------- */

#endif /* !defined( _GPUCODE ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_TRACK_H__ */

/* end: sixtracklib/common/track.h */
