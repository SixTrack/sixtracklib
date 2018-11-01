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
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Track_subset_particles_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_begin_index,
    NS(particle_num_elements_t) const particle_end_index,
    NS(particle_num_elements_t) const particle_index_stride );

SIXTRL_FN SIXTRL_STATIC void NS(Track_all_particles_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

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
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end );


SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_beam_elements_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end );


SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_beam_elements_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end );

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

struct NS(Buffer);

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const beam_index );


SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const beam_index );


SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const beam_index );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_beam_elements_subset)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const beam_index_begin,
    NS(particle_index_t) const beam_index_end );


SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_beam_elements_subset)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const beam_index_begin,
    NS(particle_index_t) const beam_index_end );


SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_beam_elements_subset)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const beam_index_begin,
    NS(particle_index_t) const beam_index_end );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements );


SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements );


SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements );

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
    #include "sixtracklib/common/be_xy_shift/track.h"

    #if !defined( SIXTRL_DISABLE_BEAM_BEAM )
        #include "sixtracklib/common/be_beambeam/track_beambeam.h"
    #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */


SIXTRL_FN SIXTRL_STATIC void NS(Track_particle_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index )
{
    if( NS(Particles_get_state_value)( particles, particle_index ) )
    {
        NS(Particles_increment_at_turn_value)( particles, particle_index );
    }

    return;
}

SIXTRL_FN SIXTRL_STATIC void NS(Track_subset_particles_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    NS(particle_num_elements_t) const particle_end_index,
    NS(particle_num_elements_t) const particle_index_stride )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    SIXTRL_ASSERT( particle_index_stride >  ( num_elem_t )0 );
    SIXTRL_ASSERT( particle_index  >= ( num_elem_t )0 );
    SIXTRL_ASSERT( particle_index  <= particle_end_index );

    for( ; particle_index < particle_end_index ;
           particle_index += particle_index_stride )
    {
        if( NS(Particles_get_state_value)( particles, particle_index ) )
        {
            NS(Particles_increment_at_turn_value)( particles, particle_index );
        }
    }

    return;
}

SIXTRL_FN SIXTRL_STATIC void NS(Track_all_particles_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t ii = ( num_elem_t )0u;
    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( paricles );

    for( ; ii < num_particles ; ++ii )
    {
        if( NS(Particles_get_state_value)( particles, ii ) )
        {
            NS(Particles_increment_at_turn_value)( particles, ii );
        }
    }

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info )
{
    typedef NS(object_type_id_t)    type_id_t;
    typedef NS(buffer_addr_t)       address_t;

    SIXTRL_TRACK_RETURN ret = ( SIXTRL_TRACK_RETURN )0;

    type_id_t const    type_id = NS(Object_get_type_id)( be_info );
    address_t const begin_addr = NS(Object_get_begin_addr)( be_info );

    SIXTRL_ASSERT( begin_addr != ( address_t )0u );
    SIXTRL_ASSERT( index >= ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( index <  NS(Particles_get_num_of_particles)( p ) );

    #if defined( SIXTRL_ENABLE_APERATURE_CHECK ) && \
        ( SIXTRL_ENABLE_APERATURE_CHECK == 1 )

    SIXTRL_ASSERT( NS(Particles_get_state_value)( p, index ) == 1 );

    #endif /* SIXTRL_ENABLE_APERATURE_CHECK  */

    switch( type_id )
    {
        case NS(OBJECT_TYPE_DRIFT):
        {
            typedef NS(Drift)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            #if defined( SIXTRL_ENABLE_APERATURE_CHECK ) && \
                       ( SIXTRL_ENABLE_APERATURE_CHECK == 1 )

            NS(Particles_set_at_element_value)( p, index, beam_element_id );

            #endif /* SIXTRL_ENABLE_APERATURE_CHECK  */

            ret = NS(Track_particle_drift)( p, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_DRIFT_EXACT):
        {
            typedef NS(DriftExact)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            #if defined( SIXTRL_ENABLE_APERATURE_CHECK ) && \
                       ( SIXTRL_ENABLE_APERATURE_CHECK == 1 )

            NS(Particles_set_at_element_value)( p, index, beam_element_id );

            #endif /* SIXTRL_ENABLE_APERATURE_CHECK  */

            ret = NS(Track_particle_drift_exact)( p, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_MULTIPOLE):
        {
            typedef NS(MultiPole)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_multipole)( p, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_CAVITY):
        {
            typedef NS(Cavity)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_cavity)( p, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_XYSHIFT):
        {
            typedef NS(XYShift)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_xy_shift)( p, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_SROTATION):
        {
            typedef NS(SRotation)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_srotation)( p, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_BEAM_MONITOR):
        {
            typedef NS(BeamMonitor)   BeamMonitor;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            NS(Particles_set_at_element_value)( p, index, beam_element_id );
            ret = NS(Track_particle_monitor)( p, index, belem );
            break;
        }

        #if !defined( SIXTRL_DISABLE_BEAM_BEAM )

        case NS(OBJECT_TYPE_BEAM_BEAM_4D):
        {
            typedef NS(BeamBeam4D)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_beam_beam_4d)( p, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_BEAM_BEAM_6D):
        {
            typedef NS(BeamBeam6D)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_beam_beam_6d)( p, index, belem );
            break;
        }

        #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */

        default:
        {
            ret = -8;
        }
    };

    return ret;
}


SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_subset_of_particles_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info )
{

}


SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_all_particles_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info )
{

}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_TRACK_RETURN
NS(Track_particle_beam_elements_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end );


SIXTRL_INLINE SIXTRL_TRACK_RETURN
NS(Track_subset_of_particles_beam_elements_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end );


SIXTRL_INLINE SIXTRL_TRACK_RETURN
NS(Track_all_particles_beam_elements_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end );

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

struct NS(Buffer);

SIXTRL_INLINE SIXTRL_TRACK_RETURN
NS(Track_particle_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const beam_index );


SIXTRL_INLINE SIXTRL_TRACK_RETURN
NS(Track_subset_of_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const beam_index );


SIXTRL_INLINE SIXTRL_TRACK_RETURN
NS(Track_all_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const beam_index );

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_TRACK_RETURN
NS(Track_particle_beam_elements_subset)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const beam_index_begin,
    NS(particle_index_t) const beam_index_end );


SIXTRL_INLINE SIXTRL_TRACK_RETURN
NS(Track_subset_of_particles_beam_elements_subset)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const beam_index_begin,
    NS(particle_index_t) const beam_index_end );


SIXTRL_INLINE SIXTRL_TRACK_RETURN
NS(Track_all_particles_beam_elements_subset)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const beam_index_begin,
    NS(particle_index_t) const beam_index_end );

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_TRACK_RETURN
NS(Track_particle_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements );


SIXTRL_INLINE SIXTRL_TRACK_RETURN
NS(Track_subset_of_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements );


SIXTRL_INLINE SIXTRL_TRACK_RETURN
NS(Track_all_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements );





SIXTRL_INLINE SIXTRL_TRACK_RETURN
NS(Track_particle_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p,
    NS(particle_num_elements_t) index,
    NS(particle_index_t) const beam_element_index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info )
{

}


SIXTRL_INLINE SIXTRL_TRACK_RETURN
NS(Track_particle_beam_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_index_t) start_beam_element_index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end )
{
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* be_it = be_begin;
    SIXTRL_TRACK_RETURN ret = ( SIXTRL_TRACK_RETURN )0;

    for( ; be_it != be_end ; ++be_it, ++start_beam_element_index )
    {
        ret |= NS(Track_particle_beam_element_obj)(
            particles, ii, start_beam_element_index, be_it );
    }

    return ret;
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_subset_beam_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p,
    NS(particle_num_elements_t) p_index_begin,
    NS(particle_num_elements_t) const p_index_end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info_end )
{
    SIXTRL_TRACK_RETURN ret = ( SIXTRL_TRACK_RETURN )0;

    SIXTRL_ASSERT( ( ( uintptr_t )be_info_end ) >= ( uintptr_t )be_info_it );

    SIXTRL_ASSERT( ( be_info_it != SIXTRL_NULLPTR ) ||
                   ( be_info_it == SIXTRL_NULLPTR ) );

    for( ; be_info_it != be_info_end ; ++be_info_it )
    {
        ret |= NS(Track_particle_subset_beam_element_obj)(
            p, p_index_begin, p_index_end, be_info_it );
    }

    return ret;
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particles_beam_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info_end )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    return NS(Track_particle_subset_beam_element_objs)( p,
        ( num_elem_t )0u, NS(Particles_get_num_of_particles)( p ),
        be_info_begin, be_info_end );
}

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

struct NS(Buffer);

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const beam_elements )
{
    return NS(Track_particle_subset_beam_element_objs)(
        particles, ( NS(particle_num_elements_t) )0u,
        NS(Particles_get_num_of_particles)( particles ),
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ) );
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const beam_elements,
    NS(buffer_size_t) const beam_element_index )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    ptr_obj_t ptr_be_obj = NS(Buffer_get_const_objects_begin)( beam_elements );

    SIXTRL_ASSERT( ptr_be_obj != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( beam_element_index <
        NS(Buffer_get_num_of_objects)( beam_elements ) );

    ptr_be_obj = ptr_be_obj + beam_element_index;

    return NS(Track_particle_subset_beam_element_obj)(
        particles, ( NS(particle_num_elements_t) )0u,
        NS(Particles_get_num_of_particles)( particles ), ptr_be_obj );
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const beam_element_index )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    ptr_obj_t ptr_be_obj = NS(Buffer_get_const_objects_begin)( beam_elements );

    SIXTRL_ASSERT( ptr_be_obj != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( beam_element_index <
        NS(Buffer_get_num_of_objects)( beam_elements ) );

    ptr_be_obj = ptr_be_obj + beam_element_index;

    return NS(Track_particle_beam_element_obj)(
        particles, particle_index, ptr_be_obj );
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_beam_element_subset)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const beam_element_begin_index,
    NS(buffer_size_t) const beam_element_end_index )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    ptr_obj_t be_end   = SIXTRL_NULLPTR;
    ptr_obj_t be_begin = NS(Buffer_get_const_objects_begin)( beam_elements );

    SIXTRL_ASSERT( be_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( beam_element_begin_index <= beam_element_end_index );
    SIXTRL_ASSERT( beam_element_end_index <=
                   NS(Buffer_get_num_of_objects)( beam_elements ) );

    be_end   = be_begin + beam_element_end_index;
    be_begin = be_begin + beam_element_begin_index;

    return NS(Track_particle_beam_element_objs)(
        particles, particle_index, be_begin, be_end );
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements )
{
    return NS(Track_particle_beam_element_objs)(
        particles, particle_index,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ) );

}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_subset_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_begin_index,
    NS(particle_num_elements_t) const particle_end_index,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const beam_element_index )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    ptr_obj_t ptr_be_obj = NS(Buffer_get_const_objects_begin)( beam_elements );

    SIXTRL_ASSERT( ptr_be_obj != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( beam_element_index <
        NS(Buffer_get_num_of_objects)( beam_elements ) );

    ptr_be_obj = ptr_be_obj + beam_element_index;

    return NS(Track_particle_subset_beam_element_obj)(
        particles, particle_begin_index, particle_end_index, ptr_be_obj );
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_subset_beam_element_subset)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_begin_index,
    NS(particle_num_elements_t) const particle_end_index,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const beam_element_begin_index,
    NS(buffer_size_t) const beam_element_end_index )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    ptr_obj_t be_end   = SIXTRL_NULLPTR;
    ptr_obj_t be_begin = NS(Buffer_get_const_objects_begin)( beam_elements );

    SIXTRL_ASSERT( be_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( beam_element_begin_index <= beam_element_end_index );
    SIXTRL_ASSERT( beam_element_end_index <=
                   NS(Buffer_get_num_of_objects)( beam_elements ) );

    be_end   = be_begin + beam_element_end_index;
    be_begin = be_begin + beam_element_begin_index;

    return NS(Track_particle_subset_beam_element_objs)(
        particles, particle_begin_index, particle_end_index, be_begin, be_end );
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_subset_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_begin_index,
    NS(particle_num_elements_t) const particle_end_index,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements )
{
    return NS(Track_particle_subset_beam_element_objs)(
        particles, particle_begin_index, particle_end_index,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ) );
}

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_TRACK_H__ */

/* end: sixtracklib/common/track.h */
