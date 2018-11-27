#ifndef SIXTRACKLIB_COMMON_BE_MONITOR_TRACK_MONITOR_C99_HEADER_H__
#define SIXTRACKLIB_COMMON_BE_MONITOR_TRACK_MONITOR_C99_HEADER_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

struct NS(BeamMonitor);

SIXTRL_FN SIXTRL_STATIC  NS(particle_num_elements_t)
NS(BeamMonitor_get_store_particle_index)(
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamMonitor)
        *const SIXTRL_RESTRICT beam_monitor,
    NS(be_monitor_turn_t) const at_turn_number,
    NS(particle_index_t)  const particle_id );

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_beam_monitor)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamMonitor)
        *const SIXTRL_RESTRICT monitor );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====        Implementation of Inline functions and methods         ===== */
/* ========================================================================= */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_monitor/be_monitor.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE NS(particle_num_elements_t)
NS(BeamMonitor_get_store_particle_index)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor,
    NS(be_monitor_turn_t) const at_turn_number,
    NS(particle_index_t)  const in_particle_id )
{
    typedef NS(be_monitor_turn_t)       nturn_t;
    typedef NS(be_monitor_index_t)      index_t;
    typedef NS(particle_num_elements_t) num_elements_t;

    SIXTRL_STATIC_VAR nturn_t const ZERO_TURNS = ( nturn_t )0u;

    num_elements_t out_particle_id = ( num_elements_t )-1;

    nturn_t const monitor_start = NS(BeamMonitor_get_start)( monitor );
    nturn_t const num_stores = NS(BeamMonitor_get_num_stores)( monitor );
    nturn_t const skip = NS(BeamMonitor_get_skip)( monitor );
    nturn_t turns_since_start = ZERO_TURNS;

    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( num_stores > ZERO_TURNS );
    SIXTRL_ASSERT( skip >= ( nturn_t )1u );

    SIXTRL_ASSERT( in_particle_id >= ( index_t
        )NS(BeamMonitor_get_min_particle_id)( monitor ) );

    SIXTRL_ASSERT( in_particle_id <= ( index_t
        )NS(BeamMonitor_get_max_particle_id)( monitor ) );

    turns_since_start = at_turn_number - monitor_start;

    if( (   turns_since_start >= ZERO_TURNS ) &&
        ( ( turns_since_start % skip ) == ZERO_TURNS ) )
    {
        nturn_t store_idx = turns_since_start / skip;

        if( ( store_idx >= num_stores ) &&
            ( NS(BeamMonitor_is_rolling)( monitor ) ) )
        {
            store_idx = store_idx % num_stores;
        }

        if( store_idx < num_stores )
        {
            num_elements_t const particle_id_offset = ( num_elements_t )(
                in_particle_id - NS(BeamMonitor_get_min_particle_id)( monitor ) );

            num_elements_t const num_particles_to_store = ( num_elements_t )(
                    NS(BeamMonitor_get_max_particle_id)( monitor ) -
                    NS(BeamMonitor_get_min_particle_id)( monitor ) +
                        ( num_elements_t )1u );

            if( NS(BeamMonitor_is_turn_ordered)( monitor ) )
            {
                out_particle_id  = num_particles_to_store * ( num_elements_t )store_idx;
                out_particle_id += particle_id_offset;
            }
            else
            {
                out_particle_id  = particle_id_offset * num_particles_to_store;
                out_particle_id += ( num_elements_t )store_idx;
            }
        }
    }

    return out_particle_id;
}

SIXTRL_INLINE int NS(Track_particle_beam_monitor)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT in_particles,
    NS(particle_num_elements_t) const idx,
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamMonitor) *const monitor )
{
    typedef NS(be_monitor_turn_t)                       nturn_t;
    typedef NS(be_monitor_addr_t)                       addr_t;
    typedef NS(particle_num_elements_t)                 num_elements_t;
    typedef NS(ParticlesGenericAddr)                    out_particle_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC out_particle_t*   ptr_out_particles_t;
    typedef NS(particle_index_t)                        index_t;

    int success = 0;

    /* Calculate destination index in the io particles object: */

    nturn_t const turn = ( nturn_t
        )NS(Particles_get_at_turn_value)( in_particles, idx );

    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(BeamMonitor_get_skip)( monitor ) > ( nturn_t )0u  );
    SIXTRL_ASSERT( NS(BeamMonitor_get_num_stores)( monitor ) > ( nturn_t)0u );

    SIXTRL_ASSERT( in_particles != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( NS(BeamMonitor_get_min_particle_id)( monitor ) <=
                   NS(BeamMonitor_get_max_particle_id)( monitor ) );

    SIXTRL_ASSERT( NS(Particles_get_particle_id_value)( in_particles, idx ) >=
                   NS(BeamMonitor_get_min_particle_id)( monitor ) );

    SIXTRL_ASSERT( NS(Particles_get_particle_id_value)( in_particles, idx ) <=
                   NS(BeamMonitor_get_max_particle_id)( monitor ) );

    SIXTRL_ASSERT( NS(Particles_get_state_value)( in_particles, idx ) ==
                   ( NS(particle_index_t) )1u );

    if( NS(BeamMonitor_get_out_address)( monitor ) != ( addr_t )0 )
    {
        index_t const particle_id = NS(Particles_get_particle_id_value)(
            in_particles, idx );

        num_elements_t const out_particle_id =
            NS(BeamMonitor_get_store_particle_index)( monitor, turn, particle_id );

        if( out_particle_id >= ( num_elements_t )0u )
        {
            ptr_out_particles_t out_particles = ( ptr_out_particles_t )(
                uintptr_t )NS(BeamMonitor_get_out_address)( monitor );

            success = NS(Particles_copy_to_generic_addr_data)(
                out_particles, out_particle_id, in_particles, idx );

            if( success != 0 )
            {
                NS(Particles_set_state_value)( in_particles, idx, 0 );
            }
        }
    }

    return success;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_BE_MONITOR_TRACK_MONITOR_C99_HEADER_H__ */

/* end: sixtracklib/common/be_monitor/track_monitor.h */
