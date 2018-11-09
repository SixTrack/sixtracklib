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

SIXTRL_FN SIXTRL_STATIC SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) const*
NS(BeamMonitor_get_const_ptr_particles)( SIXTRL_BE_ARGPTR_DEC const
    struct NS(BeamMonitor) *const SIXTRL_RESTRICT monitor,
    NS(buffer_size_t) const store_idx );

SIXTRL_FN SIXTRL_STATIC SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)*
NS(BeamMonitor_get_ptr_particles)( SIXTRL_BE_ARGPTR_DEC const
    struct NS(BeamMonitor) *const SIXTRL_RESTRICT monitor,
    NS(buffer_size_t) const store_idx );

SIXTRL_FN SIXTRL_STATIC NS(buffer_addr_t)
NS(BeamMonitor_get_particles_begin_addr)( SIXTRL_BE_ARGPTR_DEC const
    struct NS(BeamMonitor) *const SIXTRL_RESTRICT monitor,
    NS(buffer_size_t) const store_idx );

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_beam_monitor)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamMonitor) *const monitor );

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

SIXTRL_INLINE SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)*
NS(BeamMonitor_get_ptr_particles)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor,
    NS(buffer_size_t) const store_idx )
{
    return ( SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)*
        )NS(BeamMonitor_get_particles_begin_addr)( monitor, store_idx );
}

SIXTRL_INLINE SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) const*
NS(BeamMonitor_get_const_ptr_particles)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor,
    NS(buffer_size_t) const store_idx )
{
    return ( SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) const*
        )NS(BeamMonitor_get_particles_begin_addr)( monitor, store_idx );
}

SIXTRL_INLINE NS(buffer_addr_t) NS(BeamMonitor_get_particles_begin_addr)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor,
    NS(buffer_size_t) const store_idx )
{
    typedef NS(be_monitor_stride_t) stride_t;

    stride_t const stride = NS(BeamMonitor_get_io_store_stride)( monitor );

    /* Currently, only consequentive attribte storage is implemented */
    SIXTRL_ASSERT( NS(BeamMonitor_are_attributes_continous)( monitor ) );

    SIXTRL_ASSERT( NS(BeamMonitor_get_io_address)( monitor ) !=
                  ( NS(be_monitor_addr_t) )0 );
    SIXTRL_ASSERT( stride > ( stride_t )0u );
    SIXTRL_ASSERT( store_idx < ( NS(buffer_size_t)
        )NS(BeamMonitor_get_num_stores)( monitor ) );

    return ( NS(buffer_addr_t ) )(
            NS(BeamMonitor_get_io_address)( monitor ) + store_idx * stride );
}

SIXTRL_INLINE int NS(Track_particle_beam_monitor)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT in_particles,
    NS(particle_num_elements_t) const particle_index,
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

    nturn_t const monitor_start = NS(BeamMonitor_get_start)( monitor );
    nturn_t const turn = NS(Particles_get_at_turn_value)(
        in_particles, particle_index );

    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );

    /* Currently, only consequentive attribte storage is implemented */
    SIXTRL_ASSERT( NS(BeamMonitor_are_attributes_continous)( monitor ) );

    SIXTRL_ASSERT( NS(BeamMonitor_get_skip)( monitor ) > ( nturn_t )0u  );
    SIXTRL_ASSERT( NS(BeamMonitor_get_num_stores)( monitor ) > ( nturn_t)0u );

    SIXTRL_ASSERT( in_particles != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(Particles_get_num_of_particles)(
        in_particles ) > particle_index );

    SIXTRL_ASSERT( NS(Particles_get_state_value)(
        in_particles, particle_index ) == ( NS(particle_index_t) )1 );

    if( ( NS(BeamMonitor_get_io_address)( monitor ) != ( addr_t )0 ) &&
        ( turn >= monitor_start ) )
    {
        nturn_t const num_stores = NS(BeamMonitor_get_num_stores)( monitor );
        nturn_t const skip       = NS(BeamMonitor_get_skip)( monitor );

        nturn_t const turns_since_start = turn - monitor_start;

        if( 0 == ( turns_since_start % skip ) )
        {
            nturn_t store_idx = turns_since_start / skip;

            index_t const out_particle_id = NS(Particles_get_particle_id_value)(
                in_particles, particle_index );

            if( ( store_idx >= num_stores ) &&
                ( NS(BeamMonitor_is_rolling)( monitor ) ) )
            {
                store_idx = store_idx % num_stores;
            }

            if( ( store_idx < num_stores ) &&
                ( out_particle_id >= ( index_t )0u ) )
            {
                success = NS(Particles_copy_to_generic_addr_data)(
                    ( ptr_out_particles_t )( uintptr_t
                        )NS(BeamMonitor_get_particles_begin_addr)(
                            monitor, store_idx ),
                    ( num_elements_t )out_particle_id,
                    in_particles, particle_index );
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
