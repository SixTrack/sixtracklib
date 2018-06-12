#ifndef SIXTRACKLIB_COMMON_IMPL_TRACK_API_H__
#define SIXTRACKLIB_COMMON_IMPL_TRACK_API_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/track.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_drift)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(Drift) *const SIXTRL_RESTRICT drift, 
    NS(Particles)* SIXTRL_RESTRICT io_particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_drift_exact)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(DriftExact) *const SIXTRL_RESTRICT drift, 
    NS(Particles)* SIXTRL_RESTRICT io_particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_multipole)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole, 
    NS(Particles)* SIXTRL_RESTRICT io_particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_beam_elements_particle)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const particle_index,
    const NS(Blocks) *const SIXTRL_RESTRICT beam_elements, 
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT elem_by_elem_info_begin );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_beam_elements)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const start_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(Blocks) *const SIXTRL_RESTRICT beam_elements, 
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT elem_by_elem_info_begin );

/* ========================================================================= */
/* =====        Implementation of Inline functions and methods         ===== */
/* ========================================================================= */

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_drift)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(Drift) *const SIXTRL_RESTRICT drift, 
    NS(Particles)* SIXTRL_RESTRICT io_particles )
{
    SIXTRL_TRACK_RETURN status = 0;
    
    NS(block_num_elements_t) ii = start_particle_index;
    
    SIXTRL_ASSERT( 
        ( start_particle_index <= end_particle_index ) &&
        ( end_particle_index <= NS(Particles_get_num_particles)( particles ) ) 
    );
    
    if( io_particles == 0 )
    {
        for( ; ii < end_particle_index ; ++ii )
        {
            status |= NS(Track_drift_particle)( particles, ii, drift );
        }
    }
    else
    {
        SIXTRL_ASSERT( NS(Particles_get_num_particles)( io_particles ) ==
                       NS(Particles_get_num_particles)( particles ) );
        
        for( ; ii < end_particle_index ; ++ii )
        {
            status |= NS(Track_drift_particle)( particles, ii, drift );
            
            NS(Particles_copy_single_unchecked)( 
                io_particles, ii, particles, ii );
        }
    }
    
    return status;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_drift_exact)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(DriftExact) *const SIXTRL_RESTRICT drift, 
    NS(Particles)* SIXTRL_RESTRICT io_particles )
{
    SIXTRL_TRACK_RETURN status = 0;
    
    NS(block_num_elements_t) ii = start_particle_index;
    
    SIXTRL_ASSERT( ( start_particle_index <= end_particle_index ) &&
        ( end_particle_index <= NS(Particles_get_num_particles)( particles ) ) 
    );
    
    if( io_particles == 0 )        
    {
        for( ; ii < end_particle_index ; ++ii )
        {
            status |= NS(Track_drift_exact_particle)( particles, ii, drift );
        }
    }
    else
    {
        SIXTRL_ASSERT( NS(Particles_get_num_particles)( io_particles ) ==
                       NS(Particles_get_num_particles)( particles ) );
        
        for( ; ii < end_particle_index ; ++ii )
        {
            status |= NS(Track_drift_exact_particle)( particles, ii, drift );            
            NS(Particles_copy_single_unchecked)( 
                io_particles, ii, particles, ii );
        }
    }
    
    return status;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_multipole)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole, 
    NS(Particles)* SIXTRL_RESTRICT io_particles )
{
    SIXTRL_TRACK_RETURN status = 0;
    
    NS(block_num_elements_t) ii = start_particle_index;
    
    SIXTRL_ASSERT( ( start_particle_index <= end_particle_index ) &&
        ( end_particle_index <= NS(Particles_get_num_particles)( particles ) ) 
    );
    
    if( io_particles == 0 )        
    {
        for( ; ii < end_particle_index ; ++ii )
        {
            status |= NS(Track_multipole_particle)( particles, ii, multipole );
        }
    }
    else
    {
        SIXTRL_ASSERT( NS(Particles_get_num_particles)( io_particles ) ==
                       NS(Particles_get_num_particles)( particles ) );
        
        for( ; ii < end_particle_index ; ++ii )
        {
            status |= NS(Track_multipole_particle)( particles, ii, multipole );
            
            NS(Particles_copy_single_unchecked)( 
                io_particles, ii, particles, ii );
        }
    }
    
    return status;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_beam_elements_particle)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const index,
    const NS(Blocks) *const SIXTRL_RESTRICT beam_elements, 
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT elem_by_elem_info_begin )
{
    int ret = 0;
    
    SIXTRL_GLOBAL_DEC NS(BlockInfo) const* be_block_it = 
        NS(Blocks_get_const_block_infos_begin)( beam_elements );
    
    SIXTRL_GLOBAL_DEC NS(BlockInfo) const* be_block_end = 
        NS(Blocks_get_const_block_infos_end)( beam_elements );
        
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* io_block_it = elem_by_elem_info_begin;
        
    SIXTRL_ASSERT( ( beam_elements != 0 ) && ( particles != 0 ) && 
                   ( be_block_it   != 0 ) );
    
    if( io_block_it == 0 )
    {
        for( ; be_block_it != be_block_end ; ++be_block_it )
        {
            ret |= NS(Track_range_of_particles_over_beam_element)(
                particles, index, index + 1, be_block_it );
        }
    }
    else
    {
        for( ; be_block_it != be_block_end ; ++be_block_it, ++io_block_it )
        {
            #if !defined( _GPUCODE )
                
            NS(Particles)* io_particles = 
                NS(Blocks_get_particles)( io_block_it );
            
            #else /* !defined( _GPUCODE ) */
            
            NS(Particles) temp_particles;
            NS(Particles)* io_particles = 0;
            
            NS(BlockInfo) info = *io_block_it;
            
            SIXTRL_GLOBAL_DEC NS(Particles)* ptr_io_particles =
                NS(Blocks_get_particles)( &info );
                
            SIXTRL_ASSERT( ptr_io_particles != 0 );
            
            temp_particles = *ptr_io_particles;
            io_particles   = &temp_particles;
                    
            #endif /* !defined( _GPUCODE ) */
            
            SIXTRL_ASSERT( io_particles != 0 );
            
            ret |= NS(Track_range_of_particles_over_beam_element)(
                particles, index, index + 1, be_block_it );
            
            NS(Particles_copy_single_unchecked)( 
                io_particles, index, particles, index );
        }
    }
    
    return ret;
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_beam_elements)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(Blocks) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT elem_by_elem_info_begin )
{
    int ret = 0;
    
    SIXTRL_GLOBAL_DEC NS(BlockInfo) const* be_block_it = 
        NS(Blocks_get_const_block_infos_begin)( beam_elements );
    
    SIXTRL_GLOBAL_DEC NS(BlockInfo) const* be_block_end = 
        NS(Blocks_get_const_block_infos_end)( beam_elements );
        
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* io_block_it = elem_by_elem_info_begin;
        
    SIXTRL_ASSERT( ( beam_elements != 0 ) && ( particles != 0 ) && 
                   ( be_block_it   != 0 ) );
    
    if( io_block_it == 0 )
    {
        for( ; be_block_it != be_block_end ; ++be_block_it )
        {
            ret |= NS(Track_range_of_particles_over_beam_element)( particles, 
                start_particle_index, end_particle_index, be_block_it );
        }
    }
    else
    {
        for( ; be_block_it != be_block_end ; ++be_block_it, ++io_block_it )
        {
            #if !defined( _GPUCODE )
                
            NS(Particles)* io_particles = 
                NS(Blocks_get_particles)( io_block_it );
            
            #else /* !defined( _GPUCODE ) */
            
            NS(Particles) temp_particles;
            NS(Particles)* io_particles = 0;
            
            NS(BlockInfo) info = *io_block_it;
            
            SIXTRL_GLOBAL_DEC NS(Particles)* ptr_io_particles =
                NS(Blocks_get_particles)( &info );
                
            SIXTRL_ASSERT( ptr_io_particles != 0 );
            
            temp_particles = *ptr_io_particles;
            io_particles   = &temp_particles;
                    
            #endif /* !defined( _GPUCODE ) */
            
            ret |= NS(Track_range_of_particles_over_beam_element)( particles, 
                start_particle_index, end_particle_index, be_block_it );
            
            NS(Particles_copy_range_unchecked)( io_particles, particles, 
                start_particle_index, end_particle_index );
        }
    }
    
    return ret;
}

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_TRACK_API_H__ */

/* end: sixtracklib/common/impl/track_api.h */
