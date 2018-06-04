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

SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_drift)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(Drift) *const SIXTRL_RESTRICT drift, 
    NS(Particles)* SIXTRL_RESTRICT io_particles );

SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_drift_exact)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(DriftExact) *const SIXTRL_RESTRICT drift, 
    NS(Particles)* SIXTRL_RESTRICT io_particles );

SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_multipole)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole, 
    NS(Particles)* SIXTRL_RESTRICT io_particles );

SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_beam_elements)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(Blocks) *const SIXTRL_RESTRICT beam_elements, 
    NS(Blocks)* SIXTRL_RESTRICT elem_by_elem );

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

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_beam_elements)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(Blocks) *const SIXTRL_RESTRICT beam_elements,
    NS(Blocks)* SIXTRL_RESTRICT elem_by_elem )
{
    int status = 0;
    
    NS(BlockInfo) const* be_block_it  = ( NS(BlockInfo) const* 
        )NS(Blocks_get_const_block_infos_begin)( beam_elements );
        
    NS(BlockInfo) const* be_block_end = ( NS(BlockInfo) const* 
        )NS(Blocks_get_const_block_infos_end)( beam_elements );
        
    NS(BlockInfo)* io_block_it = ( NS(BlockInfo)* 
        )NS(Blocks_get_block_infos_begin)( elem_by_elem );
    
    SIXTRL_ASSERT( ( beam_elements != 0 ) && ( particles != 0 ) && 
                   ( be_block_it   != 0 ) );
        
    for( ; be_block_it != be_block_end ; ++be_block_it )
    {
        NS(BlockType) const type_id = NS(BlockInfo_get_type_id)( be_block_it );
        
        switch( type_id )
        {
            case NS(BLOCK_TYPE_DRIFT):
            {
                NS(Drift) const* drift = ( NS(Drift const* ) 
                    )NS(Blocks_get_const_drift)( be_block_it );
                
                NS(Particles)* io_particles = ( NS(Particles)* 
                    )NS(Blocks_get_particles)( io_block_it );
                
                status |= NS(Track_drift)( particles, start_particle_index, 
                    end_particle_index, drift, io_particles );
                
                break;
            }
            
            
            case NS(BLOCK_TYPE_DRIFT_EXACT):
            {
                NS(DriftExact) const* drift = ( NS(DriftExact const* ) 
                    )NS(Blocks_get_const_drift_exact)( be_block_it );
                
                NS(Particles)* io_particles = ( NS(Particles)* 
                    )NS(Blocks_get_particles)( io_block_it );
                    
                status |= NS(Track_drift_exact)( particles, 
                    start_particle_index, end_particle_index, 
                        drift, io_particles );
                
                break;
            }
            
            case NS(BLOCK_TYPE_MULTIPOLE):
            {
                NS(MultiPole) const* multipole = ( NS(MultiPole const* ) 
                    )NS(Blocks_get_const_multipole)( be_block_it );
                
                NS(Particles)* io_particles = ( NS(Particles)* 
                    )NS(Blocks_get_particles)( io_block_it );
                    
                status |= NS(Track_multipole)( particles, 
                    start_particle_index, end_particle_index, 
                        multipole, io_particles );
                
                break;
            }
            
            default:
            {
                status = -1;
            }
        };
        
        if( io_block_it != 0 ) ++io_block_it;
    }
    
    return status;
}

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_TRACK_API_H__ */

/* end: sixtracklib/common/impl/track_api.h */
