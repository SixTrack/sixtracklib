#ifndef SIXTRACKLIB_COMMON_PARTICLES_SEQUENCE_H__
#define SIXTRACKLIB_COMMON_PARTICLES_SEQUENCE_H__

#if !defined( _GPUCODE ) 

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h> 

#include "sixtracklib/_impl/definitions.h"
    
#endif /* !defined( _GPUCODE ) */
    
struct NS(MemPool);
struct NS(Particles);

typedef struct NS(ParticlesSequence)
{
    SIXTRL_SIZE_T           size;
    struct NS(MemPool)*     ptr_mem_pool;    
    struct NS(Particles)*   particles_buffer;
}
NS(ParticlesSequence);
    

SIXTRL_STATIC NS(ParticlesSequence)* NS(ParticlesSequence_preset)( 
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence );

bool NS(ParticlesSequence_init)( 
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence,
    SIXTRL_SIZE_T const size, 
    SIXTRL_SIZE_T const num_of_particles, 
    SIXTRL_SIZE_T* ptr_chunk_size,
    SIXTRL_SIZE_T* ptr_alignment, 
    bool make_packed );

SIXTRL_STATIC SIXTRL_SIZE_T NS(ParticlesSequence_get_size)(
    const NS(ParticlesSequence) *const SIXTRL_RESTRICT sequence );

SIXTRL_STATIC struct NS(MemPool) const* 
NS(ParticlesSequence_get_const_ptr_to_mem_pool)(
    const NS(ParticlesSequence) *const SIXTRL_RESTRICT sequence );

void NS(ParticlesSequence_free)(
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence );

struct NS(Particles) const* NS(ParticlesSequence_get_const_begin)(
    const NS(ParticlesSequence) *const SIXTRL_RESTRICT sequence );
                                 
struct NS(Particles) const* NS(ParticlesSequence_get_const_end)(
    const NS(ParticlesSequence) *const SIXTRL_RESTRICT sequence );

struct NS(Particles) const* NS(ParticlesSequence_get_const_particles_by_index)(
    const NS(ParticlesSequence) *const SIXTRL_RESTRICT sequence, 
    SIXTRL_SIZE_T const index );

SIXTRL_STATIC struct NS(Particles)* NS(ParticlesSequence_get_begin)(
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence );
                                 
SIXTRL_STATIC struct NS(Particles)* NS(ParticlesSequence_get_end)(
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence );

SIXTRL_STATIC struct NS(Particles)* 
NS(ParticlesSequence_get_particles_by_index)(
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence, 
    SIXTRL_SIZE_T const index );

/* ************************************************************************** */
/* ******            Implementation of inline functions                 ***** */
/* ************************************************************************** */

#if !defined( _GPUCODE )

#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/impl/particles_type.h"

#endif /* !defined( _GPUCODE ) */

SIXTRL_INLINE NS(ParticlesSequence)* NS(ParticlesSequence_preset)( 
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence )
{
    if( sequence != 0 )
    {
        sequence->size = ( SIXTRL_SIZE_T )0u;
        sequence->ptr_mem_pool = 0;
        sequence->particles_buffer = 0;
    }
    
    return sequence;
}

SIXTRL_INLINE SIXTRL_SIZE_T NS(ParticlesSequence_get_size)(
    const NS(ParticlesSequence) *const SIXTRL_RESTRICT sequence )
{
    return ( ( sequence != 0 ) && ( sequence->particles_buffer != 0 ) )
        ? sequence->size : ( SIXTRL_SIZE_T )0u;
}

SIXTRL_INLINE struct NS(MemPool) const* 
    NS(ParticlesSequence_get_const_ptr_to_mem_pool)(
        const NS(ParticlesSequence) *const SIXTRL_RESTRICT sequence )
{
    return ( sequence != 0 ) ? sequence->ptr_mem_pool : 0;
}

SIXTRL_INLINE NS(Particles)* NS(ParticlesSequence_get_begin)(
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence )
{
    return ( NS(Particles)* )NS(ParticlesSequence_get_const_begin)( sequence );
}
                                 
SIXTRL_INLINE NS(Particles)* NS(ParticlesSequence_get_end)(
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence )
{
    return ( NS(Particles)* )NS(ParticlesSequence_get_const_end)( sequence );
}

SIXTRL_INLINE NS(Particles)* NS(ParticlesSequence_get_particles_by_index)(
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence, 
    SIXTRL_SIZE_T const index )
{
    return ( NS(Particles)* )NS(ParticlesSequence_get_const_particles_by_index)( 
        sequence, index );
}
    
#if !defined( _GPUCODE ) 

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */
        
#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_PARTICLES_SEQUENCE_H__ */

/* end: sixtracklib/common/particles_sequence.h */
