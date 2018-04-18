#include "sixtracklib/common/particles_sequence.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/mem_pool.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/impl/particles_type.h"

/* -------------------------------------------------------------------------- */

extern bool NS(ParticlesSequence_init)( 
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence,
    SIXTRL_SIZE_T const size, SIXTRL_SIZE_T const num_of_particles, 
    SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_chunk_size,
    SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_alignment, 
    bool make_packed );

extern void NS(ParticlesSequence_free)(
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence );

extern NS(Particles) const* NS(ParticlesSequence_get_const_begin)(
    const NS(ParticlesSequence) *const SIXTRL_RESTRICT sequence );
                                 
extern NS(Particles) const* NS(ParticlesSequence_get_const_end)(
    const NS(ParticlesSequence) *const SIXTRL_RESTRICT sequence );

extern NS(Particles) const* NS(ParticlesSequence_get_const_particles_by_index)(
    const NS(ParticlesSequence) *const SIXTRL_RESTRICT sequence, 
    SIXTRL_SIZE_T const index );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC void NS(ParticlesSequence_set_size)(
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence, SIXTRL_SIZE_T const size );

SIXTRL_STATIC void NS(ParticlesSequence_set_ptr_to_mem_pool)(
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence, NS(MemPool)* mem_pool );

SIXTRL_STATIC NS(MemPool)* NS(ParticlesSequence_get_ptr_to_mem_pool)(
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence );

SIXTRL_STATIC void NS(ParticlesSequence_set_ptr_to_particles_buffer)(
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence, NS(Particles)* buffer );

SIXTRL_STATIC NS(Particles)* NS(ParticlesSequence_get_ptr_to_particles_buffer)(
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence );

/* -------------------------------------------------------------------------- */

bool NS(ParticlesSequence_init)( 
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence,
    SIXTRL_SIZE_T const size, SIXTRL_SIZE_T const num_of_particles, 
    SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_chunk_size,
    SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_alignment, bool make_packed )
{
    bool success = false;
    
    SIXTRL_STATIC SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;
    
    SIXTRL_SIZE_T const single_capacity = 
        NS(Particles_predict_required_capacity)( 
            num_of_particles, ptr_chunk_size, ptr_alignment, make_packed );
        
    SIXTRL_SIZE_T const required_capacity = size * single_capacity;
    
    NS(Particles)* buffer = 0;
    NS(MemPool)* pool = 0;
    
    if( ( sequence != 0 ) && ( size > ZERO_SIZE ) && ( ptr_chunk_size != 0 ) &&
        ( ptr_chunk_size != ptr_alignment ) && 
        ( required_capacity > ZERO_SIZE ) )
    {
        buffer = ( NS(Particles)* )malloc( sizeof( NS(Particles) ) * size );
        pool = ( NS(MemPool)* )malloc( sizeof( NS(MemPool) ) );
        
        if( pool != 0 )
        {
            NS(MemPool_init)( pool, required_capacity, *ptr_chunk_size );
        }
        
        if( ( buffer != 0 ) && ( pool != 0 ) &&
            ( NS(MemPool_get_remaining_bytes)( pool ) >= required_capacity ) )
        {
            NS(Particles)* it  = buffer;
            NS(Particles)* end = it + size;
            
            success = true;
            
            for( ; it != end ; ++it )
            {
                NS(Particles_preset)( it );
            }
            
            for( it = buffer ; it != end ; ++it )
            {
                NS(Particles)* temp = 
                    NS(Particles_new_on_mempool)( num_of_particles, pool );
                    
                if( temp == 0 )
                {
                    success = false;
                    break;
                }
                
                /* TODO: Add API to Particles to allow in-place initialization 
                 *       of NS(Particles) instances to avoid this awkward way
                 *       of doing things here: */
                
                *it = *temp;                
                free( temp );
                temp = 0;
                
            }
        }
    }
    
    if( success )
    {
        NS(ParticlesSequence_set_size)( sequence, size );
        NS(ParticlesSequence_set_ptr_to_mem_pool)( sequence, pool );
        NS(ParticlesSequence_set_ptr_to_particles_buffer)( sequence, buffer );
    }
    else
    {
        if( buffer != 0 )
        {
            NS(Particles)* it  = buffer;
            NS(Particles)* end = it + size;
            
            for( ; it != end ; ++it )
            {
                if( ( NS(Particles_get_size)( it ) > ZERO_SIZE ) ||
                    ( NS(Particles_get_const_mem_pool)( it ) != 0 ) )
                {
                    NS(Particles_free)( it );
                    NS(Particles_preset)( it );                    
                }
            }
            
            free( buffer );
            buffer = 0;
        }
        
        if( pool != 0 )
        {
            NS(MemPool_free)( pool );
            free( pool );
            pool = 0;
        }
    }
    
    assert( ( (  success ) && ( pool != 0 ) && ( buffer != 0 ) ) ||
            ( ( !success ) && ( pool == 0 ) && ( buffer == 0 ) ) );
    
    return success;
}

/* ------------------------------------------------------------------------- */

void NS(ParticlesSequence_free)(
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence )
{
    if( sequence != 0 )
    {
        NS(MemPool)* pool = 
            NS(ParticlesSequence_get_ptr_to_mem_pool)( sequence );
            
        NS(Particles)* buffer =
            NS(ParticlesSequence_get_ptr_to_particles_buffer)( sequence );
            
        if( buffer != 0 )
        {
            SIXTRL_SIZE_T const size = 
                NS(ParticlesSequence_get_size)( sequence );
            
            NS(Particles)* it  = buffer;
            NS(Particles)* end = buffer + size;
            
            for( ; it != end ; ++it )
            {
                SIXTRL_ASSERT( ( pool == 0 ) ||
                    ( ( !NS(Particles_manages_own_memory)( it ) ) &&
                      (  NS(Particles_uses_mempool)( it ) ) &&
                      (  NS(Particles_get_const_mem_pool)( it ) == pool ) ) );
                
                NS(Particles_free)( it );
            }
            
            free( buffer );
            NS(ParticlesSequence_set_ptr_to_particles_buffer)( sequence, 0 );
        }
        
        if( pool != 0 )
        {
            NS(MemPool_free)( pool );            
            free( pool );
            NS(ParticlesSequence_set_ptr_to_mem_pool)( sequence, 0 );
        }
    }
    
    return;
}

/* ------------------------------------------------------------------------- */

NS(Particles) const* NS(ParticlesSequence_get_const_begin)(
    const NS(ParticlesSequence) *const SIXTRL_RESTRICT sequence )
{
    return ( ( sequence != 0 ) && ( sequence->ptr_mem_pool != 0 ) )
        ? sequence->particles_buffer : 0;
}

/* ------------------------------------------------------------------------- */
                                 
NS(Particles) const* NS(ParticlesSequence_get_const_end)(
    const NS(ParticlesSequence) *const SIXTRL_RESTRICT sequence )
{
    typedef NS(Particles) particles_t;
    
    particles_t const* end = NS(ParticlesSequence_get_const_begin)( sequence );
    SIXTRL_SIZE_T const size = NS(ParticlesSequence_get_size)( sequence );
    
    return ( end != 0 ) ? ( end + size ) : end;
}

/* ------------------------------------------------------------------------- */

NS(Particles) const* NS(ParticlesSequence_get_const_particles_by_index)(
    const NS(ParticlesSequence) *const SIXTRL_RESTRICT sequence, 
    SIXTRL_SIZE_T const index )
{
    typedef NS(Particles) particles_t;
    
    particles_t const* it = NS(ParticlesSequence_get_const_begin)( sequence );
    SIXTRL_SIZE_T const size = NS(ParticlesSequence_get_size)( sequence );
    
    return ( ( it != 0 ) && ( size > index ) ) ? ( it + index ) : 0;
}

/* ------------------------------------------------------------------------- */

void NS(ParticlesSequence_set_size)(
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence, SIXTRL_SIZE_T const size )
{
    SIXTRL_ASSERT( sequence != 0 );
    sequence->size = size;
    return;
}

void NS(ParticlesSequence_set_ptr_to_mem_pool)(
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence, NS(MemPool)* mem_pool )
{
    SIXTRL_ASSERT( sequence != 0 );
    sequence->ptr_mem_pool = mem_pool;
    return;
}

NS(MemPool)* NS(ParticlesSequence_get_ptr_to_mem_pool)(
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence )
{
    SIXTRL_ASSERT( sequence != 0 );
    return sequence->ptr_mem_pool;
}

void NS(ParticlesSequence_set_ptr_to_particles_buffer)(
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence, NS(Particles)* buffer )
{
    SIXTRL_ASSERT( sequence != 0 );
    sequence->particles_buffer = buffer;
    return;
}

NS(Particles)* NS(ParticlesSequence_get_ptr_to_particles_buffer)(
    NS(ParticlesSequence)* SIXTRL_RESTRICT sequence )
{
    SIXTRL_ASSERT( sequence != 0 );
    return sequence->particles_buffer;
}

/* ------------------------------------------------------------------------- */

/* end: sixtracklib/common/details/particles_sequence.c */
