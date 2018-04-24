#ifndef SIXTRACKLIB_COMMON_PARTICLES_H__
#define SIXTRACKLIB_COMMON_PARTICLES_H__

#if !defined( _GPUCODE )
#include "sixtracklib/_impl/definitions.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/mem_pool.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */
    
struct NS( MemPool );
struct NS( SingleParticle );
struct NS( Particles );

/* ========================================================================= */

SIXTRL_STATIC SIXTRL_UINT64_T const NS(PARTICLES_PACK_INDICATOR) = ( SIXTRL_UINT64_T )1u;

SIXTRL_STATIC SIXTRL_INT64_T const NS( PARTICLE_VALID_STATE ) = ( SIXTRL_INT64_T )0;

SIXTRL_STATIC SIXTRL_UINT64_T const NS( PARTICLES_FLAGS_NONE ) = ( SIXTRL_UINT64_T )0x0000;

SIXTRL_STATIC SIXTRL_UINT64_T const NS( PARTICLES_FLAGS_PACKED ) = ( SIXTRL_UINT64_T )0x0001;

SIXTRL_STATIC SIXTRL_UINT64_T const NS( PARTICLES_FLAGS_OWNS_MEMORY ) = ( SIXTRL_UINT64_T )0x0002;

SIXTRL_STATIC SIXTRL_UINT64_T const
    NS( PARTICLES_FLAGS_MEM_CTX_MEMPOOL ) = ( SIXTRL_UINT64_T )0x0010;

SIXTRL_STATIC SIXTRL_UINT64_T const
    NS( PARTICLES_FLAGS_MEM_CTX_SINGLEPARTICLE ) = ( SIXTRL_UINT64_T )0x0020;
    
SIXTRL_STATIC SIXTRL_UINT64_T const 
    NS(PARTICLES_FLAGS_MEM_CTX_FLAT_MEMORY ) = ( SIXTRL_UINT64_T )0x0040;

SIXTRL_STATIC SIXTRL_UINT64_T const NS( PARTICLES_FLAGS_ALIGN_MASK ) = ( SIXTRL_UINT64_T )0xFFFF00;

SIXTRL_STATIC SIXTRL_UINT64_T const NS( PARTICLES_MAX_ALIGNMENT ) = ( SIXTRL_UINT64_T )0xFFFF;

SIXTRL_STATIC SIXTRL_UINT64_T const
    NS( PARTICLES_FLAGS_ALIGN_MASK_OFFSET_BITS ) = ( SIXTRL_UINT64_T )8;

/* ========================================================================= */

SIXTRL_STATIC SIXTRL_SIZE_T const NS( PARTICLES_DEFAULT_MEMPOOL_CHUNK_SIZE ) = (SIXTRL_SIZE_T)8u;
SIXTRL_STATIC SIXTRL_SIZE_T const NS( PARTICLES_DEFAULT_MEMPOOL_ALIGNMENT ) = (SIXTRL_SIZE_T)16u;
SIXTRL_STATIC SIXTRL_SIZE_T const NS( PARTICLES_NUM_OF_DOUBLE_ELEMENTS ) = (SIXTRL_SIZE_T)16u;
SIXTRL_STATIC SIXTRL_SIZE_T const NS( PARTICLES_NUM_OF_INT64_ELEMENTS ) = (SIXTRL_SIZE_T)4u;
SIXTRL_STATIC SIXTRL_SIZE_T const NS( PARTICLES_NUM_OF_ATTRIBUTES ) = ( SIXTRL_SIZE_T )20u;
SIXTRL_STATIC SIXTRL_SIZE_T const NS( PARTICLES_PACK_BLOCK_LENGTH ) = ( SIXTRL_SIZE_T )192u;


SIXTRL_STATIC SIXTRL_UINT64_T const NS( PARTICLES_UNPACK_MAP  ) = ( SIXTRL_UINT64_T )0x0000;
SIXTRL_STATIC SIXTRL_UINT64_T const NS( PARTICLES_UNPACK_COPY ) = ( SIXTRL_UINT64_T )0x0001;
SIXTRL_STATIC SIXTRL_UINT64_T const NS( PARTICLES_UNPACK_CHECK_CONSISTENCY ) = 
    ( SIXTRL_UINT64_T )0x02;


/* -------------------------------------------------------------------------- */

struct NS( Particles ) *
    NS( Particles_preset )( struct NS( Particles ) * SIXTRL_RESTRICT p );

size_t NS( Particles_predict_required_capacity )( 
    size_t num_particles,
    size_t* SIXTRL_RESTRICT chunk_size,
    size_t* SIXTRL_RESTRICT alignment, 
    bool make_packed );

struct NS( Particles ) * NS( Particles_new )( size_t npart );

struct NS( Particles ) * NS( Particles_new_aligned )( 
    SIXTRL_SIZE_T const npart, SIXTRL_SIZE_T alignment );

struct NS( Particles ) *
    NS( Particles_new_on_mempool )( size_t npart,
                                    struct NS( MemPool ) *
                                        SIXTRL_RESTRICT pool );

struct NS( Particles ) * NS( Particles_new_single )();

struct NS( Particles ) *
    NS( Particles_new_on_single )( struct NS( SingleParticle ) *
                                   ptr_single_particle );

bool NS(Particles_unpack)(
    struct NS(Particles)* SIXTRL_RESTRICT particles, 
    unsigned char* SIXTRL_RESTRICT mem, uint64_t flags );
    
void NS( Particles_free )( struct NS( Particles ) * SIXTRL_RESTRICT particles );

/* -------------------------------------------------------------------------- */

bool NS( Particles_is_packed )( const struct NS( Particles ) *
                                const SIXTRL_RESTRICT p );

bool NS( Particles_manages_own_memory )( const struct NS( Particles ) *
                                         const SIXTRL_RESTRICT p );

bool NS( Particles_uses_mempool )( const struct NS( Particles ) *
                                   const SIXTRL_RESTRICT p );

bool NS( Particles_uses_single_particle )( const struct NS( Particles ) *
                                           const SIXTRL_RESTRICT p );

bool NS( Particles_uses_flat_memory )( 
    const struct NS(Particles )* const SIXTRL_RESTRICT p );

struct NS( MemPool ) const* NS( Particles_get_const_mem_pool )(
    const struct NS( Particles ) * const SIXTRL_RESTRICT p );

struct NS( SingleParticle ) const* NS(
    Particles_get_const_base_single_particle )( const struct NS( Particles ) *
                                                const SIXTRL_RESTRICT p );

unsigned char const* NS( Particles_get_const_flat_memory )(
    const struct NS( Particles ) * const SIXTRL_RESTRICT p );
    
/* -------------------------------------------------------------------------- */

bool NS( Particles_has_defined_alignment )( const struct NS( Particles ) *
                                            const SIXTRL_RESTRICT p );

bool NS( Particles_is_aligned )( const struct NS( Particles ) *
                                     const SIXTRL_RESTRICT p,
                                 size_t alignment );

bool NS( Particles_check_alignment )( const struct NS( Particles ) *
                                          const SIXTRL_RESTRICT p,
                                      size_t alignment );

uint64_t NS( Particles_alignment )( const struct NS( Particles ) *
                                    const SIXTRL_RESTRICT p );

/* -------------------------------------------------------------------------- */

bool NS( Particles_is_consistent )( const struct NS( Particles ) *
                                    const SIXTRL_RESTRICT p );

/* -------------------------------------------------------------------------- */

bool NS( Particles_deep_copy_one )( struct NS( Particles ) *
                                        SIXTRL_RESTRICT dest,
                                    uint64_t dest_id,
                                    struct NS( Particles )
                                        const* SIXTRL_RESTRICT src,
                                    uint64_t src_id );

bool NS( Particles_deep_copy_all )( struct NS( Particles ) *
                                        SIXTRL_RESTRICT dest,
                                    struct NS( Particles )
                                        const* SIXTRL_RESTRICT src );

/* -------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_PARTICLES_H__ */

/* end: sixtracklib/sixtracklib/common/particles.h */
