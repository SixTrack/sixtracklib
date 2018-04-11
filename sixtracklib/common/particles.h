#ifndef SIXTRACKLIB_COMMON_PARTICLES_H__
#define SIXTRACKLIB_COMMON_PARTICLES_H__

#include "sixtracklib/_impl/namespace_begin.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/restrict.h"
#include "sixtracklib/common/mem_pool.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

struct NS( MemPool );
struct NS( SingleParticle );
struct NS( Particles );

/* -------------------------------------------------------------------------- */

size_t NS( Particles_predict_required_capacity )( 
    size_t num_particles,
    size_t* SIXTRL_RESTRICT chunk_size,
    size_t* SIXTRL_RESTRICT alignment, 
    bool make_packed );

struct NS( Particles ) * NS( Particles_new )( size_t npart );

struct NS( Particles ) *
    NS( Particles_new_on_mempool )( size_t npart,
                                    struct NS( MemPool ) *
                                        SIXTRL_RESTRICT pool );

struct NS( Particles ) * NS( Particles_new_single )();

struct NS( Particles ) *
    NS( Particles_new_on_single )( struct NS( SingleParticle ) *
                                   ptr_single_particle );

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

struct NS( MemPool ) const* NS( Particles_get_const_mem_pool )(
    const struct NS( Particles ) * const SIXTRL_RESTRICT p );

struct NS( SingleParticle ) const* NS(
    Particles_get_const_base_single_particle )( const struct NS( Particles ) *
                                                const SIXTRL_RESTRICT p );

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

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* SIXTRACKLIB_COMMON_PARTICLES_H__ */

/* end: sixtracklib/sixtracklib/common/particles.h */
