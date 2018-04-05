#ifndef SIXTRACKLIB_COMMON_PARTICLES_H__
#define SIXTRACKLIB_COMMON_PARTICLES_H__

#if !defined( _GPUCODE )

#include "sixtracklib/common/impl/particles_type.h"

#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

#include "sixtracklib/common/restrict.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
    
struct SingleParticle;
union  CommonValues;

#endif /* !defined( _GPUCODE ) */

/* -------------------------------------------------------------------------- */

bool Particles_has_values( const Particles *const SIXTRL_RESTRICT p );

Particles* Particles_unpack_values( 
    Particles* SIXTRL_RESTRICT p, 
    union CommonValues const* SIXTRL_RESTRICT pp );

void Particles_copy( 
    Particles* SIXTRL_RESTRICT dest, uint64_t const dest_id, 
    Particles const* SIXTRL_RESTRICT src, uint64_t const src_id );

void Particles_init_from_single( 
    Particles* SIXTRL_RESTRICT dest, 
    struct SingleParticle const* SIXTRL_RESTRICT src );


/* -------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */
    
#endif /* SIXTRACKLIB_COMMON_PARTICLES_H__ */

/* end: sixtracklib/sixtracklib/common/particles.h */
