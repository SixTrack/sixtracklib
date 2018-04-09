#ifndef SIXTRACKLIB_COMMON_PARTICLES_H__
#define SIXTRACKLIB_COMMON_PARTICLES_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/namespace_begin.h"

#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

#include "sixtracklib/common/restrict.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

struct NS(Particles);    
struct NS(SingleParticle);
union  NS(CommonValues);

#endif /* !defined( _GPUCODE ) */

/* -------------------------------------------------------------------------- */

bool Particles_has_values( const struct NS(Particles) *const SIXTRL_RESTRICT p );

struct NS(Particles)* NS(Particles_unpack_values)( 
    struct NS(Particles)* SIXTRL_RESTRICT p, 
    union NS(CommonValues) const* SIXTRL_RESTRICT pp );

void NS(Particles_copy)( 
    struct NS(Particles)* SIXTRL_RESTRICT dest, uint64_t const dest_id, 
    struct NS(Particles) const* SIXTRL_RESTRICT src, uint64_t const src_id );

void NS(Particles_init_from_single)( 
    struct NS(Particles)* SIXTRL_RESTRICT dest, 
    struct NS(SingleParticle) const* SIXTRL_RESTRICT src );

/* -------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */
    
#endif /* SIXTRACKLIB_COMMON_PARTICLES_H__ */

/* end: sixtracklib/sixtracklib/common/particles.h */
