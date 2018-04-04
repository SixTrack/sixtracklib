#ifndef SIXTRACKLIB_COMMON_IMPL_TRACK_COMMON_H__
#define SIXTRACKLIB_COMMON_IMPL_TRACK_COMMON_H__

#include "sixtracklib/common/restrict.h"

#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

struct SingleParticle;
struct Particles;

int Drift_track_single_particle_common(
    struct SingleParticle* SIXTRL_RESTRICT particle, double length );

int DriftExact_track_single_particle_common(
    struct SingleParticle* SIXTRL_RESTRICT particle, double length );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */

int Drift_track_particles_common( 
    struct Particles* SIXTRL_RESTRICT particles, uint64_t ip, double length );

int DriftExact_track_particles_common(
    struct Particles* SIXTRL_RESTRICT particles, uint64_t ip, double length );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
   
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* SIXTRACKLIB_COMMON_IMPL_TRACK_COMMON_H__ */
