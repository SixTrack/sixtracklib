#ifndef SIXTRACKLIB_COMMON_IMPL_TRACK_SINGLE_H__
#define SIXTRACKLIB_COMMON_IMPL_TRACK_SINGLE_H__

#include <stdlib.h>
#include <stdint.h>

#include "sixtracklib/common/restrict.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

struct SingleParticle;

int Drift_track_single( 
    struct SingleParticle* SIXTRL_RESTRICT particle, double length );

int DriftExact_track_single(
    struct SingleParticle* SIXTRL_RESTRICT particle, double length );
   
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* SIXTRACKLIB_COMMON_IMPL_TRACK_SINGLE_H__ */
