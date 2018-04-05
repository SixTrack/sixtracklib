#ifndef SIXTRACKLIB_COMMON_TRACK_H__
#define SIXTRACKLIB_COMMON_TRACK_H__

#include <stdlib.h>
#include <stdint.h>

#include "sixtracklib/common/restrict.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

struct Particles;

int Drift_track( 
    struct Particles* SIXTRL_RESTRICT particles, uint64_t ip, double length );

int DriftExact_track(
    struct Particles* SIXTRL_RESTRICT particles, uint64_t ip, double length );

   
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* SIXTRACKLIB_COMMON_TRACK_H__ */
