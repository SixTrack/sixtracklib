#ifndef SIXTRACKLIB_COMMON_TRACK_H__
#define SIXTRACKLIB_COMMON_TRACK_H__

#include "sixtracklib/_impl/namespace_begin.h"

#include <stdlib.h>
#include <stdint.h>

#include "sixtracklib/common/restrict.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

struct NS(Particles);

int NS(Drift_track)( 
    struct NS(Particles)* SIXTRL_RESTRICT particles, uint64_t ip, double length );

int NS(DriftExact_track)(
    struct NS(Particles)* SIXTRL_RESTRICT particles, uint64_t ip, double length );
   
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* SIXTRACKLIB_COMMON_TRACK_H__ */

/* end: sixtracklib/common/track.h */
