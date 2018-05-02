#ifndef SIXTRACKLIB_SIMD_TRACK_H__
#define SIXTRACKLIB_SIMD_TRACK_H__

#include "sixtracklib/_impl/definitions.h"

struct NS(Particles);
struct NS(Drift);

int NS(Track_simd_drift_sse2)(
    struct NS(Particles)* SIXTRL_RESTRICT particles, 
    const struct NS(Drift) *const SIXTRL_RESTRICT drift );

int NS(Track_simd_drift_avx)(
    struct NS(Particles)* SIXTRL_RESTRICT particles, 
    const struct NS(Drift) *const SIXTRL_RESTRICT drift );

#endif /* SIXTRACKLIB_SIMD_TRACK_H__ */

/* end: sixtracklib/simd/track.h */
