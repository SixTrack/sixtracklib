#ifndef SIXTRACKLIB_SIMD_TRACK_H__
#define SIXTRACKLIB_SIMD_TRACK_H__

#include "sixtracklib/_impl/definitions.h"

struct NS(Particles);

int NS(Track_simd_drift_sse2)(
    struct NS(Particles)* SIXTRL_RESTRICT particles, SIXTRL_REAL_T const length );

int NS(Track_simd_drift_avx)(
    struct NS(Particles)* SIXTRL_RESTRICT particles, SIXTRL_REAL_T const length );

#endif /* SIXTRACKLIB_SIMD_TRACK_H__ */

/* end: sixtracklib/simd/track.h */
