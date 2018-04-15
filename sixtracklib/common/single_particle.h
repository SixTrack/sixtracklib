#ifndef SIXTRACKLIB_COMMON_SINGLE_PARTICLE_H__
#define SIXTRACKLIB_COMMON_SINGLE_PARTICLE_H__

#if !defined( _GPUCODE )
#include "sixtracklib/_impl/definitions.h"

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
    
#endif /* !defined( _GPUCODE ) */

typedef struct NS( SingleParticle )
{
    SIXTRL_REAL_T q0;
    SIXTRL_REAL_T mass0;
    SIXTRL_REAL_T beta0;
    SIXTRL_REAL_T gamma0;
    SIXTRL_REAL_T p0c;

    SIXTRL_INT64_T partid;
    SIXTRL_INT64_T elemid;
    SIXTRL_INT64_T turn;
    SIXTRL_INT64_T state;

    SIXTRL_REAL_T s;
    SIXTRL_REAL_T x;
    SIXTRL_REAL_T px;
    SIXTRL_REAL_T y;
    SIXTRL_REAL_T py;
    SIXTRL_REAL_T sigma;

    SIXTRL_REAL_T psigma;
    SIXTRL_REAL_T delta;
    SIXTRL_REAL_T rpp;
    SIXTRL_REAL_T rvv;
    SIXTRL_REAL_T chi;
} NS( SingleParticle );

void NS( SingleParticle_init )( NS( SingleParticle ) *
                                SIXTRL_RESTRICT ptr_particle );

#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_SINGLE_PARTICLE_H__ */

/* end: sixtracklib/baseline/single_particle.h */
