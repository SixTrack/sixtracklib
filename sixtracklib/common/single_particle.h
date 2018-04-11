#ifndef SIXTRACKLIB_COMMON_SINGLE_PARTICLE_H__
#define SIXTRACKLIB_COMMON_SINGLE_PARTICLE_H__

#include "sixtracklib/_impl/namespace_begin.h"

#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/restrict.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef struct NS( SingleParticle )
{
    double q0;
    double mass0;
    double beta0;
    double gamma0;
    double p0c;

    int64_t partid;
    int64_t elemid;
    int64_t turn;
    int64_t state;

    double s;
    double x;
    double px;
    double y;
    double py;
    double sigma;

    double psigma;
    double delta;
    double rpp;
    double rvv;
    double chi;
} NS( SingleParticle );

void NS( SingleParticle_init )( NS( SingleParticle ) *
                                SIXTRL_RESTRICT ptr_particle );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* SIXTRACKLIB_COMMON_SINGLE_PARTICLE_H__ */

/* end: sixtracklib/baseline/single_particle.h */
