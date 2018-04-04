#ifndef SIXTRACKLIB_BASELINE_SINGLE_PARTICLE_H__
#define SIXTRACKLIB_BASELINE_SINGLE_PARTICLE_H__

#include <stdlib.h>
#include <stdint.h>

#include "sixtracklib/common/restrict.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef struct SingleParticle
{
    double      q0;
    double      mass0;
    double      beta0;
    double      gamma0;
    double      p0c;
    
    int64_t     partid;
    int64_t     elemid;
    int64_t     turn;
    int64_t     state;
    
    double      s;
    double      x;
    double      px;
    double      y;
    double      py;
    double      sigma;
    
    double      psigma;
    double      delta;
    double      rpp;
    double      rvv;
    double      chi;
}
SingleParticle;

void SingleParticle_init( SingleParticle* SIXTRL_RESTRICT ptr_particle ); 
    
#ifdef __cplusplus
}
#endif /* __cplusplus */
    
#endif /* SIXTRACKLIB_BASELINE_SINGLE_PARTICLE_H__ */

/* end: sixtracklib/baseline/single_particle.h */
