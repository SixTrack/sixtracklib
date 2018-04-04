#include "sixtracklib/common/single_particle.h"
#include "sixtracklib/common/restrict.h"

#include <assert.h>

extern void SingleParticle_init( SingleParticle* SIXTRL_RESTRICT ptr_particle ); 


void SingleParticle_init( SingleParticle* SIXTRL_RESTRICT ptr_particle )
{
    SingleParticle const empty_particle = 
    {
        0.0, 0.0, 0.0, 0.0, 0.0,        /* q0, mass0, beta0, gamma0, p0c, */
        -1,  -1,  -1,  0,               /* partid, elemid, turn, state,   */
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   /* s, x, px, y, py, sigma */
        0.0, 0.0, 0.0, 0.0, 0.0         /* psigma, delta, rpp, rvv, chi */
    };
    
    assert( ptr_particle != 0 );
    
    *ptr_particle = empty_particle;
    
    return;
}

/* end: sixtracklib/baseline/details/single_particle.c */
