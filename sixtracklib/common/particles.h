#ifndef SIXTRACKLIB_COMMON_PARTICLES_H__
#define SIXTRACKLIB_COMMON_PARTICLES_H__

#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

#include "sixtracklib/common/restrict.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

struct SingleParticle;
union  CommonValues;
    
typedef struct Particles
{
    uint64_t    npart;
    
    double* q0;     /* C */
    double* mass0;  /* eV */
    double* beta0;  /* nounit */
    double* gamma0; /* nounit */
    double* p0c;    /* eV */

    /* coordinate arrays */
    int64_t* partid;
    int64_t* elemid; /* element at which the particle was lost */
    int64_t* turn;   /* turn at which the particle was lost */
    int64_t* state;  /* negative means particle lost */
    
    double* s;       /* [m] */
    double* x;       /* [m] */
    double* px;      /* Px/P0 */
    double* y;       /* [m] */
    double* py;      /* Px/P0 */
    double* sigma;   /* s-beta0*c*t  where t is the time
                        since the beginning of the simulation */
                        
    double* psigma;  /* (E-E0) / (beta0 P0c) conjugate of sigma */
    double* delta;   /* P/P0-1 = 1/rpp-1 */
    double* rpp;     /* ratio P0 /P */
    double* rvv;     /* ratio beta / beta0 */
    double* chi;     /* q/q0 * m/m0  */
}
Particles;

bool Particles_has_values( const Particles *const SIXTRL_RESTRICT p );

Particles* Particles_unpack_values( 
    Particles* SIXTRL_RESTRICT p, 
    union CommonValues const* SIXTRL_RESTRICT pp );

void Particles_copy( 
    Particles* SIXTRL_RESTRICT dest, uint64_t const dest_id, 
    Particles const* SIXTRL_RESTRICT src, uint64_t const src_id );

void Particles_init_from_single( 
    Particles* SIXTRL_RESTRICT dest, 
    struct SingleParticle const* SIXTRL_RESTRICT src );

#ifdef __cplusplus
}
#endif /* __cplusplus */
    
#endif /* SIXTRACKLIB_COMMON_PARTICLES_H__ */

/* end: sixtracklib/sixtracklib/common/particles.h */
