#ifndef SIXTRACKLIB_COMMON_IMPL_PARTICLES_TYPE_H__
#define SIXTRACKLIB_COMMON_IMPL_PARTICLES_TYPE_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/namespace_begin.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

#include "sixtracklib/_impl/inline.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
    
#endif /* !defined( _GPUCODE ) */
    
typedef struct NS(Particles)
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
    double* py;      /* Py/P0 */
    double* sigma;   /* s-beta0*c*t  where t is the time
                        since the beginning of the simulation */
                        
    double* psigma;  /* (E-E0) / (beta0 P0c) conjugate of sigma */
    double* delta;   /* P/P0-1 = 1/rpp-1 */
    double* rpp;     /* ratio P0 /P */
    double* rvv;     /* ratio beta / beta0 */
    double* chi;     /* q/q0 * m/m0  */
}
NS(Particles);

/* ========================================================================= */

double NS(Particles_get_q0)( const NS(Particles) *const SIXTRL_RESTRICT p,
                             uint64_t id );

double NS(Particles_get_mass0)( const NS(Particles) *const SIXTRL_RESTRICT p,
                                uint64_t id );

double NS(Particles_get_beta0)( const NS(Particles) *const SIXTRL_RESTRICT p,
                                uint64_t id );

double NS(Particles_get_gamma0)( const NS(Particles) *const SIXTRL_RESTRICT p,
                                uint64_t id );

double NS(Particles_get_p0c)( const NS(Particles) *const SIXTRL_RESTRICT p,
                              uint64_t id );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(Particles_is_particle_lost)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

int64_t NS(Particles_get_particle_id )(
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

int64_t NS(Particles_get_lost_at_element_id)(
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

int64_t NS(Particles_get_lost_at_turn)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

int64_t NS(Particles_get_state)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

double NS( Particles_get_s )( const NS(Particles) *const SIXTRL_RESTRICT p,
                              uint64_t id );

double NS(Particles_get_x )( const NS(Particles) *const SIXTRL_RESTRICT p,
                             uint64_t id );

double NS(Particles_get_y )( const NS(Particles) *const SIXTRL_RESTRICT p,
                                uint64_t id );

double NS( Particles_get_px )( const NS(Particles) *const SIXTRL_RESTRICT p,
                                uint64_t id );

double NS( Particles_get_py )( const NS(Particles) *const SIXTRL_RESTRICT p,
                                 uint64_t id );

double NS( Particles_get_sigma )( const NS(Particles) *const SIXTRL_RESTRICT p,
                                uint64_t id );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

double NS(Particles_get_psigma)( const NS(Particles) *const SIXTRL_RESTRICT p, 
                            uint64_t id );

double NS(Particles_get_delta)( const NS(Particles) *const SIXTRL_RESTRICT p,
                            uint64_t id );

double NS(Particles_get_rpp )( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

double NS(Particles_get_rvv )( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

double NS(Particles_get_chi)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

/* ========================================================================= */

void NS(Particles_set_q0)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double q0 );

void NS(Particles_set_mass0)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double mass0 );

void NS(Particles_set_beta0)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double beta0 );

void NS(Particles_set_gamma0)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double gamma0 );

void NS(Particles_set_p0c)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double p0c );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

void NS(Particles_set_particle_id)(
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, int64_t partid );

void NS(Particles_set_lost_at_element_id)(
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, int64_t elemid );

void NS(Particles_set_lost_at_turn)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, int64_t turn );

void NS(Particles_set_state)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, int64_t state );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

void NS( Particles_set_s )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double s );

void NS(Particles_set_x )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double x );

void NS(Particles_set_y )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double y );

void NS( Particles_set_px )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double px );

void NS( Particles_set_py )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double py );

void NS( Particles_get_sigma )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double sigma );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

void NS(Particles_set_psigma)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double psigma );

void NS(Particles_set_delta)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double delta );

void NS(Particles_set_rpp )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double rpp );

void NS(Particles_set_rvv )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double rvv );

void NS(Particles_set_chi)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double chi );

/* ========================================================================= */

SIXTRL_INLINE double NS(Particles_get_q0)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->q0 != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->q0[ id ];
}

SIXTRL_INLINE double NS(Particles_get_mass0)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->mass0 != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->mass0[ id ];
}

SIXTRL_INLINE double NS(Particles_get_beta0)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->beta0 != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->beta0[ id ];
}

SIXTRL_INLINE double NS(Particles_get_gamma0)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->gamma0 != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->gamma0[ id ];
}

SIXTRL_INLINE double NS(Particles_get_p0c)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->p0c != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->p0c[ id ];
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE int NS(Particles_is_particle_lost)( 
    const NS(Particles) *const p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->state != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return ( p->state[ id ] < 0 ) ? 1 : 0;
}

SIXTRL_INLINE int64_t NS(Particles_get_lost_at_element_id)(
    const NS(Particles) *const p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->elemid != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->elemid[ id ];
}

SIXTRL_INLINE int64_t NS(Particles_get_lost_at_turn)( 
    const NS(Particles) *const p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->turn != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->turn[ id ];
}

SIXTRL_INLINE int64_t NS(Particles_get_state)( 
    const NS(Particles) *const p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->state != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->state[ id ];
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE double NS(Particles_get_s )( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->s != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->s[ id ];
}

SIXTRL_INLINE double NS(Particles_get_x )( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->x != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->x[ id ];
}

SIXTRL_INLINE double NS(Particles_get_y )( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->y != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->y[ id ];
}

SIXTRL_INLINE double NS( Particles_get_px )( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->px != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->px[ id ];
}

SIXTRL_INLINE double NS( Particles_get_py )( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->py != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->py[ id ];
}

SIXTRL_INLINE double NS( Particles_get_sigma )( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->sigma != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->sigma[ id ];
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE double NS(Particles_get_psigma)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->psigma != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->psigma[ id ];
}

SIXTRL_INLINE double NS(Particles_get_delta)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->delta != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->delta[ id ];
}

SIXTRL_INLINE double NS(Particles_get_rpp )( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->rpp != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->rpp[ id ];
}

SIXTRL_INLINE double NS(Particles_get_rvv )( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->rvv != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->rvv[ id ];
}

SIXTRL_INLINE double NS(Particles_get_chi)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->chi != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->chi[ id ];
}

/* ========================================================================= */

SIXTRL_INLINE void NS(Particles_set_q0)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double q0 )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->q0 != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->q0[ id ] = q0;
    return;
}

SIXTRL_INLINE void NS(Particles_set_mass0)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double mass0 )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->mass0 != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->mass0[ id ] = mass0;
    return;
}

SIXTRL_INLINE void NS(Particles_set_beta0)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double beta0 )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->beta0 != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->beta0[ id ] = beta0;
    return;
}

SIXTRL_INLINE void NS(Particles_set_gamma0)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double gamma0 )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->gamma0 != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->gamma0[ id ] = gamma0;
    return;
}

SIXTRL_INLINE void NS(Particles_set_p0c)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double p0c )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->p0c != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->p0c[ id ] = p0c;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS(Particles_set_particle_id)(
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, int64_t partid )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->partid != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->partid[ id ] = partid;
    return;
}

SIXTRL_INLINE void NS(Particles_set_lost_at_element_id)(
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, int64_t elemid )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->elemid != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->elemid[ id ] = elemid;
    return;
}

SIXTRL_INLINE void NS(Particles_set_lost_at_turn)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, int64_t turn )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->p0c != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->partid[ id ] = partid;
    return;
}

SIXTRL_INLINE void NS(Particles_set_state)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, int64_t state )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->state != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->state[ id ] = state;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS( Particles_set_s )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double s )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->p0c != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->partid[ id ] = partid;
    return;
}

SIXTRL_INLINE void NS(Particles_set_x )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double x )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->x != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->x[ id ] = x;
    return;
}

SIXTRL_INLINE void NS(Particles_set_y )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double y )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->y != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->y[ id ] = partid;
    return;
}

SIXTRL_INLINE void NS( Particles_set_px )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double px )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->px != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->px[ id ] = partid;
    return;
}

SIXTRL_INLINE void NS( Particles_set_py )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double py )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->py != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->py[ id ] = py;
    return;
}

SIXTRL_INLINE void NS( Particles_get_sigma )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double sigma )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->sigma != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->sigma[ id ] = sigma;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS(Particles_set_psigma)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double psigma )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->psigma != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->psigma[ id ] = psigma;
    return;
}

SIXTRL_INLINE void NS(Particles_set_delta)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double delta )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->p0c != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->delta[ id ] = delta;
    return;
}

SIXTRL_INLINE void NS(Particles_set_rpp )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double rpp )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->rpp != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->rpp[ id ] = rpp;
    return;
}

SIXTRL_INLINE void NS(Particles_set_rvv )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double rvv )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->rvv != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->rvv[ id ] = rvv;
    return;
}

SIXTRL_INLINE void NS(Particles_set_chi)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double chi )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->chi != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->chi[ id ] = chi;
    return;
}

/* ========================================================================= */

#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */
    
#endif /* SIXTRACKLIB_COMMON_IMPL_PARTICLES_TYPE_H__ */

/* end: sixtracklib/sixtracklib/common/impl/particles_type.h */
