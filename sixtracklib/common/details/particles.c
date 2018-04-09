#if !defined( _GPUCODE )

#include "sixtracklib/_impl/namespace_begin.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/impl/particles_type.h"

#include <assert.h>

#include "sixtracklib/common/restrict.h"
#include "sixtracklib/common/single_particle.h"
#include "sixtracklib/common/values.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
    
extern bool NS(Particles_has_values)( const NS(Particles) *const SIXTRL_RESTRICT p );
    
extern NS(Particles)* NS(Particles_unpack_values)( 
    NS(Particles)* SIXTRL_RESTRICT p, union NS(CommonValues) const* SIXTRL_RESTRICT pp );

extern void NS(Particles_copy)( 
    NS(Particles)* SIXTRL_RESTRICT dest, uint64_t const dest_id, 
    NS(Particles) const* SIXTRL_RESTRICT src, uint64_t const src_id );

extern void NS(Particles_init_from_single)( 
    NS(Particles)* SIXTRL_RESTRICT dest, 
    struct NS(SingleParticle) const* SIXTRL_RESTRICT src );

#endif /* !defined( _GPUCODE ) */

/* -------------------------------------------------------------------------- */

bool NS(Particles_has_values)( const NS(Particles) *const SIXTRL_RESTRICT p )
{
    bool const is_valid = (
        ( p != 0 ) && 
        ( p->q0     != 0 ) && ( p->mass0  != 0 ) && ( p->beta0  != 0 ) && 
        ( p->gamma0 != 0 ) && ( p->p0c    != 0 ) && 
        ( p->partid != 0 ) && ( p->elemid != 0 ) && 
        ( p->turn   != 0 ) && ( p->state  != 0 ) && 
        ( p->s != 0 ) && ( p->x  != 0 ) && ( p->px    != 0 ) &&
        ( p->y != 0 ) && ( p->py != 0 ) && ( p->sigma != 0 ) &&
        ( p->psigma != 0 ) && ( p->delta != 0 ) && 
        ( p->rpp    != 0 ) && ( p->rvv   != 0 ) && ( p->chi != 0 ) );
    
    return is_valid;
}

NS(Particles)* NS(Particles_unpack_values)( 
    NS(Particles)* SIXTRL_RESTRICT p, NS(value_t) const* SIXTRL_RESTRICT pp )
{
    double*     ptr_double = ( double*  )( p );
    int64_t*    ptr_int64  = ( int64_t* )( p );
    
    p->q0       = ptr_double + pp[  1 ].u64;
    p->mass0    = ptr_double + pp[  2 ].u64;
    p->beta0    = ptr_double + pp[  3 ].u64;
    p->gamma0   = ptr_double + pp[  3 ].u64;
    p->p0c      = ptr_double + pp[  4 ].u64;
    
    p->partid   = ptr_int64  + pp[  5 ].u64;
    p->elemid   = ptr_int64  + pp[  6 ].u64;
    p->turn     = ptr_int64  + pp[  7 ].u64;
    p->state    = ptr_int64  + pp[  8 ].u64;
                
    p->s        = ptr_double + pp[  9 ].u64;
    p->x        = ptr_double + pp[ 10 ].u64;
    p->px       = ptr_double + pp[ 11 ].u64;
    p->y        = ptr_double + pp[ 12 ].u64;
    p->py       = ptr_double + pp[ 13 ].u64;
    p->sigma    = ptr_double + pp[ 14 ].u64;
                
    p->psigma   = ptr_double + pp[ 15 ].u64; 
    p->delta    = ptr_double + pp[ 16 ].u64;
    p->rpp      = ptr_double + pp[ 17 ].u64;
    p->rvv      = ptr_double + pp[ 18 ].u64;
    p->chi      = ptr_double + pp[ 19 ].u64;
    
    return p;
}

void NS(Particles_copy)( 
    NS(Particles)* SIXTRL_RESTRICT dest, uint64_t const dest_id, 
    NS(Particles) const* SIXTRL_RESTRICT src, uint64_t const src_id )
{
    assert( ( ( NS(Particles)* SIXTRL_RESTRICT )src != dest ) &&
            ( NS(Particles_has_values)( dest ) ) &&
            ( dest_id >= dest->partid[ 0 ] ) && 
            ( dest->npart > ( dest_id - dest->partid[ 0 ] ) ) &&
            ( NS(Particles_has_values)( src  ) ) && 
            ( src_id  >= src->partid[ 0 ]  ) &&
            ( src->npart > ( src_id  - src->partid[ 0 ] ) ) );
    
    dest->q0[ dest_id ]     = src->q0[ src_id ];
    dest->mass0[ dest_id ]  = src->mass0[ src_id ];
    dest->beta0[ dest_id ]  = src->beta0[ src_id ];
    dest->gamma0[ dest_id ] = src->gamma0[ src_id ];
    dest->p0c[ dest_id ]    = src->p0c[ src_id ];
    dest->partid[ dest_id ] = src->partid[ src_id ];
    dest->elemid[ dest_id ] = src->elemid[ src_id ];        
    dest->turn[ dest_id ]   = src->turn[ src_id ];
    dest->state[ dest_id ]  = src->state[ src_id ];
    dest->s[ dest_id ]      = src->s[ src_id ];
    dest->x[ dest_id ]      = src->x[ src_id ];
    dest->y[ dest_id ]      = src->y[ src_id ];
    dest->px[ dest_id ]     = src->px[ src_id ];
    dest->y[ dest_id ]      = src->y[ src_id ];
    dest->py[ dest_id ]     = src->py[ src_id ];
    dest->sigma[ dest_id ]  = src->sigma[ src_id ];
    dest->delta[ dest_id ]  = src->delta[ src_id ];
    dest->rpp[ dest_id ]    = src->rpp[ src_id ];
    dest->rvv[ dest_id ]    = src->rvv[ src_id ];
    dest->chi[ dest_id ]    = src->chi[ src_id ];
}

void NS(Particles_init_from_single)( 
    NS(Particles)* SIXTRL_RESTRICT dest, 
    NS(SingleParticle) const* SIXTRL_RESTRICT const_src )
{
    /* Casting away the constness -> this is ugly, but the API should be const
     * to hint at the usage.
     * TODO: Search for a cleaner way to do this once the API becomes a little
     * more settled */
    
    NS(SingleParticle) * src = ( SingleParticle* )const_src; 
    
    assert( ( dest != 0 ) && ( src != 0 ) );
    
    dest->npart  = ( uint64_t )1u;
    dest->q0     = &src->q0;
    dest->mass0  = &src->mass0;
    dest->beta0  = &src->beta0;
    dest->gamma0 = &src->gamma0;
    dest->p0c    = &src->p0c;
    dest->partid = &src->partid;
    dest->elemid = &src->elemid;        
    dest->turn   = &src->turn;
    dest->state  = &src->state;
    dest->s      = &src->s;
    dest->x      = &src->x;
    dest->y      = &src->y;
    dest->px     = &src->px;
    dest->y      = &src->y;
    dest->py     = &src->py;
    dest->sigma  = &src->sigma;
    dest->delta  = &src->delta;
    dest->rpp    = &src->rpp;
    dest->rvv    = &src->rvv;
    dest->chi    = &src->chi;
    
    return;
}

/* end: sixtracklib/common/details/particles.c */
