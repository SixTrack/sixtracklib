//SixTrackLib
//
//Authors: R. De Maria, G. Iadarola, D. Pellegrini, H. Jasim
//
//Copyright 2017 CERN. This software is distributed under the terms of the GNU
//Lesser General Public License version 2.1, copied verbatim in the file
//`COPYING''.
//
//In applying this licence, CERN does not waive the privileges and immunities
//granted to it by virtue of its status as an Intergovernmental Organization or
//submit itself to any jurisdiction.


#ifndef _PARTICLE_
#define _PARTICLE_

#include "myint.h"

// Particle type

typedef CLGLOBAL struct Particles {
    // size information
    uint64_t npart; // number of particles
    // reference quantities
    CLGLOBAL double  *q0; // C
    CLGLOBAL double  *mass0; // eV
    CLGLOBAL double  *beta0; // nounit
    CLGLOBAL double  *gamma0;  // nounit
    CLGLOBAL double  *p0c; //eV

    //coordinate arrays
    CLGLOBAL int64_t *partid;
    CLGLOBAL int64_t *elemid;//element at which the particle was lost
    CLGLOBAL int64_t *turn;  //turn at which the particle was lost
    CLGLOBAL int64_t *state; //negative means particle lost
    CLGLOBAL double  *s; // [m]
    CLGLOBAL double  *x; // [m]
    CLGLOBAL double  *px; // Px/P0
    CLGLOBAL double  *y;  // [m]
    CLGLOBAL double  *py; // Px/P0
    CLGLOBAL double  *sigma; // s-beta0*c*t  where t is the time
    // since the beginning of the simulation
    CLGLOBAL double  *psigma; // (E-E0) / (beta0 P0c) conjugate of sigma
    CLGLOBAL double  *delta; // P/P0-1 = 1/rpp-1
    CLGLOBAL double  *rpp; // ratio P0 /P
    CLGLOBAL double  *rvv; // ratio beta / beta0
    CLGLOBAL double  *chi; // q/q0 * m/m0
} Particles;

Particles* Particles_unpack(Particles* p) {
    p->partid = ( (CLGLOBAL int64_t *) p + ((uint64_t) p->partid) );
    p->elemid = ( (CLGLOBAL int64_t *) p + ((uint64_t) p->elemid) );
    p->turn   = ( (CLGLOBAL int64_t *) p + ((uint64_t) p->turn)   );
    p->state  = ( (CLGLOBAL int64_t *) p + ((uint64_t) p->state)  );
    p->s      = ( (CLGLOBAL double *) p + ((uint64_t) p->s)      );
    p->x      = ( (CLGLOBAL double *) p + ((uint64_t) p->x)      );
    p->px     = ( (CLGLOBAL double *) p + ((uint64_t) p->px)     );
    p->y      = ( (CLGLOBAL double *) p + ((uint64_t) p->y)      );
    p->py     = ( (CLGLOBAL double *) p + ((uint64_t) p->py)     );
    p->sigma  = ( (CLGLOBAL double *) p + ((uint64_t) p->sigma)  );
    p->psigma = ( (CLGLOBAL double *) p + ((uint64_t) p->psigma) );
    p->delta  = ( (CLGLOBAL double *) p + ((uint64_t) p->delta)  );
    p->rpp    = ( (CLGLOBAL double *) p + ((uint64_t) p->rpp)    );
    p->rvv    = ( (CLGLOBAL double *) p + ((uint64_t) p->rvv)    );
    p->chi    = ( (CLGLOBAL double *) p + ((uint64_t) p->chi)    );
    return (Particles*) p;
};

void Particles_copy(Particles* src, Particles* dst, int64_t partid){
    if (partid==0) {
        dst->npart = src->npart;
    };
    dst->q0[partid]         = src->q0[partid]    ;
    dst->mass0[partid]      = src->mass0[partid] ;
    dst->beta0[partid]      = src->beta0[partid] ;
    dst->gamma0[partid]     = src->gamma0[partid];
    dst->p0c[partid]        = src->p0c[partid]   ;
    dst->partid[partid]     = src->partid[partid];
    dst->elemid[partid]     = src->elemid[partid];
    dst->turn[partid]       = src->turn[partid]  ;
    dst->state[partid]      = src->state[partid] ;
    dst->s[partid]          = src->s[partid]     ;
    dst->x[partid]          = src->x[partid]     ;
    dst->px[partid]         = src->px[partid]    ;
    dst->y[partid]          = src->y[partid]     ;
    dst->py[partid]         = src->py[partid]    ;
    dst->sigma[partid]      = src->sigma[partid] ;
    dst->psigma[partid]     = src->psigma[partid];
    dst->delta[partid]      = src->delta[partid] ;
    dst->rpp[partid]        = src->rpp[partid]   ;
    dst->rvv[partid]        = src->rvv[partid]   ;
    dst->chi[partid]        = src->chi[partid]   ;
};


// ElemByElem Type


typedef CLGLOBAL struct ElemByElem {
    uint64_t nturns;
    uint64_t nelems;
    uint64_t next;
    Particles* particles;
} ElemByElem;

void ElemByElem_unpack (ElemByElem* el){
    uint64_t ndata = el->nelems*el->nturns;
    for (int jj=0; jj< ndata; jj++) {
        Particles_unpack(&el->particles[jj]);
    };
};

void ElemByElem_append (ElemByElem* el, Particles* p, int64_t partid){
    Particles_copy(p, &el->particles[el->next], partid);
    el->next++;
};

// TurnByTurn Type

typedef CLGLOBAL struct TurnByTurn {
    uint64_t nturns;
    uint64_t next;
    Particles * particles;
} TurnByTurn;

void TurnByTurn_unpack (TurnByTurn* el){
    uint64_t ndata = el->nturns;
    for (int jj=0; jj< ndata; jj++) {
        Particles_unpack(&el->particles[jj]);
    };
};

void TurnByTurn_append (TurnByTurn* el, Particles* p, int64_t partid){
    Particles_copy(p, &el->particles[el->next], partid);
    el->next++;
};

#endif

