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


typedef CLGLOBAL struct Particles {
  // size information
  uint64_t npart; // number of particles
  // reference quantities
  double q0; // C
  double mass0; // eV
  double beta0; // nounit
  double gamma0;  // nounit
  double p0c; //eV

  //coordinate arrays
  CLGLOBAL int64_t *partid;
  CLGLOBAL int64_t *elemid;//element at which the particle was lost
  CLGLOBAL int64_t *turn;  //turn at which the particle was lost
  CLGLOBAL int64_t *state; //negative means particle lost
  CLGLOBAL double *s; // [m]
  CLGLOBAL double *x; // [m]
  CLGLOBAL double *px; // Px/P0
  CLGLOBAL double *y;  // [m]
  CLGLOBAL double *py; // Px/P0
  CLGLOBAL double *sigma; // s-beta0*c*t  where t is the time
                          // since the beginning of the simulation
  CLGLOBAL double *psigma; // (E-E0) / (beta0 P0c) conjugate of sigma
  CLGLOBAL double *delta; // P/P0-1 = 1/rpp-1
  CLGLOBAL double *rpp; // ratio P0 /P
  CLGLOBAL double *rvv; // ratio beta / beta0
  CLGLOBAL double *chi; // q/q0 * m/m0
} Particles;

void Particles_unpack(Particles * p) {
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
};

typedef CLGLOBAL struct ElembyElem {
    uint64_t nelems;
    uint64_t nturns;
    Particles * particles;
} ElembyElem;

void ElembyElem_unpack (ElembyElem * el){
   uint64_t ndata = el->nelems*el->nturns;
   for (int jj=0; jj< ndata; jj++) {
       Particles_unpack(&el->particles[jj]);
   };
};

typedef CLGLOBAL struct TurnbyTurn {
    uint64_t nturns;
    Particles * particles;
} TurnbyTurn;

void TurnbyTurn_unpack (TurnbyTurn * el){
   uint64_t ndata = el->nturns;
   for (int jj=0; jj< ndata; jj++) {
       Particles_unpack(&el->particles[jj]);
   };
};

#endif

