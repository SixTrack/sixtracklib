#include "value.h"

#ifndef _BEAM
#define _BEAM

typedef struct Particle {
  int partid;
  int elemid;
  int turn;
  int state; //negativeparticle lost
  double s;
  double x;
  double px; // Px/P0
  double y;
  double py; // Px/P0
  double sigma;
  double psigma; // (E-E0)/ (beta0 P0c)
  double chi; // q/q0 * m/m0
  double delta;
  double rpp; // ratio P0/P
  double rvv; // ratio beta / beta0
  double beta;
  double gamma;
  double m0; // eV
  double q0; // C
  double q; // C
  double beta0;
  double gamma0;
  double p0c; //eV
} Particle;


typedef struct Beam {
  uint64_t npart;
  Particle* particles;
} Beam;

#ifndef _GPUCODE
#include <stdlib.h>

Beam *Beam_new(uint64_t npart);

#endif

#endif


