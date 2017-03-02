//SixTrackLib
//
//Authors: R. De Maria, G. Iadarola, D. Pellegrini
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

#endif
