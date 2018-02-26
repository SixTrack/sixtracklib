// SixTrackLib
//
// Authors: R. De Maria, G. Iadarola, D. Pellegrini, H. Jasim
//
// Copyright 2017 CERN. This software is distributed under the terms of the GNU
// Lesser General Public License version 2.1, copied verbatim in the file
//`COPYING''.
//
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.

#ifndef _PARTICLE_
#define _PARTICLE_

#include "myint.h"

// Particle type

typedef CLGLOBAL struct Particles {
  // size information
  uint64_t npart; // number of particles
  // reference quantities
  CLGLOBAL double *charge0; // C
  CLGLOBAL double *mass0;   // eV
  CLGLOBAL double *beta0;   // nounit
  CLGLOBAL double *gamma0;  // nounit
  CLGLOBAL double *p0c; // eV

  // coordinate arrays
  CLGLOBAL int64_t *partid;
  CLGLOBAL int64_t *elemid; // element at which the particle was lost
  CLGLOBAL int64_t *turn; // turn at which the particle was lost
  CLGLOBAL int64_t *state; // negative means particle lost
  CLGLOBAL double *s;     // [m]
  CLGLOBAL double *x;     // [m]
  CLGLOBAL double *px;    // Px/P0
  CLGLOBAL double *y;     // [m]
  CLGLOBAL double *py;    // Px/P0
  CLGLOBAL double *sigma; // s-beta0*c*t  where t is the time
  // since the beginning of the simulation
  CLGLOBAL double *psigma;  // (E-E0) / (beta0 P0c) conjugate of sigma
  CLGLOBAL double *delta;   // P/P0-1 = 1/rpp-1
  CLGLOBAL double *rpp;     // ratio P0 /P
  CLGLOBAL double *rvv;     // ratio beta / beta0
  CLGLOBAL double *chi;     // q/charge0 * m/m0
  CLGLOBAL double *rcharge; // m/m0
  CLGLOBAL double *rmass;   // q/charge0
} Particles;

typedef __constant char *cc;

double Particles_charge0(Particles *p, uint64_t ip) { return p->charge0[ip]; };
double Particles_mass0(Particles *p, uint64_t ip) { return p->mass0[ip]; };
double Particles_beta0(Particles *p, uint64_t ip) { return p->beta0[ip]; };
double Particles_gamma0(Particles *p, uint64_t ip) { return p->gamma0[ip]; };
double Particles_p0c(Particles *p, uint64_t ip) { return p->p0c[ip]; };
int64_t Particles_partid(Particles *p, uint64_t ip) { return p->partid[ip]; };
int64_t Particles_elemid(Particles *p, uint64_t ip) { return p->elemid[ip]; };
int64_t Particles_turn(Particles *p, uint64_t ip) { return p->turn[ip]; };
int64_t Particles_state(Particles *p, uint64_t ip) { return p->state[ip]; };
double Particles_s(Particles *p, uint64_t ip) { return p->s[ip]; };
double Particles_x(Particles *p, uint64_t ip) { return p->x[ip]; };
double Particles_px(Particles *p, uint64_t ip) { return p->px[ip]; };
double Particles_y(Particles *p, uint64_t ip) { return p->y[ip]; };
double Particles_py(Particles *p, uint64_t ip) { return p->py[ip]; };
double Particles_sigma(Particles *p, uint64_t ip) { return p->sigma[ip]; };
double Particles_psigma(Particles *p, uint64_t ip) { return p->psigma[ip]; };
double Particles_delta(Particles *p, uint64_t ip) { return p->delta[ip]; };
double Particles_rpp(Particles *p, uint64_t ip) { return p->rpp[ip]; };
double Particles_rvv(Particles *p, uint64_t ip) { return p->rvv[ip]; };
double Particles_chi(Particles *p, uint64_t ip) { return p->chi[ip]; };
double Particles_mass(Particles *p, uint64_t ip) {
  return p->mass0[ip] * p->rmass[ip];
};
void Particles_addto_energy(Particles *p, uint64_t ip, double denergy) {
  double mass = Particles_mass(p, ip);
  double pc = (1 + Particles_delta(p, ip)) * Particles_p0c(p, ip);
  double energy = sqrt(pc * pc + mass * mass) + denergy;
  pc = sqrt(energy * energy - mass * mass);
  p->rpp[ip] = pc / Particles_p0c(p, ip);
  p->delta[ip] = p->rpp[ip] - 1;
  p->rvv[ip] = p->beta0[ip] * energy / pc;
};

Particles *Particles_unpack(Particles *p, CLGLOBAL value_t *pp) {
  p->charge0 = ((CLGLOBAL double *)p + pp[1].u64);
  p->mass0 = ((CLGLOBAL double *)p + pp[2].u64);
  p->beta0 = ((CLGLOBAL double *)p + pp[3].u64);
  p->gamma0 = ((CLGLOBAL double *)p + pp[4].u64);
  p->p0c = ((CLGLOBAL double *)p + pp[5].u64);
  p->partid = ((CLGLOBAL int64_t *)p + pp[6].u64);
  p->elemid = ((CLGLOBAL int64_t *)p + pp[7].u64);
  p->turn = ((CLGLOBAL int64_t *)p + pp[8].u64);
  p->state = ((CLGLOBAL int64_t *)p + pp[9].u64);
  p->s = ((CLGLOBAL double *)p + pp[10].u64);
  p->x = ((CLGLOBAL double *)p + pp[11].u64);
  p->px = ((CLGLOBAL double *)p + pp[12].u64);
  p->y = ((CLGLOBAL double *)p + pp[13].u64);
  p->py = ((CLGLOBAL double *)p + pp[14].u64);
  p->sigma = ((CLGLOBAL double *)p + pp[15].u64);
  p->psigma = ((CLGLOBAL double *)p + pp[16].u64);
  p->delta = ((CLGLOBAL double *)p + pp[17].u64);
  p->rpp = ((CLGLOBAL double *)p + pp[18].u64);
  p->rvv = ((CLGLOBAL double *)p + pp[19].u64);
  p->chi = ((CLGLOBAL double *)p + pp[20].u64);
  return (Particles *)p;
};

void Particles_copy(Particles *src, Particles *dst, int64_t srcid,
                    int64_t dstid) {
  dst->charge0[dstid] = src->charge0[srcid];
  dst->mass0[dstid] = src->mass0[srcid];
  dst->beta0[dstid] = src->beta0[srcid];
  dst->gamma0[dstid] = src->gamma0[srcid];
  dst->p0c[dstid] = src->p0c[srcid];
  dst->partid[dstid] = src->partid[srcid];
  dst->elemid[dstid] = src->elemid[srcid];
  dst->turn[dstid] = src->turn[srcid];
  dst->state[dstid] = src->state[srcid];
  dst->s[dstid] = src->s[srcid];
  dst->x[dstid] = src->x[srcid];
  dst->px[dstid] = src->px[srcid];
  dst->y[dstid] = src->y[srcid];
  dst->py[dstid] = src->py[srcid];
  dst->sigma[dstid] = src->sigma[srcid];
  dst->psigma[dstid] = src->psigma[srcid];
  dst->delta[dstid] = src->delta[srcid];
  dst->rpp[dstid] = src->rpp[srcid];
  dst->rvv[dstid] = src->rvv[srcid];
  dst->chi[dstid] = src->chi[srcid];
};

#endif
