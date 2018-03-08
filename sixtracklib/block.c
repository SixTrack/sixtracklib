#include "block.h"

#include "particle.h"
#include "track.c"

#include "BB6D.h"

// Data management

type_t get_type(CLGLOBAL value_t *elem) { return (type_t)elem[0].u64; }

// Drift

double Drift_get_length(CLGLOBAL value_t *elem) { return elem[1].f64; }

// DriftExact

double DriftExact_get_length(CLGLOBAL value_t *elem) { return elem[1].f64; }

// Multipole

int64_t Multipole_get_order(CLGLOBAL value_t *elem) { return elem[1].i64; }
double Multipole_get_length(CLGLOBAL value_t *elem) { return elem[2].f64; }
double Multipole_get_hxl(CLGLOBAL value_t *elem) { return elem[3].f64; }
double Multipole_get_hyl(CLGLOBAL value_t *elem) { return elem[4].f64; }
CLGLOBAL double *Multipole_get_bal(CLGLOBAL value_t *elem) {
  return &elem[6].f64;
}

// Cavity
double Cavity_get_voltage(CLGLOBAL value_t *elem) { return elem[1].f64; }
double Cavity_get_frequency(CLGLOBAL value_t *elem) { return elem[2].f64; }
double Cavity_get_lag(CLGLOBAL value_t *elem) { return elem[3].f64; }
double Cavity_get_lag_rad(CLGLOBAL value_t *elem) { return elem[4].f64; }

// Align

double Align_get_tilt(CLGLOBAL value_t *elem) { return elem[1].f64; }
double Align_get_cz(CLGLOBAL value_t *elem) { return elem[2].f64; }
double Align_get_sz(CLGLOBAL value_t *elem) { return elem[3].f64; }
double Align_get_dx(CLGLOBAL value_t *elem) { return elem[4].f64; }
double Align_get_dy(CLGLOBAL value_t *elem) { return elem[5].f64; }

// Tracking loop

#ifdef _GPUCODE

void track_single(Particles *particles, uint64_t partid,
                  CLGLOBAL value_t *elem) {
  // printf("single: partid=%u, typeid=%zu\n",partid, elem[0].u64);
  if (particles->state[partid] >= 0) {
    // printf("single: typeid=%zu\n",elem[0].u64);
    enum type_t typeid = get_type(elem);
    // printf("single: typedid=%u\n",typeid);
    switch (typeid) {
    case DriftID:
      Drift_track(particles, partid, Drift_get_length(elem));
      break;
    case DriftExactID:
      Drift_track(particles, partid, Drift_get_length(elem));
      break;
    case MultipoleID:
      Multipole_track(particles, partid, Multipole_get_order(elem),
                      Multipole_get_length(elem), Multipole_get_hxl(elem),
                      Multipole_get_hyl(elem), Multipole_get_bal(elem));
      break;
    case CavityID:
      Cavity_track(particles, partid, Cavity_get_voltage(elem),
                   Cavity_get_frequency(elem), Cavity_get_lag_rad(elem));
      break;
    case AlignID:
      Align_track(particles, partid, Align_get_cz(elem), Align_get_sz(elem),
                  Align_get_dx(elem), Align_get_dy(elem));
      break;
      
    case BeamBeamID:
        // printf("Strange!\n");
        {
        uint64_t data_size = elem[1].i64;
        uint64_t data_offset = elem[2].i64;

        CLGLOBAL value_t* data = elem + data_offset;
        
        BB6D_track(particles, partid, data);

        break;
        }
    } // end switch
  }   // end if state
}

CLKERNEL void Block_unpack(CLGLOBAL value_t *particles_p,  // Particles
                           CLGLOBAL value_t *elembyelem_p, // ElembyElem
                           CLGLOBAL value_t *turnbyturn_p) // TurnbyTurn
{
  uint64_t partid = get_global_id(0);

  bool elembyelem_flag = (elembyelem_p[0].i64 != 0);
  bool turnbyturn_flag = (turnbyturn_p[0].i64 != 0);

  Particles *particles = (Particles *)particles_p;
  Particles *elembyelem = (Particles *)elembyelem_p;
  Particles *turnbyturn = (Particles *)turnbyturn_p;

  // printf( (cc) "p[0] %i \n", particles_p[0].u64);
  // printf( (cc) "p[1] %i \n", particles_p[1].u64);
  // printf( (cc) "p[2] %i \n", particles_p[2].u64);
  // printf( (cc) "p[3] %i \n", particles_p[3].u64);
  // printf( (cc) "p[4] %i \n", particles_p[4].u64);
  // printf( (cc) "p[5] %i \n", particles_p[5].u64);
  // printf( (cc) "p->q0 %i \n", particles->q0);
  // printf( (cc) "p->mass0 %i \n", particles->mass0);
  // printf( (cc) "p->beta0 %i \n", particles->beta0);
  // printf( (cc) "p->gamma0 %i \n", particles->gamma0);
  // printf( (cc) "p->p0c %i \n", particles->p0c);

  Particles_unpack(particles, particles_p);

  // printf( (cc) "&p[0] %u \n", particles_p);
  // printf( (cc) "&p[0] %u \n", &particles_p[0]);
  // printf( (cc) "&p[1] %u \n", &particles_p[1]);
  // printf( (cc) "p[1] %u \n", particles_p[1].u64);
  // printf( (cc) "p[2] %u \n", particles_p[2].u64);
  // printf( (cc) "p[3] %u \n", particles_p[3].u64);
  // printf( (cc) "p[4] %u \n", particles_p[4].u64);
  // printf( (cc) "p[5] %u \n", particles_p[5].u64);

  // printf((cc) "p->npart %d\n", particles->npart);
  // printf((cc) "p->beta0[0] %g\n", particles->beta0[0]);
  // printf((cc) "p->beta0[3] %g\n", particles->beta0[3]);
  // printf((cc) "elembyelem[0] %i \n", elembyelem_p[0].u64);
  // printf((cc) "turnbyturn[0] %i \n", turnbyturn_p[0].u64);
  // printf((cc) "elembyelem[1] %i \n", elembyelem_p[1].u64);
  // printf((cc) "turnbyturn[1] %i \n", turnbyturn_p[1].u64);
  // printf((cc) "elembyelem[2] %i \n", elembyelem_p[2].u64);
  // printf((cc) "turnbyturn[2] %i \n", turnbyturn_p[2].u64);

  if (elembyelem_flag)
    Particles_unpack(elembyelem, elembyelem_p);
  if (turnbyturn_flag)
    Particles_unpack(turnbyturn, turnbyturn_p);

  // printf((cc) "elembyelem[0] %i \n", elembyelem_p[0].u64);
  // printf((cc) "turnbyturn[0] %i \n", turnbyturn_p[0].u64);
  // printf((cc) "elembyelem[1] %i \n", elembyelem_p[1].u64);
  // printf((cc) "turnbyturn[1] %i \n", turnbyturn_p[1].u64);
  // printf((cc) "elembyelem[2] %i \n", elembyelem_p[2].u64);
  // printf((cc) "turnbyturn[2] %i \n", turnbyturn_p[2].u64);
  // printf((cc) "ele->beta0[0] %g\n", elembyelem->beta0[0]);
  // printf((cc) "ele->beta0[3] %g\n", elembyelem->beta0[3]);
};

CLKERNEL void Block_track(CLGLOBAL value_t *elems, CLGLOBAL uint64_t *elemids,
                          uint64_t nelems, uint64_t nturns,
                          CLGLOBAL value_t *particles_p,  // Particles
                          CLGLOBAL value_t *elembyelem_p, // ElembyElem
                          CLGLOBAL value_t *turnbyturn_p) // TurnbyTurn
{
  CLGLOBAL value_t *elem;
  uint64_t elemid;
  uint64_t partid = get_global_id(0);

  Particles *particles = (Particles *)particles_p;

  // printf( "beta0[%d] %g\n",partid, particles->beta0[partid]);

  bool elembyelem_flag = (elembyelem_p[0].i64 != 0);
  bool turnbyturn_flag = (turnbyturn_p[0].i64 != 0);

  Particles *elembyelem = (Particles *)elembyelem_p;
  Particles *turnbyturn = (Particles *)turnbyturn_p;

  if (turnbyturn_flag) {
    Particles_copy(particles, turnbyturn, partid, partid);
  };

  // printf( "tbt->beta0[%d] %g\n",partid, turnbyturn->beta0[partid]);

  uint64_t nparts = particles->npart;
  uint64_t tbt = nparts;
  uint64_t ebe = 0;
  // printf("%g %g %g\n",nparts,tbt,ebe);

  for (int jj = 0; jj < nturns; jj++) {
    for (int ii = 0; ii < nelems; ii++) {
      elemid = elemids[ii];
      // printf("elemid %u\n",elemid);
      elem = elems + elemid;
      track_single(particles, partid, elem);
      if (elembyelem_flag) {
        Particles_copy(particles, elembyelem, partid, ebe + partid);
        ebe += nparts;
      }
    } // end elem loop
    if (turnbyturn_flag) {
      Particles_copy(particles, turnbyturn, partid, tbt + partid);
      tbt += nparts;
    }
  } // end turn loop
}

#else

int Block_track(value_t *elems, Beam *beam, uint64_t blockid) {
  uint64_t nelem = Block_get_nelen(elems, blockid);
  uint64_t *elemids = Block_get_elemids(elems, blockid);
  uint64_t elemid;
  for (int ii = 0; ii < nelem; ii++) {
    elemid = elemids[ii];
    for (uint64_t partid = 0; partid < beam->npart; partid++) {
      track_single(elems, beam->particles, elemid, partid, 0);
    };
  }
  return 1;
}

#endif
