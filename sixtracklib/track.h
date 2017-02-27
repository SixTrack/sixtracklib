#ifndef _TRACK
#define _TRACK


#define __CUDA_HOST_DEVICE__

#ifndef _GPUCODE
  #include <math.h>
  #define CLGLOBAL
#endif

#include "particle.h"


/*********************************/

typedef struct {
    double length;
} Drift ;

typedef struct {
      double length;
} DriftExact ;

typedef struct {
      long int order;
      double l ;
      double hxl;
      double hyl;
      double bal[1];
} Multipole;

/*********************************/
// **** NOT YET ADAPTED TO NEW STRUCTURE!!!
__CUDA_HOST_DEVICE__
int Drift_track(CLGLOBAL Particle* p, CLGLOBAL Drift *el);

__CUDA_HOST_DEVICE__
int DriftExact_track(CLGLOBAL Particle* p, double length);

__CUDA_HOST_DEVICE__
int Multipole_track(CLGLOBAL Particle* p, CLGLOBAL Multipole *el);

__CUDA_HOST_DEVICE__
int Cavity_track(CLGLOBAL Particle* p, double volt, double freq, double lag );
__CUDA_HOST_DEVICE__
int Align_track(CLGLOBAL Particle* p, double cz, double sz,
                                      double dx, double dy);
/******************************************/

// ADAPTED TO NEW STRUCTURE:


/******************************************/
/*LINEAR MAP*/
/******************************************/

typedef struct {
      double matrix[8];
      //double disp[4];
} LinMap_data;

__CUDA_HOST_DEVICE__
LinMap_data LinMap_init( double alpha_x_s0, double beta_x_s0, double alpha_x_s1, double beta_x_s1,
                         double alpha_y_s0, double beta_y_s0, double alpha_y_s1, double beta_y_s1,
                         double dQ_x, double dQ_y );
                         
__CUDA_HOST_DEVICE__
int LinMap_track(CLGLOBAL Particle* p, CLGLOBAL LinMap_data *el);


/******************************************/
/*Beam-beam 4d*/
/******************************************/

#include "transverse_field_gauss_round.h"
#include "transverse_field_gauss_ellip.h"

typedef struct {
    double N_s; // Population strong beam
    double beta_s;
    double q_s;
    long int trasv_field_type; //1: round gaussian
    CLGLOBAL void* field_map_data;
} BB4D_data;



__CUDA_HOST_DEVICE__
int BB4D_track(CLGLOBAL Particle* p, CLGLOBAL BB4D_data *el);




#endif
