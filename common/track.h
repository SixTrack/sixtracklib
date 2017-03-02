#ifndef _TRACK_
#define _TRACK_


#ifndef _GPUCODE
  #include <math.h>
  #define CLGLOBAL
#endif

#include "particle.h"


/******************************************/
/*Drift*/
/******************************************/

typedef struct {
  double length;
} Drift ;

/******************************************/
/*Exact Drift*/
/******************************************/

typedef struct {
  double length;
} DriftExact ;

/******************************************/
/*Multipole*/
/******************************************/

typedef struct {
  long int order;
  double l ;
  double hxl;
  double hyl;
  double bal[1];
} Multipole;

/******************************************/
/*RF Cavity*/
/******************************************/

typedef struct {
  double volt;
  double freq;
  double lag;
} Cavity;

/******************************************/
/*Align*/
/******************************************/

typedef struct {
  double cz;
  double sz;
  double dx;
  double dy;
} Align;

/******************************************/
/*Linear Map*/
/******************************************/

typedef struct {
      double matrix[8];
      //double disp[4];
} LinMap_data;

_CUDA_HOST_DEVICE_
LinMap_data LinMap_init( double alpha_x_s0, double beta_x_s0, double alpha_x_s1, double beta_x_s1,
                         double alpha_y_s0, double beta_y_s0, double alpha_y_s1, double beta_y_s1,
                         double dQ_x, double dQ_y );
                         
_CUDA_HOST_DEVICE_
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

_CUDA_HOST_DEVICE_
int BB4D_track(CLGLOBAL Particle* p, CLGLOBAL BB4D_data *el);


#endif
