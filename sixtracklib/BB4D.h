#ifndef _BB4D_
#define _BB4D_

#include "BB6D_transverse_fields.h"
#include "constants.h"

#include "block.h"

// To have uint64_t
//#include <stdint.h>

typedef struct{
    double q_part;
    double N_part;
    double sigma_x;
    double sigma_y;
    double beta_s;
    double min_sigma_diff;
    double Delta_x;
    double Delta_y;
}BB4D_data;

// void BB6D_track(double* x, double* px, double* y, double* py, double* sigma, 
//                 double* delta, double q0, double p0, BB6D_data *bb6ddata){

void BB4D_track(Particles *particles, uint64_t partid, CLGLOBAL value_t *bb4ddata_ptr){
   
    CLGLOBAL BB4D_data *bb4ddata = (CLGLOBAL BB4D_data*) bb4ddata_ptr;

    // Get weak-beam particle data
    //double p0 = particles->p0c[partid]*QELEM/C_LIGHT;  
    double q0 = particles->charge0[partid]*QELEM; 
    
    double x     = particles->x[partid] -  bb4ddata->Delta_x;
    double px    = particles->px[partid];
    double y    = particles->y[partid] - bb4ddata->Delta_y;;
    double py   = particles->py[partid];  
    
    double chi = particles->chi[partid]; 
    
    double beta = particles->beta0[partid]/particles->rvv[partid];
    double p0c = particles->p0c[partid]*QELEM;
    
    
    double Ex, Ey, Gx, Gy;
    bool skip_Gs = true;
    get_Ex_Ey_Gx_Gy_gauss(x, y, 
        bb4ddata->sigma_x,
        bb4ddata->sigma_y,
        bb4ddata->min_sigma_diff,
        &Ex, &Ey, skip_Gs, &Gx, &Gy);
        
    double fact_kick = chi * bb4ddata->N_part * bb4ddata->q_part * q0 * (1. + beta * bb4ddata->beta_s)/(p0c*(beta + bb4ddata->beta_s));
    
    px += fact_kick*Ex;
    py += fact_kick*Ey;

    particles->px[partid] = px;
    particles->py[partid] = py;


}

#endif
