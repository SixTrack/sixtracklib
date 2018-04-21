#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include <xmmintrin.h>
#include <emmintrin.h>

#include "sixtracklib/sixtracklib.h"

void Particles_init_with_some_values( st_Particles* particles );

#define ALIGN 64u

int main( int argc, char* argv[] )
{
    size_t ii = 0;
        
    size_t const NUM_ELEMS     = 100;
    size_t const NUM_PARTICLES = 100000;
    size_t const NUM_TURNS     = 100;
    
    struct timeval tstart;
    struct timeval tstop;
    
    st_Particles* particles = 0;
    double* drift_lengths = 0;
    
    if( argc >= 1 )
    {
        printf( "\r\n\r\nUsage: %s NUM_PARTICLES  NUM_ELEMS  NUM_TURNS\r\n", argv[ 0 ] );        
    }
    
    /*
    if( argc < 2 )
    {
        printf( "run with default values .... "
                "(npart = %8lu, nelem = %8lu, nturns = %8lu)\r\n",
                NUM_PARTICLES, NUM_ELEMS, NUM_TURNS );
    }
    else if( argc >= 4 )
    {
        size_t const temp_num_part = atoi( argv[ 1 ] );
        size_t const temp_num_elem = atoi( argv[ 2 ] );
        size_t const temp_num_turn = atoi( argv[ 3 ] );
        
        if( temp_num_part > 0 ) NUM_PARTICLES = temp_num_part;
        if( temp_num_elem > 0 ) NUM_ELEMS     = temp_num_elem;
        if( temp_num_turn > 0 ) NUM_TURNS     = temp_num_turn;                
    }
    */
    
    printf( "\r\n" "starting :                  "
                "(npart = %8lu, nelem = %8lu, nturns = %8lu)\r\n",
                NUM_PARTICLES, NUM_ELEMS, NUM_TURNS );
    
    particles = st_Particles_new_aligned( NUM_PARTICLES, ALIGN );
    drift_lengths = ( double* )malloc( sizeof( double ) * NUM_ELEMS );
    
    __builtin_assume_aligned( particles->q0,     ALIGN );
    __builtin_assume_aligned( particles->mass0,  ALIGN );
    __builtin_assume_aligned( particles->beta0,  ALIGN );
    __builtin_assume_aligned( particles->gamma0, ALIGN );
    __builtin_assume_aligned( particles->p0c,    ALIGN );
    __builtin_assume_aligned( particles->s,      ALIGN );
    __builtin_assume_aligned( particles->x,      ALIGN );
    __builtin_assume_aligned( particles->y,      ALIGN );
    __builtin_assume_aligned( particles->px,     ALIGN );
    __builtin_assume_aligned( particles->py,     ALIGN );
    __builtin_assume_aligned( particles->sigma,  ALIGN );
    __builtin_assume_aligned( particles->psigma, ALIGN );
    __builtin_assume_aligned( particles->delta,  ALIGN );
    __builtin_assume_aligned( particles->rpp,    ALIGN );
    __builtin_assume_aligned( particles->rvv,    ALIGN );
    __builtin_assume_aligned( particles->chi,    ALIGN );
    
    /* init the particles and blocks with some admissible values: */
    
    Particles_init_with_some_values( particles );    
        
    for( ii = 0 ; ii < NUM_ELEMS ; ++ii )
    {
        drift_lengths[ ii ] = 0.0 + 1e-3 * ii;
    }
            
    /* track over a number of turns and measure the wall-time for the tracking: */
    
    gettimeofday( &tstart, 0 );
    
    for( ii = 0 ; ii < NUM_TURNS ; ++ii )
    {
        size_t jj;
        
        for( jj = 0 ; jj < NUM_ELEMS ; ++jj )        
        {
            size_t ip;
            
            double const length = drift_lengths[ jj ];
            
            for( ip = 0 ; ip < NUM_PARTICLES ; ++ip )
            {
                st_Track_drift( particles, ip, length );                
            }                                    
        }
    }
    
    gettimeofday( &tstop, 0 );
    
    double const usec_dist = 1e-6 * ( ( tstop.tv_sec >= tstart.tv_sec ) ?
        ( tstop.tv_usec - tstart.tv_usec ) : ( 1000000 - tstart.tv_usec ) );
    
    double const delta_time = ( double )( tstop.tv_sec - tstart.tv_sec ) + usec_dist;
    printf( "time elapsed    : %16.8fs \r\n", delta_time );
    printf( "time / particle : %16.8f [us/Part]\r\n", 
                ( delta_time / NUM_PARTICLES ) * 1e6 );
    
    /* cleanup: */
              
    st_Particles_free( particles );
    free( particles );
    particles = 0;
    
    free( drift_lengths );
    drift_lengths = 0;
    
    return 0;
}

/* ------------------------------------------------------------------------- */

#include <assert.h>

void Particles_init_with_some_values( st_Particles* particles )
{
    size_t const NUM_PARTICLES = st_Particles_get_size( particles );
    
    double const Q0      = 1.0;
    double const MASS0   = 1.0;
    double const BETA0   = 1.0;
    double const GAMMA0  = 1.0;
    double const P0C     = 0.0;
        
    double const S       = 0.0;
    double const MIN_X   = 0.0;
    double const MAX_X   = 0.3;
    double const DX      = ( MAX_X - MIN_X ) / ( NUM_PARTICLES - 1 );
    
    double const MIN_Y   = 0.0;
    double const MAX_Y   = 0.3;    
    double const DY      = ( MAX_Y - MIN_Y ) / ( NUM_PARTICLES - 1 );
    
    double const PX      = 0.1;
    double const PY      = 0.1;
    double const SIGMA   = 0.0;
    
    double const PSIGMA  = 0.0;    
    double const RPP     = 1.0;
    double const RVV     = 1.0;
    double const DELTA   = 0.5;
    double const CHI     = 0.0;
    
    int64_t const STATE  = INT64_C( 0 );
    
    size_t ii; 
    int64_t particle_id = 0;
        
    for( ii = 0 ; ii < NUM_PARTICLES ; ++ii, ++particle_id )
    {
        st_Particles_set_q0_value(     particles, ii, Q0 );
        st_Particles_set_mass0_value(  particles, ii, MASS0 );
        st_Particles_set_beta0_value(  particles, ii, BETA0 );
        st_Particles_set_gamma0_value( particles, ii, GAMMA0 );
        st_Particles_set_p0c_value(    particles, ii, P0C );
        
        st_Particles_set_particle_id_value(        particles, ii, particle_id );
        st_Particles_set_lost_at_element_id_value( particles, ii, -1 );
        st_Particles_set_lost_at_turn_value(       particles, ii, -1 );
        st_Particles_set_state_value(              particles, ii, STATE );
        
        st_Particles_set_s_value(      particles, ii, S );
        st_Particles_set_x_value(      particles, ii, MIN_X + DX * ii );
        st_Particles_set_y_value(      particles, ii, MIN_Y + DY * ii );
        st_Particles_set_px_value(     particles, ii, PX );
        st_Particles_set_py_value(     particles, ii, PY );
        st_Particles_set_sigma_value(  particles, ii, SIGMA );
        
        st_Particles_set_psigma_value( particles, ii, PSIGMA );
        st_Particles_set_delta_value(  particles, ii, DELTA );
        st_Particles_set_rpp_value(    particles, ii, RPP );
        st_Particles_set_rvv_value(    particles, ii, RVV );
        st_Particles_set_chi_value(    particles, ii, CHI );
    }
    
    return;
}

/* end:  sixtracklib/examples/simple_drift.c */
