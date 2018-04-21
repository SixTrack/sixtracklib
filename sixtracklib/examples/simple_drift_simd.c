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

int main( int argc, char* argv[] )
{
    size_t ii = 0;
        
    size_t NUM_ELEMS     = 100;
    size_t NUM_PARTICLES = 100000;
    size_t NUM_TURNS     = 100;
    
    struct timeval tstart;
    struct timeval tstop;
    
    st_Particles* particles = 0;
    double* drift_lengths = 0;
    
    /* first: init the pseudo random number generator for the particle
     * values randomization - choose a constant seed to have reproducible
     * results! */
    
    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );
    
    if( argc >= 1 )
    {
        printf( "\r\n\r\nUsage: %s NUM_PARTICLES  NUM_ELEMS  NUM_TURNS\r\n", argv[ 0 ] );        
    }
    
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
    
    printf( "\r\n" "starting :                   "
                "(npart = %8lu, nelem = %8lu, nturns = %8lu)\r\n",
                NUM_PARTICLES, NUM_ELEMS, NUM_TURNS );
    
    #if defined( __AVX__ )
    printf( "Info: using avx  implementation\r\n" );
    #elif defined( __SSE2__ )    
    printf( "Info: using sse2 implementation\r\n" );
    #else
    #error Undefined/illegal architecture selected for this example -> check your compiler flags
    #endif /* CPU architecture */
    
    particles = st_Particles_new_aligned( NUM_PARTICLES, SIXTRL_ALIGN );
    drift_lengths = ( double* )malloc( sizeof( double ) * NUM_ELEMS );
    
    /* init the particles and blocks with some admissible randomized values: */    
    st_Particles_random_init( particles );
        
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
            double const length = drift_lengths[ jj ];
            
            #if defined( __AVX__ )
            st_Track_simd_drift_avx( particles, length );
            #elif defined( __SSE2__ )
            st_Track_simd_drift_sse2( particles, length );
            #endif /* defined( __SSE2__ ) */
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

/* end:  sixtracklib/examples/simple_drift.c */
