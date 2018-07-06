#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.h"

int main( int argc, char* argv[] )
{
    st_block_num_elements_t jj = 0;

    st_block_size_t PARTICLES_DATA_CAPACITY = 0u;
    st_block_size_t DRIFTS_DATA_CAPACITY    = 0u;
    st_block_size_t NUM_OF_BLOCKS           = 64u;
    st_block_size_t THREADS_PER_BLOCK       = 16u;

    st_block_num_elements_t NUM_ELEMS       = 100;
    st_block_num_elements_t NUM_PARTICLES   = 100000;
    size_t NUM_TURNS     = 100;

    struct timeval tstart;
    struct timeval tstop;

    st_Particles* particles = 0;
    st_Blocks     particles_buffer;
    st_Blocks     beam_elements;

    int ret = 0;

    /* first: init the pseudo random number generator for the particle
     * values randomization - choose a constant seed to have reproducible
     * results! */

    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );

    if( argc >= 1 )
    {
        printf( "\r\n\r\nUsage: %s NUM_PARTICLES  NUM_ELEMS  NUM_TURNS "
                "NUM_OF_BLOCKS THREADS_PER_BLOCK \r\n",
                argv[ 0 ] );
    }

    if( argc >= 2 )
    {
        st_block_num_elements_t const temp_num_part = atoi( argv[ 1 ] );
        if( temp_num_part > 0 ) NUM_PARTICLES = temp_num_part;
    }
    else
    {
        printf( " -> using default value for NUM_PARTICLES     = %8lu\r\n",
                NUM_PARTICLES );

    }

    if( argc >= 3 )
    {
        st_block_num_elements_t const temp_num_elem = atoi( argv[ 2 ] );
        if( temp_num_elem > 0 ) NUM_ELEMS = temp_num_elem;
    }
    else
    {
        printf( " -> using default value for NUM_ELEMS         = %8lu\r\n",
                NUM_ELEMS );
    }

    if( argc >= 4 )
    {
        size_t const temp_num_turn = atoi( argv[ 3 ] );
        if( temp_num_turn > 0 ) NUM_TURNS = temp_num_turn;
    }
    else
    {
        printf( " -> using default value for NUM_TURNS         = %8lu\r\n",
                NUM_TURNS );
    }

    if( argc >= 5 )
    {
        size_t const temp_num_blocks = atoi( argv[ 4 ] );
        if( temp_num_blocks > 0 ) NUM_OF_BLOCKS = temp_num_blocks;
    }
    else
    {
        printf( " -> using default value for NUM_OF_BLOCKS     = %8lu\r\n",
                NUM_OF_BLOCKS );
    }

    if( argc >= 6 )
    {
        size_t const temp_threads_per_block = atoi( argv[ 5 ] );

        if( temp_threads_per_block > 0 )
        {
            THREADS_PER_BLOCK = temp_threads_per_block;
        }
    }
    else
    {
        printf( " -> using default value for THREADS_PER_BLOCK = %8lu\r\n",
                THREADS_PER_BLOCK );
    }

    printf( "\r\n"
            "Starting with values : \r\n"
            "NUM_PARTICLES     = %8lu\r\n"
            "NUM_ELEMS         = %8lu\r\n"
            "NUM_TURNS         = %8lu\r\n"
            "NUM_OF_BLOCKS     = %8lu\r\n"
            "THREADS_PER_BLOCK = %8lu\r\n",
            NUM_PARTICLES, NUM_ELEMS, NUM_TURNS,
            NUM_OF_BLOCKS, THREADS_PER_BLOCK );

    /* --------------------------------------------------------------------- */
    /* particles: */

    st_Blocks_preset( &particles_buffer );
    PARTICLES_DATA_CAPACITY = st_Particles_predict_blocks_data_capacity(
        &particles_buffer, 1u, NUM_PARTICLES );

    st_Blocks_init( &particles_buffer, 1u, PARTICLES_DATA_CAPACITY );
    particles = st_Blocks_add_particles( &particles_buffer, NUM_PARTICLES );

    printf( "\r\n" "\r\nAdding Particles: \r\n" );
    st_Particles_random_init( particles );
    st_Blocks_serialize( &particles_buffer );

    printf( " -> Added %6ld particles in %6ld blocks \r\n\r\n",
            NUM_PARTICLES, st_Blocks_get_num_of_blocks( &particles_buffer ) );

    /* --------------------------------------------------------------------- */
    /* beam_elements: */

    st_Blocks_preset( &beam_elements );

    DRIFTS_DATA_CAPACITY = st_Blocks_predict_data_capacity_for_num_blocks(
            &beam_elements, NUM_ELEMS ) +
        st_Drift_predict_blocks_data_capacity( &beam_elements, NUM_ELEMS );

    st_Blocks_init( &beam_elements, NUM_ELEMS, DRIFTS_DATA_CAPACITY );

    printf( "\r\n" "\r\nAdding Beam Elements: \r\n" );

    for( jj = 0 ; jj < NUM_ELEMS ; ++jj )
    {
        static SIXTRL_REAL_T const MIN_DRIFT_LEN = ( SIXTRL_REAL_T )0.005;
        static SIXTRL_REAL_T const MAX_DRIFT_LEN = ( SIXTRL_REAL_T )1.000;
        SIXTRL_REAL_T const        DELTA_LEN = MAX_DRIFT_LEN - MIN_DRIFT_LEN;

        SIXTRL_REAL_T const drift_length =
            MIN_DRIFT_LEN + DELTA_LEN * st_Random_genrand64_real1();

        st_Drift* drift = st_Blocks_add_drift( &beam_elements, drift_length );

        printf( " -> added drift #%6ld / %6ld with length = %.8f\r\n",
                 jj, NUM_ELEMS, st_Drift_get_length( drift ) );
    }

    st_Blocks_serialize( &beam_elements );

    /* --------------------------------------------------------------------- */
    /* track over a num of turns and measure the wall-time for the tracking: */

    printf( "\r\nStart tracking for %lu turns:\r\n", NUM_TURNS );

    gettimeofday( &tstart, 0 );

    ret = st_Track_particles_on_cuda(
        NUM_OF_BLOCKS, THREADS_PER_BLOCK, NUM_TURNS,
            &particles_buffer, &beam_elements, 0 );

    gettimeofday( &tstop, 0 );

    if( ret == 0 )
    {
        double const usec_dist = 1e-6 * ( ( tstop.tv_sec >= tstart.tv_sec )
            ? ( tstop.tv_usec - tstart.tv_usec )
            : ( 1000000 - tstart.tv_usec ) );

        double const delta_time =
            ( double )( tstop.tv_sec - tstart.tv_sec ) + usec_dist;

        printf( "\r\n"
                " -> time elapsed    : %16.8fs \r\n", delta_time );
        printf( " -> time / particle : %16.8f [us/Part]\r\n",
                    ( delta_time / NUM_PARTICLES ) * 1e6 );

    }
    else
    {
        printf( "\r\n" "-> Error while running tracking code "
                           "[error_code=%d]\r\n", ret );
    }

    /* --------------------------------------------------------------------- */
    /* cleanup: */

    st_Blocks_free( &particles_buffer );
    st_Blocks_free( &beam_elements );

    return 0;
}

/* end:  examples/c99/simple_drift_cuda.c */
