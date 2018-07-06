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

    st_ComputeNodeId selected_compute_id;

    st_block_size_t PARTICLES_DATA_CAPACITY = 0u;
    st_block_size_t DRIFTS_DATA_CAPACITY    = 0u;

    st_block_num_elements_t NUM_ELEMS       = 100;
    st_block_num_elements_t NUM_PARTICLES   = 100000;
    size_t NUM_TURNS     = 100;

    struct timeval tstart;
    struct timeval tstop;

    st_Particles* particles = 0;
    st_Blocks     particles_buffer;
    st_Blocks     beam_elements;

    st_OclEnvironment* ocl_env = st_OclEnvironment_new();

    int ret = 0;

    /* first: init the pseudo random number generator for the particle
     * values randomization - choose a constant seed to have reproducible
     * results! */

    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );

    /* --------------------------------------------------------------------- */
    /* present list of available computing devices                           */

    st_ComputeNodeId_preset( &selected_compute_id );

    if( ( argc == 1 ) &&
        ( ocl_env != 0 ) &&
        ( st_OclEnvironment_get_num_available_nodes( ocl_env ) ) )
    {
        bool first = true;

        st_ComputeNodeId const* nodes_it =
            st_OclEnvironment_get_available_nodes_begin( ocl_env );

        st_ComputeNodeId const* nodes_end   =
            st_OclEnvironment_get_available_nodes_end( ocl_env );

        printf( "Run without arguments -> "
            "list all available OpenCL devices: \r\n"
            "\r\n"
            "\r\n"
            "   ID   | Platform / Name / Extensions \r\n"
            "-----------------------------------------------------------------"
            "-----------------------------------------------------------------"
            "\r\n" );

        for( ; nodes_it != nodes_end ; ++nodes_it )
        {
            st_ComputeNodeInfo const* info =
                st_OclEnvironment_get_ptr_node_info( ocl_env, nodes_it );

            char const* ptr_name =
                st_ComputeNodeInfo_get_name( info );

            char const* ptr_platform =
                st_ComputeNodeInfo_get_platform( info );

            char const* ptr_description =
                st_ComputeNodeInfo_get_description( info );

            const char NA_STR[] = "n/a";

            if( first )
            {
                st_ComputeNodeId_set_platform_id( &selected_compute_id,
                    st_ComputeNodeInfo_get_platform_id( info ) );

                st_ComputeNodeId_set_device_id( &selected_compute_id,
                    st_ComputeNodeInfo_get_device_id( info ) );

                first = false;
            }

            if( ptr_platform    == 0 ) ptr_platform    = &NA_STR[ 0 ];
            if( ptr_name        == 0 ) ptr_name        = &NA_STR[ 0 ];
            if( ptr_description == 0 ) ptr_description = &NA_STR[ 0 ];

            printf( "%3ld.%-3ld | %s\r\n"
                    "        | %s\r\n"
                    "        | %s\r\n\r\n",
                    st_ComputeNodeInfo_get_platform_id( info ),
                    st_ComputeNodeInfo_get_device_id( info ),
                    ptr_platform, ptr_name, ptr_description );
        }

        if( first )
        {
            st_ComputeNodeId_set_platform_id( &selected_compute_id, 0 );
            st_ComputeNodeId_set_device_id(   &selected_compute_id, 0 );
        }

        printf( "\r\n" "Program usage: %s ID "
                "[NUM_PARTICLES] [NUM_ELEMS] [NUM_TURNS]\r\n"
                "\r\n"
                "omitting these values, the default values below are used:"
                "\r\n"
                "ID            = %3ld.%-3ld\r\n"
                "NUM_PARTICLES = %8lu\r\n"
                "NUM_ELEMS     = %8lu\r\n"
                "NUM_TURNS     = %8lu\r\n",
                argv[0],
                st_ComputeNodeId_get_platform_id( &selected_compute_id ),
                st_ComputeNodeId_get_device_id( &selected_compute_id ),
                NUM_PARTICLES, NUM_ELEMS, NUM_TURNS );
    }
    else if( ( ocl_env == 0 ) ||
             ( st_OclEnvironment_get_num_available_nodes( ocl_env ) ) )
    {
        printf( "\r\n"
                "No OpenCL computing devices available\r\n\r\n" );

        st_OclEnvironment_free( ocl_env );
        ocl_env = 0;

        return 0;
    }

    if( argc >= 1 )
    {
        printf( "\r\n\r\nUsage: %s ID "
                "NUM_PARTICLES  NUM_ELEMS  NUM_TURNS\r\n", argv[ 0 ] );
    }

    if( argc >= 2 )
    {
        st_ComputeNodeId temp;

        int platform_id = -1;
        int device_id   = -1;

        sscanf( argv[ 1 ], "%d.%d", &platform_id, &device_id );
        st_ComputeNodeId_set_platform_id( &temp, platform_id );
        st_ComputeNodeId_set_device_id( &temp, device_id );

        if( 0 != st_OclEnvironment_get_ptr_node_info( ocl_env, &temp ) )
        {
            selected_compute_id = temp;
        }
    }
    else
    {
        printf( " -> using default value for ID            = %3ld.%-3ld\r\n",
                st_ComputeNodeId_get_platform_id( &selected_compute_id ),
                st_ComputeNodeId_get_device_id( &selected_compute_id ) );
    }

    if( argc >= 3 )
    {
        st_block_num_elements_t const temp_num_part = atoi( argv[ 2 ] );
        if( temp_num_part > 0 ) NUM_PARTICLES = temp_num_part;
    }
    else
    {
        printf( " -> using default value for NUM_PARTICLES = %8lu\r\n",
                NUM_PARTICLES );

    }

    if( argc >= 4 )
    {
        st_block_num_elements_t const temp_num_elem = atoi( argv[ 3 ] );
        if( temp_num_elem > 0 ) NUM_ELEMS = temp_num_elem;
    }
    else
    {
        printf( " -> using default value for NUM_ELEMS     = %8lu\r\n",
                NUM_ELEMS );
    }

    if( argc >= 5 )
    {
        size_t const temp_num_turn = atoi( argv[ 4 ] );
        if( temp_num_turn > 0 ) NUM_TURNS = temp_num_turn;
    }
    else
    {
        printf( " -> using default value for NUM_TURNS     = %8lu\r\n",
                NUM_TURNS );
    }

    printf( "\r\n"
            "Starting with values : \r\n"
            "ID            = %3ld.%-3ld\r\n"
            "NUM_PARTICLES = %8lu\r\n"
            "NUM_ELEMS     = %8lu\r\n"
            "NUM_TURNS     = %8lu\r\n",
            st_ComputeNodeId_get_platform_id( &selected_compute_id ),
            st_ComputeNodeId_get_device_id( &selected_compute_id ),
            NUM_PARTICLES, NUM_ELEMS, NUM_TURNS );

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

    printf( "\r\nPrepare kernel for tracking:\r\n" );

    gettimeofday( &tstart, 0 );

    ret = st_OclEnvironment_prepare_particles_tracking(
        ocl_env, &particles_buffer, &beam_elements, 0,
            &selected_compute_id, 1u );

    gettimeofday( &tstop, 0 );

    if( ret == 0 )
    {
        double usec_dist = 1e-6 * ( ( tstop.tv_sec >= tstart.tv_sec )
            ? ( tstop.tv_usec - tstart.tv_usec )
            : ( 1000000 - tstart.tv_usec ) );

        double delta_time =
            ( double )( tstop.tv_sec - tstart.tv_sec ) + usec_dist;

        printf( " -> time elapsed    : %16.8fs \r\n", delta_time );
    }
    else
    {
        printf( "\r\n" "-> Error while preparing for tracking "
                           "[error_code=%d]\r\n", ret );

        st_Blocks_free( &particles_buffer );
        st_Blocks_free( &beam_elements );
        st_OclEnvironment_free( ocl_env );

        return 0;
    }

    printf( "\r\nStart tracking for %lu turns:\r\n", NUM_TURNS );
    gettimeofday( &tstart, 0 );

    ret = st_OclEnvironment_run_particle_tracking(
        ocl_env, NUM_TURNS, &particles_buffer, &beam_elements, 0 );

    gettimeofday( &tstop, 0 );

    if( ret == 0 )
    {
        double const usec_dist = 1e-6 * ( ( tstop.tv_sec >= tstart.tv_sec )
            ? ( tstop.tv_usec - tstart.tv_usec )
            : ( 1000000 - tstart.tv_usec ) );

        double const delta_time =
            ( double )( tstop.tv_sec - tstart.tv_sec ) + usec_dist;

        printf( " -> time elapsed    : %16.8fs \r\n", delta_time );
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
    st_OclEnvironment_free( ocl_env );

    return 0;
}

/* end:  examples/c99/simple_drift_opencl.c */
