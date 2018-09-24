#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.h"

int main( int argc, char* argv[] )
{
    typedef st_buffer_size_t buf_size_t;

    st_Buffer* lhc_particles_buffer = st_Buffer_new_from_file(
        st_PATH_TO_TEST_LHC_PARTICLES_DATA_T1_P2_NO_BEAM_BEAM );

    st_Buffer* lhc_beam_elements_buffer = st_Buffer_new_from_file(
        st_PATH_TO_TEST_LHC_BEAM_ELEMENTS_DATA_NO_BEAM_BEAM );

    st_Buffer* pb = st_Buffer_new( ( buf_size_t )( 1u << 24u ) );

    buf_size_t NUM_PARTICLES                = 20000;
    buf_size_t NUM_TURNS                    = 20;
    int        SELECTED_DEVICE_ID           = -1;

    st_Particles*       particles           = SIXTRL_NULLPTR;
    st_Particles const* input_particles     = SIXTRL_NULLPTR;
    buf_size_t          num_input_particles = 0;

    buf_size_t ii = 0;

    /* ********************************************************************** */
    /* ****   Handling of command line parameters                             */
    /* ********************************************************************** */

    int success = -1;
    ( void )success;

    int total_num_devices = 0;
    struct cudaDeviceProp* device_properties = SIXTRL_NULLPTR;
    char** device_id_str_list = SIXTRL_NULLPTR;

    cudaError_t cu_err = cudaGetDeviceCount( &total_num_devices );
    ( void ) cu_err;

    if( total_num_devices > 0 )
    {
        int device_id = 0;

        device_id_str_list = ( char** )malloc(
            sizeof( char* ) * total_num_devices );

        device_properties  = ( struct cudaDeviceProp* )malloc(
            sizeof( struct cudaDeviceProp ) * total_num_devices );

        SIXTRL_ASSERT( device_id_str_list != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( device_properties != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( total_num_devices <  1000 );

        for( ; device_id < total_num_devices ; ++device_id )
        {
            device_id_str_list[ device_id ] = ( char* )malloc(
                sizeof( char ) * 4 );

            cu_err = cudaGetDeviceProperties(
                &device_properties[ device_id ], device_id );

            SIXTRL_ASSERT( device_id_str_list[ device_id ] != SIXTRL_NULLPTR );
            SIXTRL_ASSERT( cu_err == cudaSuccess );

            sprintf( device_id_str_list[ device_id ], "%lu",
                     ( unsigned long )( device_id % 1000 ) );
        }
    }

    /* ---------------------------------------------------------------------- */
    /* Handle input parameters: */

    if( argc == 1 )
    {
        printf( "Usage: %s [ID] [NUM_PARTICLES] [NUM_TURNS]\r\n", argv[ 0 ] );

        printf( "\r\n"
                "ID           : ID of the CUDA device to run the test on\r\n" );

        if( total_num_devices > 0 )
        {
            printf( "               %3lu devices found on the system\r\n"
                    "               Default = %4s\r\n"
                    "\r\n", ( unsigned long )total_num_devices,
                    device_id_str_list[ 0 ] );
        }
        else
        {
            printf( "               no devices found on the system!! "
                    "               -> Stopping\r\n" );

            return 0;
        }

        printf( "\r\n"
                "NUM_PARTICLES: Number of particles for the simulation\r\n"
                "               Default = %lu\r\n",
                ( unsigned long )NUM_PARTICLES );

        printf( "\r\n"
                "NUM_TURNS    : Number of turns for the simulation\r\n"
                "               Default = %lu\r\n",
                ( unsigned long )NUM_TURNS );

        if( total_num_devices > 0 )
        {
            int device_id = 0;

            printf( "-------------------------------------------------------"
                    "---------------------------------------------------\r\n"
                    "List of available device IDs : \r\n"
                    "\r\n" );

            for( ; device_id < total_num_devices ; ++device_id )
            {
                printf( "%4s          : %s\r\n"
                        "               compute capability %d.%d\r\n"
                        "               num multiprocessors %d\r\n"
                        "               max threads per block %d\r\n"
                        "               max grid size %d / %d / %d\r\n"
                        "               warp size %d\r\n"
                        "\r\n",
                        device_id_str_list[ device_id ],
                        device_properties[ device_id ].name,
                        ( int )device_properties[ device_id ].major,
                        ( int )device_properties[ device_id ].minor,
                        ( int )device_properties[ device_id ].multiProcessorCount,
                        ( int )device_properties[ device_id ].maxThreadsPerBlock,
                        ( int )device_properties[ device_id ].maxGridSize[ 0 ],
                        ( int )device_properties[ device_id ].maxGridSize[ 1 ],
                        ( int )device_properties[ device_id ].maxGridSize[ 2 ],
                        ( int )device_properties[ device_id ].warpSize );
            }
        }
    }

    SIXTRL_ASSERT( total_num_devices > 0 );

    if( argc >= 2 )
    {
        int test_device_id = 0 ;
        SELECTED_DEVICE_ID = -1;

        for( ; test_device_id < total_num_devices ; ++test_device_id )
        {
            if( strcmp( argv[ 1 ], device_id_str_list[ test_device_id ] ) == 0 )
            {
                SELECTED_DEVICE_ID = test_device_id;
                break;
            }
        }

        if( SELECTED_DEVICE_ID < 0 )
        {
            SELECTED_DEVICE_ID = 0;

            printf( "Device ID %s not avilable -> using default device instead %s\r\n",
                    argv[ 1 ], device_id_str_list[ SELECTED_DEVICE_ID ] );
        }
    }



    if( argc >= 3 )
    {
        int temp = atoi( argv[ 2 ] );

        if( temp > 0 )
        {
            NUM_PARTICLES = ( buf_size_t )temp;
        }
    }

    if( argc >= 4 )
    {
        int temp = atoi( argv[ 3 ] );

        if( temp > 0 )
        {
            NUM_TURNS = ( uint64_t )temp;
        }
    }

    cu_err = cudaSetDevice( SELECTED_DEVICE_ID );
    SIXTRL_ASSERT( cu_err == cudaSuccess );

    printf( "Selected Device ID     = %4s ( %s )\r\n"
            "Selected NUM_PARTICLES = %10lu\r\n"
            "Selected NUM_TURNS     = %10lu\r\n"
            "\r\n",
            device_id_str_list[ SELECTED_DEVICE_ID ],
            device_properties[  SELECTED_DEVICE_ID ].name,
            NUM_PARTICLES, NUM_TURNS );

    /* ********************************************************************** */
    /* ****   Building Particles Data from LHC Particle Dump Data        **** */
    /* ********************************************************************** */

    particles = st_Particles_new( pb, NUM_PARTICLES );
    input_particles = st_Particles_buffer_get_const_particles( lhc_particles_buffer, 0u );
    num_input_particles = st_Particles_get_num_of_particles( input_particles );

    for( ii = 0 ; ii < NUM_PARTICLES ; ++ii )
    {
        buf_size_t const jj = ii % num_input_particles;
        st_Particles_copy_single( particles, ii, input_particles, jj );
    }

    success = st_Track_particles_in_place_on_cuda(
        pb, lhc_beam_elements_buffer, NUM_TURNS );


    /* ********************************************************************** */
    /* ****                         Clean-up                             **** */
    /* ********************************************************************** */

    st_Buffer_delete( pb );
    st_Buffer_delete( lhc_particles_buffer );
    st_Buffer_delete( lhc_beam_elements_buffer );
}

/* end: examples/c99/track_lhc_no_bb_cuda.c */
