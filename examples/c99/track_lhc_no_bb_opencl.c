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
    st_CLContextBase* context = st_CLContextBase_create();

    int NUM_TURNS     = 20;
    int NUM_PARTICLES = 20000;

    st_context_node_id_t device_id;
    st_ComputeNodeId_preset( &device_id );

    /* --------------------------------------------------------------------- */
    /* Handle command line arguments: */
    /* --------------------------------------------------------------------- */

    if( argc < 2  )
    {
        unsigned int const num_devices =
            st_CLContextBase_get_num_available_nodes( context );

        printf( "Usage: %s [ID] [NUM_PARTICLES] [NUM_TURNS]\r\n", argv[ 0 ] );

        if( num_devices > 0u )
        {
            char id_str[ 16 ];
            unsigned int ii = 0u;

            st_context_node_info_t const* nodes_info =
                st_CLContextBase_get_available_nodes_info_begin( context );

            printf( "\r\n"
                    "[ID]            :  "
                    "ID of the OpenCL device to use for the tracking\r\n" );

            for( ; ii < num_devices ; ++ii )
            {
                st_ComputeNodeId_to_string(
                    &( nodes_info[ ii ].id ), &id_str[ 0 ], 16 );

                printf( "%-10s      :: %s\r\n"
                        "                :: %s\r\n"
                        "                :: %s\r\n"
                        "\r\n", id_str, nodes_info[ ii ].name,
                        nodes_info[ ii ].platform, nodes_info[ ii ].description );
            }

            device_id = nodes_info[ 0 ].id;
        }
        else
        {
            printf( "No OpenCL Devices found -> quitting program!\r\n" );
            return 0;
        }

        printf( "\r\n"
                "[NUM_PARTICLES] :  Number of particles for the simulation\r\n"
                "                :: Default = %d\r\n", NUM_PARTICLES );

        printf( "\r\n"
                "[NUM_TURNS]     :  Number of turns for the simulation\r\n"
                "                :: Default = %d\r\n", NUM_TURNS );
    }

    if( argc >= 2 )
    {
        st_context_node_id_t sel_node_id;
        st_ComputeNodeId_preset( &sel_node_id );

        if( ( 0 == st_ComputeNodeId_from_string( &sel_node_id, argv[ 1 ] ) ) &&
            ( st_CLContextBase_is_node_id_available( context, &sel_node_id ) ) )
        {
            device_id = sel_node_id;
        }
        else
        {
            st_context_node_info_t const* nodes_info =
                st_CLContextBase_get_available_nodes_info_begin( context );

            printf( "Warning         : Provided ID %s not found "
                    "-> use the first device instead\r\n",
                    argv[ 1 ] );

            if( nodes_info != SIXTRL_NULLPTR )
            {
                device_id = nodes_info[ 0 ].id;
            }
        }
    }

    if( argc >= 3 )
    {
        int const temp = atoi( argv[ 2 ] );

        if( temp > 0 ) NUM_PARTICLES = temp;
    }

    if( argc >= 4 )
    {
        int const temp = atoi( argv[ 3 ] );

        if( temp > 0 ) NUM_TURNS = temp;
    }

    if( ( st_ComputeNodeId_is_valid( &device_id ) ) &&
        ( NUM_TURNS > 0 ) && ( NUM_PARTICLES > 0 ) )
    {
        char id_str[ 16 ];

        st_ComputeNodeId_to_string( &device_id, &id_str[ 0 ], 16 );

        st_ComputeNodeInfo const* node_info =
            st_CLContextBase_get_available_node_info_by_node_id(
                context, &device_id );

        printf( "Selected [ID]            = %s (%s/%s)\r\n"
                "         [NUM_PARTICLES] = %d\r\n"
                "         [NUM_TURNS]     = %d\r\n"
                "\r\n", id_str, st_ComputeNodeInfo_get_name( node_info ),
                st_ComputeNodeInfo_get_platform( node_info ),
                NUM_PARTICLES, NUM_TURNS );

    }
    else
    {
        return 0;
    }

    st_CLContextBase_delete( context );

    return 0;
}

/* end: examples/c99/track_lhc_no_bb_opencl.c */
