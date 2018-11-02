#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.h"

int main( int argc, char* argv[] )
{
    st_ClContext* context                   = SIXTRL_NULLPTR;

    st_Buffer* particle_dump                = SIXTRL_NULLPTR;
    st_Buffer* beam_elements_buffer         = SIXTRL_NULLPTR;
    st_Buffer* pb                           = SIXTRL_NULLPTR;

    st_buffer_size_t NUM_PARTICLES          = 20000;
    st_buffer_size_t NUM_TURNS              = 20;

    st_Particles*       particles           = SIXTRL_NULLPTR;
    st_Particles const* input_particles     = SIXTRL_NULLPTR;
    st_buffer_size_t    num_input_particles = 0;

    st_buffer_size_t ii                     = 0u;

    int tracking_kernel_id                  = -1;
    double tracking_time                    = 0.0;

    /* --------------------------------------------------------------------- */
    /* Handle command line arguments: */
    /* --------------------------------------------------------------------- */

    if( argc < 2  )
    {
        unsigned int num_devices = 0u;

        printf( "Usage: %s [ID] [NUM_PARTICLES] [NUM_TURNS]\r\n", argv[ 0 ] );

        context     = st_ClContext_create();
        num_devices = st_ClContextBase_get_num_available_nodes( context );

        st_ClContextBase_print_nodes_info( context );

        if( num_devices == 0u )
        {
            printf( "Quitting program!\r\n" );
            return 0;
        }

        printf( "\r\n"
                "[NUM_PARTICLES] :: Number of particles for the simulation\r\n"
                "                :: Default = %d\r\n", ( int )NUM_PARTICLES );

        printf( "\r\n"
                "[NUM_TURNS]     :: Number of turns for the simulation\r\n"
                "                :: Default = %d\r\n", ( int )NUM_TURNS );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /* Select the node based on the first command line param or
     * select the default node: */

    if( argc >= 2 )
    {
        context = st_ClContext_new( argv[ 1 ] );

        if( context == SIXTRL_NULLPTR )
        {
            printf( "Warning         : Provided ID %s not found "
                    "-> use default device instead\r\n",
                    argv[ 1 ] );
        }
    }

    if( !st_ClContextBase_has_selected_node( context ) )
    {
        /* select default node */
        st_context_node_id_t const default_node_id =
            st_ClContextBase_get_default_node_id( context );

        st_ClContextBase_select_node_by_node_id( context, &default_node_id );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /* take the number of particles from the third command line parameter,
     * otherwise keep using the default value */
    if( argc >= 3 )
    {
        int const temp = atoi( argv[ 2 ] );

        if( temp > 0 ) NUM_PARTICLES = temp;
    }

    /* take the number of turns from the fourth command line parameter,
     * otherwise keep using the default value */
    if( argc >= 4 )
    {
        int const temp = atoi( argv[ 3 ] );

        if( temp > 0 ) NUM_TURNS = temp;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /* Give a summary of the provided parameters */

    if( ( context != SIXTRL_NULLPTR ) &&
        ( st_ClContextBase_has_selected_node( context ) ) &&
        ( NUM_TURNS > 0 ) && ( NUM_PARTICLES > 0 ) )
    {
        st_context_node_id_t const* node_id =
            st_ClContextBase_get_selected_node_id( context );

        st_context_node_info_t const* node_info =
            st_ClContextBase_get_selected_node_info( context );

        char id_str[ 16 ];
        st_ComputeNodeId_to_string( node_id, &id_str[ 0 ], 16  );

        printf( "\r\n"
                "Selected [ID]            = %s (%s/%s)\r\n"
                "         [NUM_PARTICLES] = %d\r\n"
                "         [NUM_TURNS]     = %d\r\n"
                "\r\n", id_str, st_ComputeNodeInfo_get_name( node_info ),
                st_ComputeNodeInfo_get_platform( node_info ),
                ( int )NUM_PARTICLES, ( int )NUM_TURNS );

    }
    else
    {
        /* If we get here, something went wrong, most likely with the
         * selection of the device -> bailing out */

        return 0;
    }

    /* ---------------------------------------------------------------------- */
    /* Prepare the buffers: */
    /* ---------------------------------------------------------------------- */

    particle_dump = st_Buffer_new_from_file(
        st_PATH_TO_BEAMBEAM_PARTICLES_DUMP );

    beam_elements_buffer = st_Buffer_new_from_file(
        st_PATH_TO_BEAMBEAM_BEAM_ELEMENTS );

    pb = st_Buffer_new( ( st_buffer_size_t )( 1u << 24u ) );

    particles = st_Particles_new( pb, NUM_PARTICLES );
    input_particles = st_Particles_buffer_get_const_particles( particle_dump, 0u );
    num_input_particles = st_Particles_get_num_of_particles( input_particles );

    for( ii = 0 ; ii < NUM_PARTICLES ; ++ii )
    {
        st_buffer_size_t const jj = ii % num_input_particles;
        st_Particles_copy_single( particles, ii, input_particles, jj );
    }

    st_ClArgument* particles_arg =
        st_ClArgument_new_from_buffer( pb, context );

    st_ClArgument* beam_elements_arg =
        st_ClArgument_new_from_buffer( beam_elements_buffer, context );

    /* --------------------------------------------------------------------- */
    /* Perform tracking over NUM_TURNS */
    /* --------------------------------------------------------------------- */

    st_ClContext_track_num_turns(
        context, particles_arg, beam_elements_arg, NUM_TURNS );

    tracking_kernel_id = st_ClContext_get_tracking_kernel_id( context );

    tracking_time = st_ClContextBase_get_last_exec_time( context, tracking_kernel_id );

    printf( "Tracking time : %10.6f \r\n"
            "              : %10.6f / turn \r\n"
            "              : %10.6f / turn / particle \r\n\r\n",
            tracking_time, tracking_time / NUM_TURNS,
            tracking_time / ( NUM_TURNS * NUM_PARTICLES ) );

    /* --------------------------------------------------------------------- */
    /* Clean-up */
    /* --------------------------------------------------------------------- */

    st_ClContextBase_delete( context );
    st_ClArgument_delete( particles_arg );
    st_ClArgument_delete( beam_elements_arg );

    st_Buffer_delete( particle_dump );
    st_Buffer_delete( beam_elements_buffer );
    st_Buffer_delete( pb );

    return 0;
}

/* end: examples/c99/track_lhc_no_bb_opencl.c */
