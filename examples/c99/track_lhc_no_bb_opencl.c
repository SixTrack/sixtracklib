#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.h"

int main( int argc, char* argv[] )
{
    NS(ClContext)* ctx                       = SIXTRL_NULLPTR;

    NS(Buffer)* particle_dump                = SIXTRL_NULLPTR;
    NS(Buffer)* lhc_beam_elements_buffer     = SIXTRL_NULLPTR;
    NS(Buffer)* pb                           = SIXTRL_NULLPTR;

    NS(buffer_size_t) NUM_PARTICLES          = 20000;
    NS(buffer_size_t) NUM_TURNS              = 20;

    NS(Particles)*       particles           = SIXTRL_NULLPTR;
    NS(Particles) const* input_particles     = SIXTRL_NULLPTR;
    NS(buffer_size_t)    num_input_particles = 0;

    NS(buffer_size_t) ii                     = 0u;

    int tracking_kernel_id                  = -1;
    double tracking_time                    = 0.0;

    /* --------------------------------------------------------------------- */
    /* Handle command line arguments: */
    /* --------------------------------------------------------------------- */

    if( argc < 2  )
    {
        unsigned int num_devices = 0u;

        printf( "Usage: %s [ID] [NUM_PARTICLES] [NUM_TURNS]\r\n", argv[ 0 ] );

        ctx     = NS(ClContext_create)();
        num_devices = NS(ClContextBase_get_num_available_nodes)( ctx );

        NS(ClContextBase_print_nodes_info)( ctx );

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
        ctx = NS(ClContext_new)( argv[ 1 ] );

        if( ctx == SIXTRL_NULLPTR )
        {
            printf( "Warning         : Provided ID %s not found "
                    "-> use default device instead\r\n",
                    argv[ 1 ] );
        }
    }

    if( !NS(ClContextBase_has_selected_node)( ctx ) )
    {
        /* select default node */
        NS(context_node_id_t) const default_node_id =
            NS(ClContextBase_get_default_node_id)( ctx );

        NS(ClContextBase_select_node_by_node_id)( ctx, &default_node_id );
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

    if( ( ctx != SIXTRL_NULLPTR ) &&
        ( NS(ClContextBase_has_selected_node)( ctx ) ) &&
        ( NUM_TURNS > 0 ) && ( NUM_PARTICLES > 0 ) )
    {
        NS(context_node_id_t) const* node_id =
            NS(ClContextBase_get_selected_node_id)( ctx );

        NS(context_node_info_t) const* node_info =
            NS(ClContextBase_get_selected_node_info)( ctx );

        char id_str[ 16 ];
        NS(ComputeNodeId_to_string)( node_id, &id_str[ 0 ], 16  );

        printf( "\r\n"
                "Selected [ID]            = %s (%s/%s)\r\n"
                "         [NUM_PARTICLES] = %d\r\n"
                "         [NUM_TURNS]     = %d\r\n"
                "\r\n", id_str, NS(ComputeNodeInfo_get_name)( node_info ),
                NS(ComputeNodeInfo_get_platform)( node_info ),
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

    particle_dump = NS(Buffer_new_from_file)(
        NS(PATH_TO_LHC_NO_BB_PARTICLES_DUMP) );

    lhc_beam_elements_buffer = NS(Buffer_new_from_file)(
        NS(PATH_TO_LHC_NO_BB_BEAM_ELEMENTS) );

    pb = NS(Buffer_new)( ( NS(buffer_size_t) )( 1u << 24u ) );

    particles = NS(Particles_new)( pb, NUM_PARTICLES );
    input_particles = NS(Particles_buffer_get_const_particles)(
        particle_dump, 0u );

    num_input_particles = NS(Particles_get_num_of_particles)( input_particles );

    for( ii = 0 ; ii < NUM_PARTICLES ; ++ii )
    {
        NS(buffer_size_t) const jj = ii % num_input_particles;
        NS(Particles_copy_single)( particles, ii, input_particles, jj );
    }

    NS(ClArgument)* particles_arg =
        NS(ClArgument_new_from_buffer)( pb, ctx );

    NS(ClArgument)* beam_elements_arg =
        NS(ClArgument_new_from_buffer)( lhc_beam_elements_buffer, ctx );

    tracking_kernel_id = NS(ClContext_track_until_kernel_id)( ctx );

    /* --------------------------------------------------------------------- */
    /* Perform tracking over NUM_TURNS */
    /* --------------------------------------------------------------------- */

    NS(ClContext_assign_particles_arg)( ctx, particles_arg );
    NS(ClContext_assign_particle_set_arg)( ctx, 0u, NUM_PARTICLES );
    NS(ClContext_assign_beam_elements_arg)( ctx, beam_elements_arg );

    NS(ClContext_track_until)( ctx, NUM_TURNS );

    tracking_time = NS(ClContextBase_get_last_exec_time)(
        ctx, tracking_kernel_id );

    printf( "Tracking time : %10.6f \r\n"
            "              : %10.6f / turn \r\n"
            "              : %10.6f / turn / particle \r\n\r\n",
            tracking_time, tracking_time / NUM_TURNS,
            tracking_time / ( NUM_TURNS * NUM_PARTICLES ) );

    /* --------------------------------------------------------------------- */
    /* Clean-up */
    /* --------------------------------------------------------------------- */

    NS(ClContextBase_delete)( ctx );
    NS(ClArgument_delete)( particles_arg );
    NS(ClArgument_delete)( beam_elements_arg );

    NS(Buffer_delete)( particle_dump );
    NS(Buffer_delete)( lhc_beam_elements_buffer );
    NS(Buffer_delete)( pb );

    return 0;
}

/* end: examples/c99/track_lhc_no_bb_opencl.c */
