#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.h"

int main( int argc, char* argv[] )
{
    st_Buffer* input_pb       = SIXTRL_NULLPTR;
    st_Buffer* track_pb       = SIXTRL_NULLPTR;
    st_Buffer* eb             = SIXTRL_NULLPTR;
    st_Buffer* out_buffer     = SIXTRL_NULLPTR;

    st_Particles*   particles = SIXTRL_NULLPTR;

    st_ClArgument* particle_buffer_arg = SIXTRL_NULLPTR;
    st_ClArgument* beam_elements_arg   = SIXTRL_NULLPTR;
    st_ClArgument* out_buffer_arg       = SIXTRL_NULLPTR;

    char* path_output_particles = SIXTRL_NULLPTR;

    int NUM_PARTICLES                  = 0;
    int NUM_TURNS                      = 1;
    int NUM_TURNS_IO_ELEM_BY_ELEM      = 0;
    int NUM_TURNS_IO_TURN_BY_TURN      = 0;
    int NUM_IO_SKIP                    = 1;

    st_ClContext* context = st_ClContext_create();

    unsigned int num_devices = 0u;
    num_devices = st_ClContextBase_get_num_available_nodes( context );

    /* -------------------------------------------------------------------- */
    /* Read command line parameters */

    if( argc < 4 )
    {
        printf( "Usage: %s DEVICE_ID_STR PATH_TO_PARTICLES PATH_TO_BEAM_ELEMENTS "
                "[NTURNS] [NTURNS_IO_ELEM_BY_ELEM] [NTURNS_IO_EVERY_TURN] "
                "[IO_SKIP] [NPARTICLES] "
                "[PATH_TO_OUTPUT_PARTICLES]\r\n",
                argv[ 0 ] );

        printf( "\r\n"
                "DEVICE_ID_STR  ......... Device ID string of the OpenCL node\r\n"
                "                         Default: 0.0\r\n" );

        st_ClContextBase_print_nodes_info( context );

        if( num_devices == 0u )
        {
            st_ClContext_delete( context );

            printf( "Qutting program\r\n" );
            return 0;
        }


        printf( "PATH_TO_PARTICLES ...... Path to input praticle dump file \r\n\r\n"
                "PATH_TO_BEAM_ELEMENTS .. Path to input machine description \r\n\r\n"
                "NTURNS ................. Total num of turns to track\r\n"
                "                         Default: 1\r\n"
                "NTURNS_IO_ELEM_BY_ELEM.. Num of turns to dump particles after "
                "each beam element\r\n"
                "                         Default: 1\r\n"
                "NTURNS_IO_EVERY_TURN.... Num of turns to dump particles once "
                "per turn\r\n"
                "                         Default: NTURNS - NTURNS_IO_ELEM_BY_ELEM\r\n"
                "IO_SKIP ................ After NTURNS_IO_ELEM_BY_ELEM and "
                "NTURNS_IO_EVERY_TURN, dump particles every IO_SKIP turn\r\n"
                "                         Default: 1\r\n"
                "PATH_TO_OUTPUT_PARTICLES Path to output file for dumping "
                "the final particle state\r\n"
                "                         Default: ./output_particles.bin\r\n"
                "NPARTICLES ............. number of particles to simulate\r\n"
                "                         Default: same as in PATH_TO_PARTICLES\r\n"
                "\r\n" );

        return 0;
    }

    /* These commands control the selection of kernels in
     * the OpenCL context; Default values are
     * - debug / erro checking kernels are disabled
     * - optimized tracking is enabled
     *
     * Please uncomment the following lines if you want to change this
     * behaviour */
    // st_ClContextBase_enable_debug_mode( context );
    // st_ClContext_disable_optimized_tracking_by_default( context );

    if( ( argc >= 4 ) &&
        ( argv[ 2 ] != SIXTRL_NULLPTR ) && ( strlen( argv[ 2 ] ) > 0u ) &&
        ( argv[ 3 ] != SIXTRL_NULLPTR ) && ( strlen( argv[ 3 ] ) > 0u ) )
    {
        st_Object const* eb_it  = SIXTRL_NULLPTR;
        st_Object const* eb_end = SIXTRL_NULLPTR;

        int enable_beam_beam = 0;

        input_pb = st_Buffer_new_from_file( argv[ 2 ] );
        eb = st_Buffer_new_from_file( argv[ 3 ] );

        /* Scan over the machine description and enable
         * beam-beam tracking if either a 4D or a 6D
         * BeamBeam element has been found */

        eb_it  = st_Buffer_get_const_objects_begin( eb );
        eb_end = st_Buffer_get_const_objects_end( eb );

        for( ; eb_it != eb_end ; ++eb_it )
        {
            st_object_type_id_t const type_id = st_Object_get_type_id( eb_it );

            if( ( type_id == st_OBJECT_TYPE_BEAM_BEAM_4D ) ||
                ( type_id == st_OBJECT_TYPE_BEAM_BEAM_6D ) )
            {
                enable_beam_beam = 1;
                break;
            }
        }

        if( enable_beam_beam == 1 )
        {
            st_ClContext_enable_beam_beam_tracking( context );
        }
        else
        {
            st_ClContext_disable_beam_beam_tracking( context );
        }

        SIXTRL_ASSERT( input_pb != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( st_Buffer_is_particles_buffer( input_pb ) );
        SIXTRL_ASSERT( st_Particles_buffer_get_num_of_particle_blocks(
            input_pb ) == 1u );

        NUM_PARTICLES = st_Particles_buffer_get_total_num_of_particles(
            input_pb );

        SIXTRL_ASSERT( NUM_PARTICLES > 0 );
    }

    if( !st_ClContextBase_has_selected_node( context ) )
    {
        st_ClContextBase_select_node( context, argv[ 1 ] );
    }

    if( st_ClContextBase_has_selected_node( context ) )
    {
        st_context_node_id_t const* node_id =
            st_ClContextBase_get_selected_node_id( context );

        st_context_node_info_t const* node_info =
            st_ClContextBase_get_selected_node_info( context );

        char id_str[ 16 ];
        st_ComputeNodeId_to_string( node_id, &id_str[ 0 ], 16  );

        printf( "\r\n"
                "Selected DEVICE_ID_STR   = %s (%s/%s)\r\n"
                "\r\n", id_str, st_ComputeNodeInfo_get_name( node_info ),
                st_ComputeNodeInfo_get_platform( node_info ) );
    }

    if( ( argc >= 5 ) &&
        ( argv[ 4 ] != SIXTRL_NULLPTR ) && ( strlen( argv[ 4 ] ) > 0u ) )
    {
        int const temp = atoi( argv[ 4 ] );
        if( temp > 0 ) NUM_TURNS = temp;
    }

    if( ( argc >= 6 ) &&
        ( argv[ 5 ] != SIXTRL_NULLPTR ) && ( strlen( argv[ 5 ] ) > 0u ) )
    {
        int const temp = atoi( argv[ 5 ] );

        if( ( temp >= 0 ) && ( temp <= NUM_TURNS ) )
        {
            NUM_TURNS_IO_ELEM_BY_ELEM = temp;
        }
    }

    if( ( argc >= 7 ) &&
        ( argv[ 6 ] != SIXTRL_NULLPTR ) && ( strlen( argv[ 6 ] ) > 0u ) )
    {
        int const temp = atoi( argv[ 6 ] );

        if( ( temp > 0 ) && ( NUM_TURNS >= ( NUM_TURNS_IO_ELEM_BY_ELEM + temp ) ) )
        {
            NUM_TURNS_IO_TURN_BY_TURN = temp;
        }
    }

    if( ( argc >= 8 ) &&
        ( argv[ 7 ] != SIXTRL_NULLPTR ) && ( strlen( argv[ 7 ] ) > 0u ) )
    {
        int const temp = atoi( argv[ 7 ] );

        if( ( temp > 0 )  && ( NUM_TURNS >
                ( NUM_TURNS_IO_ELEM_BY_ELEM + NUM_TURNS_IO_TURN_BY_TURN ) ) )
        {
            NUM_IO_SKIP = temp;
        }
    }

    if( ( argc >= 9 ) && ( argv[ 8 ] != SIXTRL_NULLPTR ) )
    {
        size_t const output_path_len = strlen( argv[ 8 ] );

        if( output_path_len > 0u )
        {
            path_output_particles = ( char* )malloc(
                sizeof( char ) * ( output_path_len + 1u ) );

            if( path_output_particles != SIXTRL_NULLPTR )
            {
                memset(  path_output_particles, ( int )'\0',
                         output_path_len );

                strncpy( path_output_particles, argv[ 8 ],
                         output_path_len );
            }
        }
    }

    if( ( argc >= 10 ) &&
        ( argv[ 9 ] != SIXTRL_NULLPTR ) && ( strlen( argv[ 9 ] ) > 0u ) )
    {
        int const temp = atoi( argv[ 9 ] );
        if( temp >= 0 ) NUM_PARTICLES = temp;
    }

    /* --------------------------------------------------------------------- */
    /* Prepare input and tracking data from run-time parameters: */

    if( ( NUM_PARTICLES >= 0 ) && ( input_pb != SIXTRL_NULLPTR ) &&
        ( st_ClContextBase_has_selected_node( context ) ) )
    {
        st_Particles const* in_particles =
            st_Particles_buffer_get_const_particles( input_pb, 0u );

        int const in_num_particles =
            st_Particles_get_num_of_particles( in_particles );

        printf( "NUM_TURNS                 = %6d\r\n"
                "NUM_TURNS_IO_ELEM_BY_ELEM = %6d\r\n"
                "NUM_TURNS_IO_TURN_BY_TURN = %6d\r\n"
                "NUM_IO_SKIP               = %6d\r\n",
                NUM_TURNS, NUM_TURNS_IO_ELEM_BY_ELEM,
                NUM_TURNS_IO_TURN_BY_TURN, NUM_IO_SKIP );

        if( ( NUM_PARTICLES == INT32_MAX ) || ( NUM_PARTICLES == 0 ) )
        {
            NUM_PARTICLES = in_num_particles;
        }

        SIXTRL_ASSERT( in_num_particles > 0 );
        track_pb  = st_Buffer_new( 0u );

        if( NUM_PARTICLES == in_num_particles )
        {
            particles = st_Particles_add_copy(
                track_pb, in_particles );
        }
        else
        {
            int ii = 0;
            particles = st_Particles_new( track_pb, NUM_PARTICLES );

            for( ; ii < NUM_PARTICLES ; ++ii )
            {
                int const jj = ii % in_num_particles;
                st_Particles_copy_single( particles, ii,
                                          in_particles, jj );
            }
        }

        st_Buffer_delete( input_pb );
        input_pb = SIXTRL_NULLPTR;
    }

    if( path_output_particles == SIXTRL_NULLPTR )
    {
        path_output_particles = ( char* )malloc(
            sizeof( char ) * 64u );

        memset(  path_output_particles, ( int )'\0', 64u );
        strncpy( path_output_particles, "./output_particles.bin", 63u );
    }

    if( ( eb != SIXTRL_NULLPTR ) &&
        ( st_Buffer_get_num_of_objects( eb ) > 0 ) )
    {
        if( NUM_TURNS_IO_TURN_BY_TURN > 0 )
        {
            st_BeamMonitor* beam_monitor = st_BeamMonitor_new( eb );
            st_BeamMonitor_set_num_stores( beam_monitor, NUM_TURNS_IO_TURN_BY_TURN );
            st_BeamMonitor_set_start( beam_monitor, NUM_TURNS_IO_ELEM_BY_ELEM );
            st_BeamMonitor_set_skip( beam_monitor, 1 );
            st_BeamMonitor_set_is_rolling( beam_monitor, false );
        }

        if( ( NUM_IO_SKIP > 0 ) && ( NUM_TURNS > (
                NUM_TURNS_IO_ELEM_BY_ELEM + NUM_TURNS_IO_TURN_BY_TURN ) ) )
        {
            int const remaining_turns = NUM_TURNS - (
                NUM_TURNS_IO_ELEM_BY_ELEM + NUM_TURNS_IO_TURN_BY_TURN );

            int const num_stores = remaining_turns / NUM_IO_SKIP;

            st_BeamMonitor* beam_monitor = st_BeamMonitor_new( eb );
            st_BeamMonitor_set_num_stores( beam_monitor, num_stores );
            st_BeamMonitor_set_start( beam_monitor,
                    NUM_TURNS_IO_ELEM_BY_ELEM + NUM_TURNS_IO_TURN_BY_TURN );
            st_BeamMonitor_set_skip( beam_monitor, NUM_IO_SKIP );
            st_BeamMonitor_set_is_rolling( beam_monitor, true );
        }

        out_buffer = st_Buffer_new( 0u );

        st_BeamMonitor_prepare_particles_out_buffer( eb, out_buffer,
            particles, NUM_TURNS_IO_ELEM_BY_ELEM );

        particle_buffer_arg = st_ClArgument_new_from_buffer( track_pb, context );
        beam_elements_arg   = st_ClArgument_new_from_buffer( eb, context );
        out_buffer_arg       = st_ClArgument_new_from_buffer( out_buffer, context );
    }

    /* ********************************************************************* */
    /* ****            PERFORM TRACKING AND IO OPERATIONS            ******* */
    /* ********************************************************************* */

    if( ( particles != SIXTRL_NULLPTR ) &&
        ( NUM_PARTICLES > 0 ) && ( NUM_TURNS > 0 ) &&
        ( particle_buffer_arg != SIXTRL_NULLPTR ) &&
        ( beam_elements_arg   != SIXTRL_NULLPTR ) &&
        ( out_buffer_arg       != SIXTRL_NULLPTR ) )
    {
        int ii = 0;
        int const NUM_BEAM_ELEMENTS = st_Buffer_get_num_of_objects( eb );

        st_ClContext_assign_beam_monitor_out_buffer( context,
            beam_elements_arg, out_buffer_arg, NUM_TURNS_IO_ELEM_BY_ELEM * NUM_BEAM_ELEMENTS );


        for( ; ii < NUM_TURNS_IO_ELEM_BY_ELEM ; ++ii )
        {
            st_ClContext_track_element_by_element(
                context, particle_buffer_arg, beam_elements_arg,
                out_buffer_arg, ii * NUM_BEAM_ELEMENTS );
        }

        st_ClContext_track( context, particle_buffer_arg,
            beam_elements_arg, NUM_TURNS );

        st_ClArgument_read( particle_buffer_arg, track_pb );
        st_ClArgument_read( out_buffer_arg, out_buffer );
        st_Particles_add_copy( out_buffer,
            st_Particles_buffer_get_const_particles( track_pb, 0u ) );

        st_Buffer_write_to_file( track_pb, path_output_particles );
    }

    /* ********************************************************************* */
    /* ****            SEQUENTIALLY PRINT ALL PARTICLES              ******* */
    /* ********************************************************************* */

    if( st_Buffer_get_num_of_objects( out_buffer ) ==
        ( ( NUM_TURNS_IO_ELEM_BY_ELEM * st_Buffer_get_num_of_objects( eb ) ) +
          ( NUM_TURNS_IO_TURN_BY_TURN ) +
          ( ( NUM_TURNS - (
            NUM_TURNS_IO_ELEM_BY_ELEM + NUM_TURNS_IO_TURN_BY_TURN ) )
            / NUM_IO_SKIP ) + 1 ) )
    {
        int ii = 0;
        printf( "Sequentially print out particles stored in out buffer: \r\n" );

        if( NUM_TURNS_IO_ELEM_BY_ELEM > 0 )
        {
            int jj = 0;
            int const NUM_BEAM_ELEMENTS = st_Buffer_get_num_of_objects( eb );
            st_Object const* eb_begin = st_Buffer_get_const_objects_begin( eb );
            st_Object const* eb_end   = st_Buffer_get_const_objects_end( eb );

            printf( "----------------------------------------------------------"
                    "----------------------------------------------------\r\n" );
            printf( " - %3d turns element - by element (%6d elements per line)\r\n",
                    NUM_TURNS_IO_ELEM_BY_ELEM, NUM_BEAM_ELEMENTS );

            for( ; jj < NUM_TURNS_IO_ELEM_BY_ELEM ; ++jj )
            {
                int kk = 0;
                st_Object const* eb_it = eb_begin;

                for( ; eb_it != eb_end ; ++eb_it, ++ii, ++kk )
                {
                    st_Particles const* out_particles =
                        st_Particles_buffer_get_const_particles( out_buffer, ii );

                    printf( "out particles | at turn = %6d | "
                            "beam_element_id = %6d | "
                            "object_type_id = %2d ::\n",
                            jj, kk, ( int )st_Object_get_type_id( eb_it ) );

                    st_Particles_print_out( out_particles );
                    printf( "\r\n" );
                }
            }
        }

        if( NUM_TURNS_IO_TURN_BY_TURN > 0 )
        {
            printf( "----------------------------------------------------------"
                    "----------------------------------------------------\r\n" );
            printf( " - %3d turns every turn \r\n", NUM_TURNS_IO_TURN_BY_TURN );

            int jj = 0;
            for( ; jj < NUM_TURNS_IO_TURN_BY_TURN ; ++jj, ++ii )
            {
                st_Particles const* out_particles =
                    st_Particles_buffer_get_const_particles( out_buffer, ii );

                printf( "out particles | at turn = %6d ::\n",
                        jj + NUM_TURNS_IO_ELEM_BY_ELEM );

                st_Particles_print_out( out_particles );
                printf( "\r\n" );
            }
        }

        if( NUM_TURNS > ( NUM_TURNS_IO_ELEM_BY_ELEM + NUM_TURNS_IO_TURN_BY_TURN ) )
        {
            int const remaining_turns = NUM_TURNS -
                ( NUM_TURNS_IO_ELEM_BY_ELEM + NUM_TURNS_IO_TURN_BY_TURN );

            printf( "----------------------------------------------------------"
                    "----------------------------------------------------\r\n" );
            printf( " - %3d remaining turns, dump every %3d th turn \r\n",
                    remaining_turns, NUM_IO_SKIP );

            int jj = NUM_TURNS_IO_ELEM_BY_ELEM + NUM_TURNS_IO_TURN_BY_TURN;
            for( ; jj < NUM_TURNS ; jj += NUM_IO_SKIP, ++ii )
            {
                st_Particles const* out_particles =
                    st_Particles_buffer_get_const_particles( out_buffer, ii );

                printf( "out particles | at turn = %6d ::\n", jj );

                st_Particles_print_out( out_particles );
                printf( "\r\n" );
            }
        }

         printf( "Print final particle state after tracking %6d turns: \r\n",
                 NUM_TURNS );

         st_Particles_print_out( st_Particles_buffer_get_const_particles(
             out_buffer, ii ) );
    }

    /* ********************************************************************* */
    /* ********                       CLEANUP                        ******* */
    /* ********************************************************************* */

    st_ClArgument_delete( particle_buffer_arg );
    st_ClArgument_delete( beam_elements_arg );
    st_ClArgument_delete( out_buffer_arg );

    st_Buffer_delete( eb );
    st_Buffer_delete( track_pb );
    st_Buffer_delete( out_buffer );

    free( path_output_particles );

    return 0;
}

/* end: examples/c99/track_io.c */
