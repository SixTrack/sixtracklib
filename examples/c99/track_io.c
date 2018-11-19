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
    st_Buffer* io_buffer      = SIXTRL_NULLPTR;

    st_Particles*   particles = SIXTRL_NULLPTR;

    char* path_output_particles = SIXTRL_NULLPTR;

    int NUM_PARTICLES                  = 0;
    int NUM_TURNS                      = 1;
    int NUM_TURNS_IO_ELEM_BY_ELEM      = 0;
    int NUM_TURNS_IO_TURN_BY_TURN      = 0;
    int NUM_IO_SKIP                    = 1;

    /* -------------------------------------------------------------------- */
    /* Read command line parameters */

    if( argc < 3 )
    {
        printf( "Usage: %s PATH_TO_PARTICLES PATH_TO_BEAM_ELEMENTS "
                "[NTURNS] [NTURNS_IO_ELEM_BY_ELEM] [NTURNS_IO_EVERY_TURN] "
                "[IO_SKIP] [NPARTICLES] "
                "[PATH_TO_OUTPUT_PARTICLES]\r\n",
                argv[ 0 ] );

        printf( "\r\n"
                "PATH_TO_PARTICLES ...... Path to input praticle dump file \r\n\r\n"
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

    if( ( argc >= 3 ) &&
        ( argv[ 1 ] != SIXTRL_NULLPTR ) && ( strlen( argv[ 1 ] ) > 0u ) &&
        ( argv[ 2 ] != SIXTRL_NULLPTR ) && ( strlen( argv[ 2 ] ) > 0u ) )
    {
        input_pb = st_Buffer_new_from_file( argv[ 1 ] );
        eb = st_Buffer_new_from_file( argv[ 2 ] );

        SIXTRL_ASSERT( input_pb != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( st_Buffer_is_particles_buffer( input_pb ) );
        SIXTRL_ASSERT( st_Particles_buffer_get_num_of_particle_blocks(
            input_pb ) > 0u );

        NUM_PARTICLES = st_Particles_get_num_of_particles(
            st_Particles_buffer_get_const_particles( input_pb, 0u ) );

        SIXTRL_ASSERT( NUM_PARTICLES > 0 );
    }

    if( ( argc >= 4 ) &&
        ( argv[ 3 ] != SIXTRL_NULLPTR ) && ( strlen( argv[ 3 ] ) > 0u ) )
    {
        int const temp = atoi( argv[ 3 ] );
        if( temp > 0 ) NUM_TURNS = temp;
    }

    if( ( argc >= 5 ) &&
        ( argv[ 4 ] != SIXTRL_NULLPTR ) && ( strlen( argv[ 4 ] ) > 0u ) )
    {
        int const temp = atoi( argv[ 4 ] );

        if( ( temp >= 0 ) && ( temp <= NUM_TURNS ) )
        {
            NUM_TURNS_IO_ELEM_BY_ELEM = temp;
        }
    }

    if( ( argc >= 6 ) &&
        ( argv[ 5 ] != SIXTRL_NULLPTR ) && ( strlen( argv[ 5 ] ) > 0u ) )
    {
        int const temp = atoi( argv[ 5 ] );

        if( ( temp > 0 ) && ( NUM_TURNS >= ( NUM_TURNS_IO_ELEM_BY_ELEM + temp ) ) )
        {
            NUM_TURNS_IO_TURN_BY_TURN = temp;
        }
    }

    if( ( argc >= 7 ) &&
        ( argv[ 6 ] != SIXTRL_NULLPTR ) && ( strlen( argv[ 6 ] ) > 0u ) )
    {
        int const temp = atoi( argv[ 6 ] );

        if( ( temp > 0 )  && ( NUM_TURNS >
                ( NUM_TURNS_IO_ELEM_BY_ELEM + NUM_TURNS_IO_TURN_BY_TURN ) ) )
        {
            NUM_IO_SKIP = temp;
        }
    }

    if( ( argc >= 8 ) && ( argv[ 7 ] != SIXTRL_NULLPTR ) )
    {
        size_t const output_path_len = strlen( argv[ 7 ] );

        if( output_path_len > 0u )
        {
            path_output_particles = ( char* )malloc(
                sizeof( char ) * ( output_path_len + 1u ) );

            if( path_output_particles != SIXTRL_NULLPTR )
            {
                memset(  path_output_particles, ( int )'\0',
                         output_path_len );

                strncpy( path_output_particles, argv[ 7 ],
                         output_path_len );
            }
        }
    }

    if( ( argc >= 9 ) &&
        ( argv[ 8 ] != SIXTRL_NULLPTR ) && ( strlen( argv[ 8 ] ) > 0u ) )
    {
        int const temp = atoi( argv[ 8 ] );
        if( temp >= 0 ) NUM_PARTICLES = temp;
    }

    /* --------------------------------------------------------------------- */
    /* Prepare input and tracking data from run-time parameters: */


    printf("%-30s = %10d\n","NUM_PARTICLES",NUM_PARTICLES);
    printf("%-30s = %10d\n","NUM_TURNS",NUM_TURNS);
    printf("%-30s = %10d\n","NUM_TURNS_IO_ELEM_BY_ELEM",NUM_TURNS_IO_ELEM_BY_ELEM);
    printf("%-30s = %10d\n","NUM_TURNS_IO_TURN_BY_TURN",NUM_TURNS_IO_TURN_BY_TURN);
    printf("%-30s = %10d\n","NUM_IO_SKIP",NUM_IO_SKIP);

    if( ( NUM_PARTICLES >= 0 ) && ( input_pb != SIXTRL_NULLPTR ) )
    {
        st_Particles const* in_particles =
            st_Particles_buffer_get_const_particles( input_pb, 0u );

        int const in_num_particles =
            st_Particles_get_num_of_particles( in_particles );

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

            int ii = 0;

            SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
            SIXTRL_ASSERT( ( int )st_Particles_get_num_of_particles(
                particles ) == NUM_PARTICLES );

            for( ; ii < NUM_PARTICLES ; ++ii )
            {
                st_Particles_set_particle_id_value( particles, ii, ii );
            }
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

                st_Particles_set_particle_id_value( particles, ii, ii );
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

        io_buffer = st_Buffer_new( 0u );

        st_BeamMonitor_prepare_io_buffer( eb, io_buffer,
            NUM_PARTICLES, NUM_TURNS_IO_ELEM_BY_ELEM );

        st_BeamMonitor_assign_io_buffer( eb, io_buffer,
            NUM_PARTICLES, NUM_TURNS_IO_ELEM_BY_ELEM );
    }

    /* ********************************************************************* */
    /* ****            PERFORM TRACKING AND IO OPERATIONS            ******* */
    /* ********************************************************************* */


    if( ( particles != SIXTRL_NULLPTR ) &&
        ( NUM_PARTICLES > 0 ) && ( NUM_TURNS > 0 ) )
    {
        int ii = 0;
        int const NUM_BEAM_ELEMENTS = st_Buffer_get_num_of_objects( eb );

        for( ; ii < NUM_TURNS_IO_ELEM_BY_ELEM ; ++ii )
        {
            st_Track_all_particles_element_by_element(
                particles, 0u, eb, io_buffer, ii * NUM_BEAM_ELEMENTS );

            st_Track_all_particles_increment_at_turn(
                particles, 0u );
        }

        st_Track_all_particles_until_turn( particles, eb, NUM_TURNS );

        st_Particles_add_copy( io_buffer, particles );

        st_Buffer_write_to_file( track_pb, path_output_particles );
    }

    /* ********************************************************************* */
    /* ****            SEQUENTIALLY PRINT ALL PARTICLES              ******* */
    /* ********************************************************************* */

    if( st_Buffer_get_num_of_objects( io_buffer ) ==
        ( ( NUM_TURNS_IO_ELEM_BY_ELEM * st_Buffer_get_num_of_objects( eb ) ) +
          ( NUM_TURNS_IO_TURN_BY_TURN ) +
          ( ( NUM_TURNS - (
            NUM_TURNS_IO_ELEM_BY_ELEM + NUM_TURNS_IO_TURN_BY_TURN ) )
            / NUM_IO_SKIP ) + 1 ) )
    {
        int ii = 0;
        printf( "Sequentially print out particles stored in io buffer: \r\n" );

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
                    st_Particles const* io_particles =
                        st_Particles_buffer_get_const_particles( io_buffer, ii );

                    printf( "io particles | at turn = %6d | "
                            "beam_element_id = %6d | "
                            "object_type_id = %2d ::\n",
                            jj, kk, ( int )st_Object_get_type_id( eb_it ) );

                    st_Particles_print_out( io_particles );
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
                st_Particles const* io_particles =
                    st_Particles_buffer_get_const_particles( io_buffer, ii );

                printf( "io particles | at turn = %6d ::\n",
                        jj + NUM_TURNS_IO_ELEM_BY_ELEM );

                st_Particles_print_out( io_particles );
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
                st_Particles const* io_particles =
                    st_Particles_buffer_get_const_particles( io_buffer, ii );

                printf( "io particles | at turn = %6d ::\n", jj );

                st_Particles_print_out( io_particles );
                printf( "\r\n" );
            }
        }

         printf( "Print final particle state after tracking %6d turns: \r\n",
                 NUM_TURNS );

         st_Particles_print_out( st_Particles_buffer_get_const_particles(
             io_buffer, ii ) );
    }

    /* ********************************************************************* */
    /* ********                       CLEANUP                        ******* */
    /* ********************************************************************* */

    st_Buffer_delete( eb );
    st_Buffer_delete( track_pb );
    st_Buffer_delete( io_buffer );

    free( path_output_particles );

    return 0;
}

/* end: examples/c99/track_io.c */
