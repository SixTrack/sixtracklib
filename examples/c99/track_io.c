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

    st_TrackJobCpu* job       = SIXTRL_NULLPTR;
    st_Particles*   particles = SIXTRL_NULLPTR;

    char* path_output_particles = SIXTRL_NULLPTR;

    int NUM_PARTICLES                  = 0;
    int NUM_TURNS                      = 1;
    int DUMP_ELEM_BY_ELEM_TURNS      = 0;
    int DUMP_TURN_BY_TURN_TURNS      = 0;
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
            DUMP_ELEM_BY_ELEM_TURNS = temp;
        }
    }

    if( ( argc >= 6 ) &&
        ( argv[ 5 ] != SIXTRL_NULLPTR ) && ( strlen( argv[ 5 ] ) > 0u ) )
    {
        int const temp = atoi( argv[ 5 ] );

        if( ( temp > 0 ) && ( NUM_TURNS >= ( DUMP_ELEM_BY_ELEM_TURNS + temp ) ) )
        {
            DUMP_TURN_BY_TURN_TURNS = temp;
        }
    }

    if( ( argc >= 7 ) &&
        ( argv[ 6 ] != SIXTRL_NULLPTR ) && ( strlen( argv[ 6 ] ) > 0u ) )
    {
        int const temp = atoi( argv[ 6 ] );

        if( ( temp > 0 )  && ( NUM_TURNS >
                ( DUMP_ELEM_BY_ELEM_TURNS + DUMP_TURN_BY_TURN_TURNS ) ) )
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

                strcpy( path_output_particles, argv[ 7 ] );
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
    printf("%-30s = %10d\n","DUMP_ELEM_BY_ELEM_TURNS",DUMP_ELEM_BY_ELEM_TURNS);
    printf("%-30s = %10d\n","DUMP_TURN_BY_TURN_TURNS",DUMP_TURN_BY_TURN_TURNS);
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

            SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
            SIXTRL_ASSERT( ( int )st_Particles_get_num_of_particles(
                particles ) == NUM_PARTICLES );
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
        if( DUMP_TURN_BY_TURN_TURNS > 0 )
        {
            st_BeamMonitor* beam_monitor = st_BeamMonitor_new( eb );
            st_BeamMonitor_set_num_stores( beam_monitor, DUMP_TURN_BY_TURN_TURNS );
            st_BeamMonitor_set_start( beam_monitor, DUMP_ELEM_BY_ELEM_TURNS );
            st_BeamMonitor_set_skip( beam_monitor, 1 );
            st_BeamMonitor_set_is_rolling( beam_monitor, false );
        }

        if( ( NUM_IO_SKIP > 0 ) && ( NUM_TURNS > (
                DUMP_ELEM_BY_ELEM_TURNS + DUMP_TURN_BY_TURN_TURNS ) ) )
        {
            int const remaining_turns = NUM_TURNS - (
                DUMP_ELEM_BY_ELEM_TURNS + DUMP_TURN_BY_TURN_TURNS );

            int const num_stores = remaining_turns / NUM_IO_SKIP;

            st_BeamMonitor* beam_monitor = st_BeamMonitor_new( eb );
            st_BeamMonitor_set_num_stores( beam_monitor, num_stores );
            st_BeamMonitor_set_start( beam_monitor,
                    DUMP_ELEM_BY_ELEM_TURNS + DUMP_TURN_BY_TURN_TURNS );
            st_BeamMonitor_set_skip( beam_monitor, NUM_IO_SKIP );
            st_BeamMonitor_set_is_rolling( beam_monitor, true );
        }

        if( DUMP_ELEM_BY_ELEM_TURNS > 0u )
        {
            job = st_TrackJobCpu_new_with_output(
                track_pb, eb, SIXTRL_NULLPTR, DUMP_ELEM_BY_ELEM_TURNS );
        }
        else
        {
            job = st_TrackJobCpu_new( track_pb, eb );
        }


    }

    /* ********************************************************************* */
    /* ****            PERFORM TRACKING AND IO OPERATIONS            ******* */
    /* ********************************************************************* */

    if( DUMP_ELEM_BY_ELEM_TURNS > 0u )
    {
        st_TrackJobCpu_track_elem_by_elem( job, DUMP_ELEM_BY_ELEM_TURNS );
    }

    if( DUMP_ELEM_BY_ELEM_TURNS > NUM_TURNS )
    {
        double const denom_turns =
            ( double )( NUM_TURNS - DUMP_ELEM_BY_ELEM_TURNS );

        double const denom_turns_particles =
            denom_turns * NUM_PARTICLES;

        double tracking_time     = ( double )0.0;
        double end_tracking_time = ( double )0.0;

        double const start_tracking_time = st_Time_get_seconds_since_epoch();

        st_TrackJobCpu_track_until_turn( job, NUM_TURNS );

        end_tracking_time = st_Time_get_seconds_since_epoch();

        tracking_time = ( start_tracking_time < end_tracking_time )
            ? end_tracking_time - start_tracking_time : ( double )0.0;

        printf( "time / turn / particle : %.3e\r\n"
                "time / turn            : %.3e\r\n"
                "time total             : %.3e\r\n",
                tracking_time / denom_turns_particles,
                tracking_time / denom_turns, tracking_time );

        if( path_output_particles != SIXTRL_NULLPTR )
        {
            st_Buffer* out_buffer      = SIXTRL_NULLPTR;
            st_Buffer const* result_pb = SIXTRL_NULLPTR;

            st_TrackJobCpu_collect( job );

            out_buffer = st_TrackJob_get_output_buffer( job );
            result_pb  = st_TrackJob_get_const_particles_buffer( job );

            st_Particles_add_copy( out_buffer,
                st_Particles_buffer_get_const_particles( result_pb, 0u ) );

            st_Buffer_write_to_file( out_buffer, path_output_particles );
        }
    }

    /* ********************************************************************* */
    /* ********                       CLEANUP                        ******* */
    /* ********************************************************************* */

    st_Buffer_delete( eb );
    st_Buffer_delete( track_pb );
    st_TrackJobCpu_delete( job );

    free( path_output_particles );

    return 0;
}

/* end: examples/c99/track_io.c */
