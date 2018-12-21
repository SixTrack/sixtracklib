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
    st_Buffer* pb             = SIXTRL_NULLPTR;
    st_Buffer* eb             = SIXTRL_NULLPTR;
    st_Particles*   particles = SIXTRL_NULLPTR;

    int NUM_PARTICLES                  = 0;
    int NUM_TURNS                      = 10;
    int NUM_TURNS_ELEM_BY_ELEM         = 3;
    int OUTPUT_SKIP                    = 1;

    /* -------------------------------------------------------------------- */
    /* Read command line parameters */

    if( argc < 3 )
    {
        printf( "Usage: %s PATH_TO_PARTICLES PATH_TO_BEAM_ELEMENTS\r\n",
                argv[ 0 ] );

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

    /* --------------------------------------------------------------------- */
    /* Prepare input and tracking data from run-time parameters: */

    printf("%-30s = %10d\n","NUM_PARTICLES",NUM_PARTICLES);
    printf("%-30s = %10d\n","NUM_TURNS",NUM_TURNS);
    printf("%-30s = %10d\n","NUM_TURNS_ELEM_BY_ELEM",NUM_TURNS_ELEM_BY_ELEM);
    printf("%-30s = %10d\n","OUTPUT_SKIP", OUTPUT_SKIP);

    if( ( NUM_PARTICLES >= 0 ) && ( input_pb != SIXTRL_NULLPTR ) )
    {
        st_Particles const* in_particles =
            st_Particles_buffer_get_const_particles( input_pb, 0u );

        pb  = st_Buffer_new( 0u );
        particles = st_Particles_add_copy( pb, in_particles );

        st_Buffer_delete( input_pb );
        input_pb = SIXTRL_NULLPTR;
    }

    if( ( eb != SIXTRL_NULLPTR ) &&
        ( st_Buffer_get_num_of_objects( eb ) > 0 ) &&
        ( NUM_TURNS > NUM_TURNS_ELEM_BY_ELEM ) )
    {
        st_BeamMonitor* beam_monitor = st_BeamMonitor_new( eb );
        st_BeamMonitor_set_num_stores( beam_monitor,
            NUM_TURNS - NUM_TURNS_ELEM_BY_ELEM );

        st_BeamMonitor_set_start( beam_monitor, NUM_TURNS_ELEM_BY_ELEM );
        st_BeamMonitor_set_skip( beam_monitor, OUTPUT_SKIP );
        st_BeamMonitor_set_is_rolling( beam_monitor, false );
    }

    /* ********************************************************************* */
    /* ****            PERFORM TRACKING AND IO OPERATIONS            ******* */
    /* ********************************************************************* */

    if( ( particles != SIXTRL_NULLPTR ) &&
        ( NUM_PARTICLES > 0 ) && ( NUM_TURNS > 0 ) )
    {
        /* Create the track_job AND
         * Track NUM_TURNS_ELEM_BY_ELEM turns element-by-element, then
         * track 0 turns using the beam-monitor */

        st_TrackJobCpu* track_job = st_TrackJobCpu_new(
            pb, eb, 0, NUM_TURNS_ELEM_BY_ELEM );

        /* get the pointer to the output buffer */
        st_Buffer const* output_buffer =
            st_TrackJobCpu_get_output_buffer( track_job );

        /* Track until NUM_TURNS_ELEM_BY_ELEM + 3: */
        int const TRACK_UNTIL_T2 = NUM_TURNS_ELEM_BY_ELEM + 3;
        st_TrackJobCpu_track( track_job ,TRACK_UNTIL_T2 );

        /* if we want to get the intermediate results for the particle_buffer
         * itself, we have to call NS(TrackJobCpu_collec)() */
        st_TrackJobCpu_collect( track_job );

        st_Buffer const* temp_particles_buffer =
            st_TrackJobCpu_get_particle_buffer( track_job );

        st_Particles_buffer_print_out( temp_particles_buffer );

        /* Track until NUM_TURNS: */
        st_TrackJobCpu_track( track_job, NUM_TURNS );

        st_Particles_buffer_print_out( output_buffer );

        st_TrackJobCpu_collect( track_job );

        temp_particles_buffer = st_TrackJobCpu_get_particle_buffer(
            track_job );

        /* Delete/free ressources */
        st_TrackJobCpu_delete( track_job );
    }

    /* ********************************************************************* */
    /* ********                       CLEANUP                        ******* */
    /* ********************************************************************* */

    st_Buffer_delete( eb );
    st_Buffer_delete( pb );

    return 0;
}

/* end: examples/c99/track_job_cpu.c */
