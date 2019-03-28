#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.h"

int main( int argc, char* argv[] )
{
    typedef st_buffer_size_t st_size_t;

    st_Buffer* input_pb     = SIXTRL_NULLPTR;
    st_Buffer* pb           = SIXTRL_NULLPTR;
    st_Buffer* eb           = SIXTRL_NULLPTR;
    st_Particles* particles = SIXTRL_NULLPTR;

    st_TrackJobCpu* job     = SIXTRL_NULLPTR;

    int NUM_PARTICLES                  = 0;
    int NUM_TURNS                      = 10;
    int NUM_TURNS_ELEM_BY_ELEM         = 3;
    int OUTPUT_SKIP                    = 1;

    int track_status                   = 0;

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
        st_BeamMonitor_set_is_rolling( beam_monitor, true );
    }

    /* ********************************************************************* */
    /* ****               PERFORM TRACKING OPERATIONS                ******* */
    /* ********************************************************************* */

    if( ( particles != SIXTRL_NULLPTR ) &&
        ( NUM_PARTICLES > 0 ) && ( NUM_TURNS > 0 ) )
    {
        /* Create the track_job
         * If the number of element by element turns is 0, then we can
         * use the simplier API.
         *
         * NOTE: calling st_TrackJobCpu_new_with_output() for 0 elem by elem
         * turns would work just as well!
         */

        if( NUM_TURNS_ELEM_BY_ELEM <= 0 )
        {
            job = st_TrackJobCpu_new( pb, eb );
        }
        else
        {
            job = st_TrackJobCpu_new_with_output(
                pb, eb, SIXTRL_NULLPTR, NUM_TURNS_ELEM_BY_ELEM );

            track_status |= st_TrackJobCpu_track_elem_by_elem(
                job, NUM_TURNS_ELEM_BY_ELEM );
        }

        if( NUM_TURNS > NUM_TURNS_ELEM_BY_ELEM )
        {
            track_status |= st_TrackJobCpu_track_until_turn( job, NUM_TURNS );
        }

        /* ****************************************************************** */
        /* ****               PERFORM OUTPUT OPERATIONS                ****** */
        /* ****************************************************************** */

        /* NOTE: for the CPU Track Job, collect (currently) performs no
         * operations. Since this *might* change in the future, it's
         * mandated to always call NS(TrackJobCpu_collect)() before
         * accessing the particles, the beam elements or the
         * output buffer */

        st_TrackJobCpu_collect( job );

        if( track_status == 0 )
        {
            st_Buffer* out_buffer = SIXTRL_NULLPTR;

            /* trk_pb and trk_eb should still point to the
             * input buffers pb and eb */

            st_Buffer* trk_pb = st_TrackJob_get_particles_buffer( job );
            st_Buffer* trk_eb = st_TrackJob_get_beam_elements_buffer( job );

            ( void )trk_pb;
            ( void )trk_eb;

            if( st_TrackJob_has_output_buffer( job ) )
            {
                out_buffer = st_TrackJob_get_output_buffer( job );
            }

            if( st_TrackJob_has_elem_by_elem_output( job ) )
            {
                st_ElemByElemConfig const* elem_by_elem_config =
                    st_TrackJob_get_elem_by_elem_config( job );

                st_size_t const out_offset =
                    st_TrackJob_get_elem_by_elem_output_buffer_offset(
                        job );

                st_Particles* elem_by_elem_output =
                    st_Particles_buffer_get_particles( out_buffer, out_offset );

                /* st_Particles_print_out( elem_by_elem_output ); */

                ( void )elem_by_elem_config;
                ( void )elem_by_elem_output;
            }

            if( st_TrackJob_has_beam_monitor_output( job ) )
            {
                st_size_t const bemon_start_index =
                    st_TrackJob_get_beam_monitor_output_buffer_offset(
                        job );

                st_size_t const bemon_stop_index = bemon_start_index +
                    st_TrackJob_get_num_beam_monitors( job );

                st_size_t ii = bemon_start_index;

                for(  ; ii < bemon_stop_index ; ++ii )
                {
                    st_Particles* out_particles =
                        st_Particles_buffer_get_particles( out_buffer, ii );

                    /* st_Particles_print_out( out_particles ); */

                    ( void )out_particles;
                }
            }
        }
    }

    /* ****************************************************************** */
    /* ****                   CLEANUP OPERATIONS                   ****** */
    /* ****************************************************************** */

    st_TrackJobCpu_delete( job );

    st_Buffer_delete( pb );
    st_Buffer_delete( eb );
    st_Buffer_delete( input_pb );

    return 0;
}

/* end: examples/c99/track_job_cpu.c */
