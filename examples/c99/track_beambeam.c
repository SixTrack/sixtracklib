#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.h"

int main( int argc, char* argv[] )
{
    st_Buffer* input_particle_buffer = st_Buffer_new_from_file(
        st_PATH_TO_BEAMBEAM_PARTICLES_DUMP );

    st_Buffer* beam_elements_buffer = st_Buffer_new_from_file(
        st_PATH_TO_BEAMBEAM_BEAM_ELEMENTS );

    st_Buffer* particles_buffer = st_Buffer_new( 0u );

    st_buffer_size_t NUM_PARTICLES          = 2;
    st_buffer_size_t NUM_TURNS              = 2;

    st_Particles*       particles           = SIXTRL_NULLPTR;
    st_Particles const* input_particles     = SIXTRL_NULLPTR;
    st_buffer_size_t    num_input_particles = 0;

    st_buffer_size_t ii = 0;

    double start_tracking_time = 0.0;
    double end_tracking_time   = 0.0;
    double tracking_time       = 0.0;

    /* ********************************************************************** */
    /* ****   Handling of command line parameters                             */
    /* ********************************************************************** */

    if( argc == 1 )
    {
        printf( "Usage: %s [NUM_PARTICLES] [NUM_TURNS]\r\n", argv[ 0 ] );

        printf( "\r\n"
                "NUM_PARTICLES: Number of particles for the simulation\r\n"
                "               Default = %lu\r\n",
                ( unsigned long )NUM_PARTICLES );

        printf( "\r\n"
                "NUM_TURNS    : Number of turns for the simulation\r\n"
                "               Default = %lu\r\n"
                "\r\n",
                ( unsigned long )NUM_TURNS );
    }

    if( argc >= 2 )
    {
        int temp = atoi( argv[ 1 ] );

        if( temp > 0 )
        {
            NUM_PARTICLES = ( st_buffer_size_t )temp;
        }
    }

    if( argc >= 3 )
    {
        int temp = atoi( argv[ 2 ] );

        if( temp > 0 )
        {
            NUM_TURNS = ( uint64_t )temp;
        }
    }

    printf( "Use: NUM_PARTICLES = %10lu\r\n"
            "     NUM_TURNS     = %10lu\r\n\r\n", NUM_PARTICLES, NUM_TURNS );

    /* ********************************************************************** */
    /* ****   Building Particles Data from Input Example Particle Data   **** */
    /* ********************************************************************** */

    particles = st_Particles_new( particles_buffer, NUM_PARTICLES );

    input_particles = st_Particles_buffer_get_const_particles(
        input_particle_buffer, 0u );

    num_input_particles = st_Particles_get_num_of_particles( input_particles );

    for( ii = 0 ; ii < NUM_PARTICLES ; ++ii )
    {
        st_buffer_size_t const jj = ii % num_input_particles;
        st_Particles_copy_single( particles, ii, input_particles, jj );
        st_Particles_set_at_turn_value( particles, ii, 0 );
        st_Particles_set_at_element_id_value( particles, ii, 0 );
    }

    /* ********************************************************************** */
    /* ****  Track particles over the beam-elements for NUM_TURNS turns  **** */
    /* ********************************************************************** */

    start_tracking_time = st_Time_get_seconds_since_epoch();

    st_Track_all_particles_until_turn(
        particles, beam_elements_buffer, NUM_TURNS );

    end_tracking_time = st_Time_get_seconds_since_epoch();

    tracking_time = ( end_tracking_time >= start_tracking_time )
        ? ( end_tracking_time - start_tracking_time ) : 0.0;

    printf( "Tracking time : %10.6f \r\n"
            "              : %10.6f / turn \r\n"
            "              : %10.6f / turn / particle \r\n\r\n",
            tracking_time, tracking_time / NUM_TURNS,
            tracking_time / ( NUM_TURNS * NUM_PARTICLES ) );

    /* ********************************************************************** */
    /* ****                         Clean-up                             **** */
    /* ********************************************************************** */

    st_Buffer_delete( particles_buffer );
    st_Buffer_delete( input_particle_buffer );
    st_Buffer_delete( beam_elements_buffer );
}

/* end: examples/c99/track_bbsimple.c */
