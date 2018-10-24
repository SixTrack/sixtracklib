#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.h"

int main( int argc, char* argv[] )
{
    st_Buffer* particle_dump = st_Buffer_new_from_file(
        st_PATH_TO_BBSIMPLE_PARTICLES_SIXTRACK_DUMP );

    st_Buffer* beam_elements_buffer = st_Buffer_new_from_file(
        st_PATH_TO_BBSIMPLE_BEAM_ELEMENTS );

    st_Buffer* pb = st_Buffer_new( 0u );

    buf_size_t NUM_PARTICLES                = 2;
    buf_size_t NUM_TURNS                    = 2;

    st_Particles*       particles           = SIXTRL_NULLPTR;
    st_Particles const* input_particles     = SIXTRL_NULLPTR;
    buf_size_t          num_input_particles = 0;

    buf_size_t ii = 0;


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
            NUM_PARTICLES = ( buf_size_t )temp;
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

    printf( "Selected NUM_PARTICLES = %10lu\r\n"
            "Selected NUM_TURNS     = %10lu\r\n"
            "\r\n", NUM_PARTICLES, NUM_TURNS );

    /* ********************************************************************** */
    /* ****      Building Particles Data from Particle Dump Data         **** */
    /* ********************************************************************** */

    st_Buffer* dump = st_Buffer_new( 0u );

    particles = st_Particles_new( pb, NUM_PARTICLES );
    input_particles = st_Particles_buffer_get_const_particles( particle_dump, 0u );
    num_input_particles = st_Particles_get_num_of_particles( input_particles );

    for( ii = 0 ; ii < NUM_PARTICLES ; ++ii )
    {
        buf_size_t const jj = ii % num_input_particles;
        st_Particles_copy_single( particles, ii, input_particles, jj );
    }

    int const N_elem = st_Buffer_get_num_of_objects( beam_elements_buffer );
    printf("N_elem= %d \n",N_elem);

    for( ii = 0 ; ii < NUM_TURNS ; ++ii )
    {
        int jj = 0;

        for( ; jj < N_elem; ++jj )
        {
            st_Particles* pdump = st_Particles_add_copy(dump, particles);
            assert(pdump != 0);

            int const ret = st_Track_particles_beam_element(
                particles, beam_elements_buffer, jj );

            assert( ret == 0 );

            #if defined( NDEBUG )
            ( void )pdump;
            ( void )ret;
            #endif /* !defined( NDEBUG ) */
        }
    }

    printf("N objects in dump %d\n", (int)st_Buffer_get_num_of_objects(dump));

    // Write to file
    st_Buffer_write_to_file(dump, st_PATH_TO_BBSIMPLE_PARTICLES_DUMP );

    //st_Buffer* test = st_Buffer_new_from_file("stlib_dump.bin");
    //st_Particles_buffer_print(stdout, test);


    /* ********************************************************************** */
    /* ****                         Clean-up                             **** */
    /* ********************************************************************** */

    st_Buffer_delete( pb );
    st_Buffer_delete( particle_dump );
    st_Buffer_delete( beam_elements_buffer );
    st_Buffer_delete( dump );
}

/* end: examples/c99/track_simple_bb_from_python.c */
