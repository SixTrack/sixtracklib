#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.h"

#define PARTFILE "python/bbsimple_st_dump.bin"
#define ELEMFILE "python/bbsimple_st_input.bin"

//#define PARTFILE "python/beambeam_st_dump.bin"
//#define ELEMFILE "python/beambeam_st_input.bin"

int main( int argc, char* argv[] )
{
    typedef st_buffer_size_t buf_size_t;

    char path2particle_file[ 256 ];
    char path2beam_elements_file[ 256 ];
    
    memset( path2particle_file, (int)'\0', 256 );
    strncpy( path2particle_file, st_PATH_TO_BASE_DIR, 256 );
    strcat( path2particle_file,  PARTFILE);

    memset( path2beam_elements_file, (int)'\0', 256 );
    strncpy( path2beam_elements_file, st_PATH_TO_BASE_DIR, 256 );
    strcat( path2beam_elements_file, ELEMFILE );

    st_Buffer* lhc_particle_dump = st_Buffer_new_from_file( path2particle_file );

    st_Buffer* lhc_beam_elements_buffer = st_Buffer_new_from_file( path2beam_elements_file );

    st_Buffer* pb = st_Buffer_new( ( buf_size_t )( 1u << 24u ) );

    buf_size_t NUM_PARTICLES                = 2;
    buf_size_t NUM_TURNS                    = 1.;

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
    /* ****   Building Particles Data from LHC Particle Dump Data        **** */
    /* ********************************************************************** */

    st_Buffer* dump = st_Buffer_new(1<<24);//16MB

    particles = st_Particles_new( pb, NUM_PARTICLES );
    input_particles = st_Particles_buffer_get_const_particles( lhc_particle_dump, 0u );
    num_input_particles = st_Particles_get_num_of_particles( input_particles );

    for( ii = 0 ; ii < NUM_PARTICLES ; ++ii )
    {
        buf_size_t const jj = ii % num_input_particles;
        st_Particles_copy_single( particles, ii, input_particles, jj );
    }

    int N_elem = st_Buffer_get_num_of_objects(lhc_beam_elements_buffer);
    printf("N_elem= %d \n",N_elem);

    for( ii = 0 ; ii < NUM_TURNS ; ++ii )
    {
        st_Object* elem = st_Buffer_get_objects_begin( lhc_beam_elements_buffer );
        for( int i_ele=0; i_ele<N_elem; i_ele++)
        {
            st_Track_particle_subset_beam_element_obj( particles, 0, NUM_PARTICLES, elem);
            st_Particles* pdump = st_Particles_add_copy(dump, particles);
            assert(pdump != 0);
            elem++;
            
            //st_Particles_print(stdout, pdump);
        }
        
    }

    printf("N objects in dump %d\n", (int)st_Buffer_get_num_of_objects(dump));

    // Write to file
    st_Buffer_write_to_file(dump, "stlib_dump.bin");
    
    st_Buffer* test = st_Buffer_new_from_file("stlib_dump.bin");
    st_Particles_buffer_print(stdout, test);


    /* ********************************************************************** */
    /* ****                         Clean-up                             **** */
    /* ********************************************************************** */

    st_Buffer_delete( pb );
    st_Buffer_delete( lhc_particle_dump );
    st_Buffer_delete( lhc_beam_elements_buffer );
    st_Buffer_delete( dump );
}

/* end: examples/c99/track_lhc_no_bb.c */
