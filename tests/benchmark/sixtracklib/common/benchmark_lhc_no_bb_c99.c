#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "sixtracklib/sixtracklib.h"
#include "sixtracklib/testlib.h"

int main()
{
    typedef st_buffer_size_t buf_size_t;

    buf_size_t const NUM_TURNS     = 20u;
    buf_size_t const NUM_PARTICLES = 20000u;

    double program_begin = NS(Time_get_seconds_since_epoch)();
    double track_begin   = program_begin;
    double track_end     = program_begin;

    /* --------------------------------------------------------------------- */

    st_Buffer* be_buffer = st_Buffer_new_from_file(
        st_PATH_TO_TEST_LHC_BEAM_ELEMENTS_DATA_NO_BEAM_BEAM );

    st_Object const* be_begin = st_Buffer_get_const_objects_begin( be_buffer );
    st_Object const* be_end   = st_Buffer_get_const_objects_end( be_buffer );

    /* --------------------------------------------------------------------- */

    st_Buffer* lhc_particles_buffer = st_Buffer_new_from_file(
        st_PATH_TO_TEST_LHC_PARTICLES_DATA_T1_P2_NO_BEAM_BEAM );

    st_Particles const* lhc_particles = ( st_Particles const* )( uintptr_t
        )st_Object_get_begin_addr( st_Buffer_get_const_objects_begin(
            lhc_particles_buffer ) );

    buf_size_t const lhc_num_particles =
        st_Particles_get_num_of_particles( lhc_particles );

    /* --------------------------------------------------------------------- */

    st_Buffer* particles_buffer = st_Buffer_new(
        st_Buffer_calculate_required_buffer_length(
            lhc_particles_buffer, NUM_PARTICLES,
        st_Particles_get_required_num_slots(
            lhc_particles_buffer, NUM_PARTICLES ),
        st_Particles_get_required_num_dataptrs(
            lhc_particles_buffer, NUM_PARTICLES ), ( buf_size_t )0u ) );

    st_Particles* particles =
        st_Particles_new( particles_buffer, NUM_PARTICLES );

    /* --------------------------------------------------------------------- */

    buf_size_t ii = ( buf_size_t )0u;

    for( ; ii < NUM_PARTICLES ; ++ii )
    {
        buf_size_t const jj = ii % lhc_num_particles;
        st_Particles_copy_single( particles, ii, lhc_particles, jj );
    }

    /* --------------------------------------------------------------------- */

    ii = ( buf_size_t )0u;

    track_begin = NS(Time_get_seconds_since_epoch)();

    for( ; ii < NUM_TURNS ; ++ii )
    {
        st_Track_beam_elements_particles( particles, be_begin, be_end );
    }

    track_end = NS(Time_get_seconds_since_epoch)();

    if( ( NUM_PARTICLES * NUM_TURNS ) > ( buf_size_t )0 )
    {
        double const setup_time = ( track_begin >= program_begin )
            ? track_begin - program_begin : ( double )0.0;

        double const track_time = ( track_end >= track_begin )
            ? track_end - track_begin : ( double )0.0;

        double time_per_particle_and_turn =
            track_time / ( NUM_PARTICLES * NUM_TURNS );

        printf( "NUM_TURNS         : %14lu\r\n"
                "NUM_PARTICLES     : %14lu\r\n"
                "NUM_BEAM_ELEMENTS : %14lu\r\n"
                "------------------------------------------------------------"
                "---------------------------------------------------------\r\n"
                "setup time        : %14.8f sec \r\n"
                "\r\n"
                "tracking time     : %14.8f sec for tracking\r\n",
                ( unsigned long )NUM_TURNS, ( unsigned long )NUM_PARTICLES,
                ( unsigned long )st_Buffer_get_num_of_objects( be_buffer ),
                setup_time, track_time );

        if( time_per_particle_and_turn >= ( double )0.1 )
        {
            printf( "                  : %14.4f sec  / ( particle * turn )\r\n",
                    time_per_particle_and_turn / 1e-3 );
        }
        else if( time_per_particle_and_turn >= ( double )1e-4 )
        {
            printf( "                  : %14.4f msec / ( particle * turn )\r\n",
                    time_per_particle_and_turn / 1e-3 );
        }
        else
        {
            printf( "                  : %14.4f usec / ( particle * turn )\r\n",
                    time_per_particle_and_turn / 1e-6 );
        }

        printf( "============================================================"
                "=========================================================\r\n"
                "\r\n" );

    }

    /* --------------------------------------------------------------------- */

    st_Buffer_delete( be_buffer );
    st_Buffer_delete( lhc_particles_buffer );
    st_Buffer_delete( particles_buffer );
}

/* end: tests/benchmark/sixtracklib/common/benchmark_lhc_no_bb_baseline_c99.c */
