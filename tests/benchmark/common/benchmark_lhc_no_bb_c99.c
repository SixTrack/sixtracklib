#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "sixtracklib/sixtracklib.h"
#include "sixtracklib/testlib.h"

int main()
{
    typedef  st_buffer_size_t   buf_size_t;
    typedef  st_Buffer          buffer_t;
    typedef  st_Particles       particles_t;

    /* ===================================================================== */
    /* ==== Prepare Host Buffers                                             */

    buf_size_t const NUM_CONFIGURATIONS = 13;

    buf_size_t num_particles_list[] =
    {
        1, 16, 128, 1024, 2048, 4096, 8192, 10000, 16384, 20000, 32768, 40000,
        65536, 100000, 200000, 500000, 1000000
    };

    buf_size_t num_turns_list[] =
    {
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5, 5, 5, 2
    };

    buf_size_t kk = ( buf_size_t )0u;

    double start_time = 0.0;
    double end_time   = 0.0;

    /* --------------------------------------------------------------------- */
    /* Perform NUM_CONFIGURATIONS benchmark runs: */

    printf( "                 NUM_PARTICLES"
            "                     NUM_TURNS"
            "                 wall time [s]"
            "            norm wall time [s]\r\n" );


    for( ; kk < NUM_CONFIGURATIONS ; ++kk )
    {
        buf_size_t const NUM_PARTICLES = num_particles_list[ kk ];
        buf_size_t const NUM_TURNS     = num_turns_list[ kk ];

        buffer_t* lhc_particle_dump = st_Buffer_new_from_file(
            st_PATH_TO_TEST_LHC_PARTICLES_DATA_T1_P2_NO_BEAM_BEAM );

        buffer_t* lhc_beam_elements_buffer = st_Buffer_new_from_file(
            st_PATH_TO_TEST_LHC_BEAM_ELEMENTS_DATA_NO_BEAM_BEAM );

        buffer_t* pb = st_Buffer_new( ( buf_size_t )( 1u << 28u ) );

        particles_t* particles = st_Particles_new( pb, NUM_PARTICLES );

        particles_t const* input_particles =
            st_Particles_buffer_get_const_particles( lhc_particle_dump, 0u );

        buf_size_t const num_input_particles =
            st_Particles_get_num_of_particles( input_particles );

        int success = -1;

        if( ( NUM_PARTICLES >   ( buf_size_t )0 ) &&
            ( NUM_TURNS     >   ( buf_size_t )0 ) &&
            ( particles       != SIXTRL_NULLPTR ) &&
            ( input_particles != SIXTRL_NULLPTR ) &&
            ( num_input_particles > ( buf_size_t )0u ) )
        {
            buf_size_t ii = ( buf_size_t )0;

            for(  ; ii < NUM_PARTICLES ; ++ii )
            {
                buf_size_t const jj = ii % num_input_particles;
                st_Particles_copy_single( particles, ii, input_particles, jj );
            }

            success = 0;
        }

        /* ----------------------------------------------------------------- */
        /* Perform tracking over NUM_TURNS */
        /* ----------------------------------------------------------------- */

        if( success == 0 )
        {
            start_time = st_Time_get_seconds_since_epoch();

            buf_size_t jj = ( buf_size_t )0u;

            for(  ; jj < NUM_TURNS ; ++jj )
            {
                success |= st_Track_particles_beam_elements(
                    particles, lhc_beam_elements_buffer );
            }

            end_time = st_Time_get_seconds_since_epoch();

            if( success != 0 )
            {
                printf( "ERROR TRACKING \r\n" );
            }
        }

        /* ----------------------------------------------------------------- */
        /* Printout timing */
        /* ----------------------------------------------------------------- */

        if( success == 0 )
        {
            double const wall_time = ( end_time >= start_time )
                ? ( end_time - start_time ) : ( double )0.0;

            double const norm_wall_time =
                wall_time / ( double )( NUM_TURNS * NUM_PARTICLES );

            printf( "%30d" "%30d" "%30.6f" "%30.6f\r\n",
                    ( int )NUM_PARTICLES, ( int )NUM_TURNS,
                    wall_time, norm_wall_time );
        }

        /* ----------------------------------------------------------------- */
        /* Clean-up */
        /* ----------------------------------------------------------------- */

        st_Buffer_delete( lhc_particle_dump );
        st_Buffer_delete( lhc_beam_elements_buffer );
        st_Buffer_delete( pb );
    }

    return 0;
}

/* end: tests/benchmark/sixtracklib/opencl/benchmark_lhc_no_bb_c99.c */
