#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "sixtracklib/sixtracklib.h"
#include "sixtracklib/testlib.h"

int main( int argc, char* argv[] )
{
    typedef  st_buffer_size_t   buf_size_t;
    typedef  st_Buffer          buffer_t;
    typedef  st_Particles       particles_t;

    #if !defined( SIXTRL_DISABLE_BEAM_BEAM )
    printf( "# Info :: beam-beam elements enabled\r\n" );
    #else /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */
    printf( "# Info :: beam-beam elements disabled\r\n" );
    #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */

    buffer_t* lhc_particle_dump = st_Buffer_new_from_file(
            st_PATH_TO_TEST_LHC_PARTICLES_DATA_T1_P2_NO_BEAM_BEAM );

    buffer_t* lhc_beam_elements_buffer = st_Buffer_new_from_file(
            st_PATH_TO_TEST_LHC_BEAM_ELEMENTS_DATA_NO_BEAM_BEAM );

    particles_t const* input_particles =
            st_Particles_buffer_get_const_particles( lhc_particle_dump, 0u );

    buf_size_t const num_input_particles =
            st_Particles_get_num_of_particles( input_particles );

    buf_size_t  NUM_CONFIGURATIONS   = 0;
    buf_size_t* num_particles_list   = SIXTRL_NULLPTR;
    buf_size_t* num_turns_list       = SIXTRL_NULLPTR;
    buf_size_t* num_repetitions_list = SIXTRL_NULLPTR;

    /* ===================================================================== */
    /* ==== Prepare Host Buffers                                             */

    if( argc == 1 )
    {
        printf( "# Usage %s [path_to_param_file] | \r\n"
                "#          [NUM_PARTICLES] [NUM_TURNS] [NUM_REPETITIONS]\r\n",
                argv[ 0 ] );

        printf( "# \r\n"
                "# path_to_param_file :: path to a text file containing \r\n"
                "#                       lines with NUM_PARTICLES NUM_TURNS "
                                        "NUM_REPETITIONS tripples\r\n"
                "#            default :: %s\r\n"
                "# \r\n", st_PATH_TO_DEFAULT_BENCHMARKS_PARAM_FILE );

        printf( "# \r\n"
                "# path_to_param_file :: path to a text file containing \r\n"
                "#                       lines with NUM_PARTICLES NUM_TURNS "
                                        "NUM_REPETITIONS tripples\r\n"
                "#            default :: %s\r\n"
                "# \r\n", st_PATH_TO_DEFAULT_BENCHMARKS_PARAM_FILE );

    }

    buf_size_t const NUM_CONFIGURATIONS = 22;

    buf_size_t num_particles_list[] =
    {
              1,       16,     128,     256,
            512,     1024,    2048,    4096,
           8192,    10000,   16384,   20000,
          32768,    40000,   65536,  100000,
         200000,   500000, 1000000, 2000000,
        5000000, 10000000
    };

    buf_size_t num_turns_list[] =
    {
        100, 100, 100, 100,
        100, 100,  50,  50,
         50,  50,  20,  20,
         20,  20,  10,  10,
         10,  10,   5,   5,
          1,   1
    };

    buf_size_t num_repetitions_list[] =
    {
          5,   5,   5,   5,
          5,   5,   5,   5,
          5,   5,   5,   5,
          5,   5,   5,   5,
          5,   5,   3,   3,
          3,   3
    };

    buf_size_t kk = ( buf_size_t )0u;

    /* --------------------------------------------------------------------- */
    /* Perform NUM_CONFIGURATIONS benchmark runs: */

    printf( "#      NUM_PARTICLES"
            "           NUM_TURNS"
            "     NUM_REPETITIONS"
            "                 wall time [s]"
            "        min norm wall time [s]"
            "        avg norm wall time [s]"
            "        max norm wall time [s]\r\n" );


    for( ; kk < NUM_CONFIGURATIONS ; ++kk )
    {
        buf_size_t const NUM_PARTICLES   = num_particles_list[ kk ];
        buf_size_t const NUM_TURNS       = num_turns_list[ kk ];
        buf_size_t const NUM_REPETITIONS = num_repetitions_list[ kk ];

        buf_size_t ll = ( buf_size_t )0u;

        double sum_wall_time = ( double )0.0;
        double min_wall_time = ( double )+1e30;
        double max_wall_time = ( double )-1e30;

        int success = 0;

        for( ; ll < NUM_REPETITIONS ; ++ll )
        {
            buffer_t* pb = st_Buffer_new( ( buf_size_t )( 1u << 28u ) );
            particles_t* particles = st_Particles_new( pb, NUM_PARTICLES );

            if( ( success == 0 ) &&
                ( NUM_PARTICLES >   ( buf_size_t )0 ) &&
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
            }
            else
            {
                success = -1;
            }

            /* ------------------------------------------------------------- */
            /* Perform tracking over NUM_TURNS */
            /* ------------------------------------------------------------- */

            if( success == 0 )
            {
                double exec_time = ( double )0.0;
                double const start_time = st_Time_get_seconds_since_epoch();

                buf_size_t jj = ( buf_size_t )0u;

                for(  ; jj < NUM_TURNS ; ++jj )
                {
                    success |= st_Track_particles_beam_elements(
                        particles, lhc_beam_elements_buffer );
                }

                exec_time = st_Time_get_seconds_since_epoch();

                if( exec_time >= start_time )
                {
                    exec_time -= start_time;
                }
                else
                {
                    exec_time = ( double )0.0;
                }

                sum_wall_time += exec_time;

                if( success == 0 )
                {
                    if( min_wall_time > exec_time ) min_wall_time = exec_time;
                    if( max_wall_time < exec_time ) max_wall_time = exec_time;
                }
                else
                {
                    printf( "# ERROR TRACKING \r\n" );
                }
            }

            st_Buffer_delete( pb );
        }

        /* ----------------------------------------------------------------- */
        /* Printout timing */
        /* ----------------------------------------------------------------- */

        if( ( success == 0 ) && ( NUM_REPETITIONS > ( buf_size_t )0u ) )
        {
            double const norm_denom = ( double )( NUM_PARTICLES * NUM_TURNS );
            double const avg_denom  = ( double )NUM_REPETITIONS;

            double const avg_wall_time      = sum_wall_time / avg_denom;
            double const norm_avg_wall_time = avg_wall_time / norm_denom;
            double const norm_min_wall_time = min_wall_time / norm_denom;
            double const norm_max_wall_time = max_wall_time / norm_denom;

            printf( "%20d" "%20d" "%20d"
                    " %29.8f" " %29.8f" " %29.8f" "% 29.8f" "\r\n",
                    ( int )NUM_PARTICLES, ( int )NUM_TURNS,
                    ( int )ll, avg_wall_time, norm_min_wall_time,
                    norm_avg_wall_time, norm_max_wall_time );

            fflush( stdout );
        }
    }

    /* --------------------------------------------------------------------- */
    /* Clean-up */
    /* --------------------------------------------------------------------- */

    st_Buffer_delete( lhc_particle_dump );
    st_Buffer_delete( lhc_beam_elements_buffer );

    return 0;
}

/* end: tests/benchmark/sixtracklib/opencl/benchmark_lhc_no_bb_c99.c */
