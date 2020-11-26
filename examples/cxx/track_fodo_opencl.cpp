#include <cmath>
#include <chrono>
#include <iostream>
#include <numeric>
#include <string>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.hpp"

int main( int const argc, char* argv[] )
{
    namespace st = sixtrack;
    using std::sqrt;
    using track_job_type = st::TrackJobCl;

    unsigned int NUM_PARTICLES = 50000; // Default
    unsigned int NUM_TURNS = 10000; // Default
    std::string device_id = "0:0";

    double const Q0 = 1.0;
    double const MASS0 = ( double )SIXTRL_PHYS_CONST_MASS_PROTON_EV;
    double const P0_C = 450.0e9;
    double const MIN_X = 0.0;
    double const MAX_X = 1e-8;
    double const CHI   = 1.0;
    double const CHARGE_RATIO = 1.0;

    if( argc == 1 )
    {
        std::cout << "Usage: " << argv[ 0 ]
                  << "device_id=" << device_id
                  << " num_particles=" << NUM_PARTICLES
                  << " num_turns=" << NUM_TURNS << std::endl;
    }

    if( argc >= 2 )
    {
        device_id = std::string{ argv[ 1 ] };

        if( argc >= 3 )
        {
            NUM_PARTICLES = std::stoi( argv[ 2 ] );

            if( argc >= 4 )
            {
                NUM_TURNS = std::stoi( argv[ 3 ] );
            }
        }
    }

    /* ---------------------------------------------------------------------- */
    /* Build lattice */

    st::Buffer lattice;
    ::NS(TrackTestdata_generate_fodo_lattice)( lattice.getCApiPtr(), NUM_TURNS );

    /* ---------------------------------------------------------------------- */
    /* Init particle distribution */

    st::Buffer pbuffer;
    ::NS(TrackTestdata_generate_particle_distr_x)( pbuffer.getCApiPtr(),
        NUM_PARTICLES, P0_C, MIN_X, MAX_X, MASS0, Q0, CHI, CHARGE_RATIO );

    /* ---------------------------------------------------------------------- */
    /* Create Track Job */

    track_job_type job( device_id, pbuffer, lattice );
    auto start_time = std::chrono::steady_clock::now();
    job.track( NUM_TURNS );
    auto stop_time = std::chrono::steady_clock::now();

    st::collect( job );

    std::chrono::duration< double > const wtime = stop_time - start_time;

    std::cout << "elapsed wall time: " << wtime.count() << " sec\r\n"
              << "                 = " << wtime.count() / ( NUM_PARTICLES * NUM_TURNS )
              << " sec/particles/turn" << std::endl;

    return 0;
}
