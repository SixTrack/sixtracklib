#define _USE_MATH_DEFINES


#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>

#include "sixtracklib/sixtracklib.h"
#include "sixtracklib/testlib.h"

int main( int argc, char* argv[] )
{
    ( void )argc;
    ( void )argv;

    std::size_t const NUM_PARTICLE_BLOCKS         = 1u;
    std::size_t const PARTICLES_DATA_CAPACITY     = 1048576u;
    std::size_t const NUM_PARTICLES               = 1000u;

    std::size_t const NUM_BEAM_ELEMENTS           = 100u;
    std::size_t const BEAM_ELEMENTS_DATA_CAPACITY = 1048576u;
    std::size_t const NUM_TURNS                   = 100u;

    st_Blocks* particles_buffer = st_Blocks_new(
        NUM_PARTICLE_BLOCKS, PARTICLES_DATA_CAPACITY );

    st_Blocks* beam_elements = st_Blocks_new(
        NUM_BEAM_ELEMENTS, BEAM_ELEMENTS_DATA_CAPACITY );

    assert( particles_buffer != nullptr );
    assert( beam_elements    != nullptr );

    st_Particles* particles = st_Blocks_add_particles(
        particles_buffer, NUM_PARTICLES );

    assert( particles != nullptr );
    st_Particles_random_init( particles );

    for( std::size_t ii = 0u ; ii < NUM_BEAM_ELEMENTS ; ++ii )
    {
        st_Drift* drift = st_Blocks_add_drift( beam_elements, 1.0 );
        (void) drift;
        assert( drift != nullptr );
    }

    st_Blocks_serialize( particles_buffer );
    st_Blocks_serialize( beam_elements );

     auto start = std::chrono::steady_clock::now();

    /* ********************************************************* */
    /* START CODE TO BENCHMARK HERE:                             */
    /* ********************************************************* */

    for( std::size_t ii = 0 ; ii < NUM_TURNS ; ++ii )
    {
        st_Track_beam_elements( particles, 0u, NUM_PARTICLES, beam_elements, 0 );
    }

    st_Blocks_delete( particles_buffer );
    st_Blocks_delete( beam_elements );

    /* ********************************************************* */
    /* END CODE TO BENCHMARK HERE:                             */
    /* ********************************************************* */

    auto end = std::chrono::steady_clock::now();
    auto const diff = end - start;

    std::cout.precision( 9 );
    std::cout << std::setw( 20 ) << NUM_BEAM_ELEMENTS
              << std::setw( 20 ) << NUM_PARTICLES
              << std::setw( 20 ) << NUM_TURNS
              << std::setw( 20 ) << std::boolalpha << false
              << std::setw( 20 ) << std::boolalpha << false
              << std::setw( 20 )
              << std::chrono::duration< double, std::ratio< 1 > >( diff ).count()
              << std::endl;

    return 0;
}

/* end: tests/benchmark/sixtracklib/common/benchmark_drift_baseline.cpp */
