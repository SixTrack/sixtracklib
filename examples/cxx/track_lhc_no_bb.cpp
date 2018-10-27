#include <cassert>
#include <iostream>
#include <iomanip>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.hpp"

int main( int argc, char* argv[] )
{
    namespace st = sixtrack;

    using buf_size_t = st::Buffer::size_type;

    st::Buffer lhc_particle_dump( ::st_PATH_TO_BBSIMPLE_PARTICLES_DUMP );
    st::Buffer lhc_beam_elements_buffer( ::st_PATH_TO_BBSIMPLE_BEAM_ELEMENTS );
    st::Buffer pb( buf_size_t{ 1u << 24u } );

    buf_size_t NUM_PARTICLES = buf_size_t{ 20000 };
    buf_size_t NUM_TURNS     = buf_size_t{ 20 };

    /* ********************************************************************** */
    /* ****   Handling of command line parameters                             */
    /* ********************************************************************** */

    if( argc == 1 )
    {
        std::cout << "Usage: " << argv[ 0 ]
                  << " [NUM_PARTICLES] [NUM_TURNS]\r\n"
                  << "\r\n"
                  << "NUM_PARTICLES: Number of particles for the simulation\r\n"
                  << "               Default = "
                  <<  NUM_PARTICLES << "\r\n\r\n"
                  << "NUM_TURNS    : Number of turns for the simulation\r\n"
                  << "               Default = "
                  <<  NUM_TURNS << "\r\n\r\n";
    }

    if( argc >= 2 )
    {
        int const temp = std::atoi( argv[ 1 ] );
        if( temp > 0 ) NUM_PARTICLES = static_cast< buf_size_t >( temp );
    }

    if( argc >= 3 )
    {
        int const temp = std::atoi( argv[ 2 ] );
        if( temp > 0 ) NUM_TURNS = static_cast< buf_size_t >( temp );
    }

    std::cout << "Selected NUM_PARTICLES = "
              << std::setw( 10 ) << NUM_PARTICLES << "\r\n"
              << "Selected NUM_TURNS     = "
              << std::setw( 10 ) << NUM_TURNS << "\r\n\r\n"
              << std::endl;

    /* ********************************************************************** */
    /* ****   Building Particles Data from LHC Particle Dump Data        **** */
    /* ********************************************************************** */

    st::Particles* particles =
        pb.createNew< st::Particles >( NUM_PARTICLES );

    st::Particles const* input_particles =
        st::Particles::FromBuffer( lhc_particle_dump, 0u );

    buf_size_t const num_input_particles = input_particles->getNumParticles();

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < NUM_PARTICLES ; ++ii )
    {
        buf_size_t const jj = ii % num_input_particles;
        particles->copySingle( input_particles, jj, ii );
    }

    /* ********************************************************************** */
    /* ****  Track particles over the beam-elements for NUM_TURNS turns  **** */
    /* ********************************************************************** */

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < NUM_TURNS ; ++ii )
    {
        // Still C99 API call:
        ::st_Track_particles_beam_elements( particles, &lhc_beam_elements_buffer );
    }
}

/* end: examples/c99/track_lhc_no_bb.cpp */
