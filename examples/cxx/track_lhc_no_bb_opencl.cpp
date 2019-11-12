#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <memory>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.hpp"

int main( int argc, char* argv[] )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using context_t = st::ClContext;
    using st_size_t = context_t::size_type;

    std::unique_ptr< context_t > ptr_ctx = nullptr;

    using st_size_t = st::Buffer::size_type;

    st_size_t NUM_PARTICLES = st_size_t{ 20000 };
    st_size_t NUM_TURNS     = st_size_t{ 20 };

    /* --------------------------------------------------------------------- */
    /* Handle command line arguments: */
    /* --------------------------------------------------------------------- */

    if( argc < 2  )
    {
        ptr_ctx.reset( new context_t );

        std::cout << "Usage: " << argv[ 0 ]
                  << " [ID] [NUM_PARTICLES] [NUM_TURNS] \r\n";

        ptr_ctx->printNodesInfo();

        if( ptr_ctx->numAvailableNodes() > st_size_t{ 0 } )
        {
            std::cout << "INFO            :: "
                      << "Selecting default node\r\n";
        }
        else
        {
            std::cout << "Quitting program!\r\n"
                      << std::endl;

            return 0;
        }

        std::cout << "\r\n[NUM_PARTICLES] :: "
                  << "Number of particles for the simulation\r\n"
                  << "                :: Default = "
                  << NUM_PARTICLES
                  << "\r\n\r\n"
                  << "[NUM_TURNS]     :: "
                  << "Number of turns for the simulation\r\n"
                  << "                :: Default = "
                  << NUM_TURNS
                  << "\r\n";
    }

    if( argc >= 2 )
    {
        ptr_ctx.reset( new context_t( argv[ 1 ], nullptr ) );

        if( !ptr_ctx->hasSelectedNode() )
        {
            std::cout << "Warning         : Provided ID " << argv[ 1 ]
                      << " not found -> use default device instead\r\n";
        }
    }

    if( argc >= 3 )
    {
        int const temp = std::atoi( argv[ 2 ] );
        if( temp > 0 ) NUM_PARTICLES = static_cast< st_size_t >( temp );
    }

    if( argc >= 4 )
    {
        int const temp = std::atoi( argv[ 3 ] );
        if( temp > 0 ) NUM_TURNS = static_cast< st_size_t >( temp );
    }

    if( ( ptr_ctx.get() != nullptr ) && ( !ptr_ctx->hasSelectedNode() ) )
    {
        context_t::node_id_t const default_node_id =
            ptr_ctx->defaultNodeId();

        ptr_ctx->selectNode( default_node_id );
    }

    if( (  ptr_ctx.get() == nullptr ) ||
        ( !ptr_ctx->hasSelectedNode() ) )
    {
        return 0;
    }

    context_t& ctx = *ptr_ctx;

    std::cout << "Selected Node     [ID] = "
              << ctx.selectedNodeIdStr()
              << " ( "
              << ::NS(ComputeNodeInfo_get_name)( ctx.ptrSelectedNodeInfo() )
              << " / "
              << ::NS(ComputeNodeInfo_get_platform)( ctx.ptrSelectedNodeInfo() )
              << " ) \r\n"
              << "Selected NUM_PARTICLES = "
              << std::setw( 10 ) << NUM_PARTICLES << "\r\n"
              << "Selected NUM_TURNS     = "
              << std::setw( 10 ) << NUM_TURNS << "\r\n\r\n"
              << std::endl;

    /* ---------------------------------------------------------------------- */
    /* Prepare the buffers: */
    /* ---------------------------------------------------------------------- */

    st::Buffer lhc_particle_dump( ::NS(PATH_TO_BBSIMPLE_PARTICLES_DUMP) );
    st::Buffer lhc_beam_elements_buffer( ::NS(PATH_TO_BBSIMPLE_BEAM_ELEMENTS) );
    st::Buffer pb( st_size_t{ 1u << 24u } );

    st::Particles* particles = pb.createNew< st::Particles >( NUM_PARTICLES );
    st::Particles const* input_particles =
        st::Particles::FromBuffer( lhc_particle_dump, 0u );

    st_size_t const num_input_particles = input_particles->getNumParticles();

    for( st_size_t ii = st_size_t{ 0 } ; ii < NUM_PARTICLES ; ++ii )
    {
        st_size_t const jj = ii % num_input_particles;
        particles->copySingle( input_particles, jj, ii );
    }

    st::ClArgument particles_arg( pb, &ctx );
    st::ClArgument beam_elements_arg( lhc_beam_elements_buffer, &ctx );

    /* --------------------------------------------------------------------- */
    /* Perform tracking over NUM_TURNS */
    /* --------------------------------------------------------------------- */

    ctx.assign_particles_arg( particles_arg );
    ctx.assign_particle_set_arg( 0u, NUM_PARTICLES );
    ctx.assign_beam_elements_arg( beam_elements_arg );

    ctx.track_until( NUM_TURNS );

    /* --------------------------------------------------------------------- */
    /* Read particle data back to pb Buffer */
    /* --------------------------------------------------------------------- */

    particles_arg.read( pb );

    return 0;
}

/* end: examples/cxx/track_lhc_no_bb_opencl.cpp */
