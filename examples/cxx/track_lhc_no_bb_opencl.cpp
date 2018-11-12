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
    namespace st = sixtrack;

    std::unique_ptr< st::ClContext > ptr_context = nullptr;

    using buf_size_t = st::Buffer::size_type;

    buf_size_t NUM_PARTICLES = buf_size_t{ 20000 };
    buf_size_t NUM_TURNS     = buf_size_t{ 20 };

    /* --------------------------------------------------------------------- */
    /* Handle command line arguments: */
    /* --------------------------------------------------------------------- */

    if( argc < 2  )
    {
        ptr_context.reset( new st::ClContext );

        std::cout << "Usage: " << argv[ 0 ]
                  << " [ID] [NUM_PARTICLES] [NUM_TURNS] \r\n";

        ptr_context->printNodesInfo();

        if( ptr_context->numAvailableNodes() > buf_size_t{ 0 } )
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
        ptr_context.reset( new st::ClContext( argv[ 1 ] ) );

        if( !ptr_context->hasSelectedNode() )
        {
            std::cout << "Warning         : Provided ID " << argv[ 1 ]
                      << " not found -> use default device instead\r\n";
        }
    }

    if( argc >= 3 )
    {
        int const temp = std::atoi( argv[ 2 ] );
        if( temp > 0 ) NUM_PARTICLES = static_cast< buf_size_t >( temp );
    }

    if( argc >= 4 )
    {
        int const temp = std::atoi( argv[ 3 ] );
        if( temp > 0 ) NUM_TURNS = static_cast< buf_size_t >( temp );
    }

    if( ( ptr_context.get() != nullptr ) &&
        ( !ptr_context->hasSelectedNode() ) )
    {
        st::ClContext::node_id_t const default_node_id =
            ptr_context->defaultNodeId();

        ptr_context->selectNode( default_node_id );
    }

    if( (  ptr_context.get() == nullptr ) ||
        ( !ptr_context->hasSelectedNode() ) )
    {
        return 0;
    }

    st::ClContext& context = *ptr_context;

    std::cout << "Selected Node     [ID] = "
              << context.selectedNodeIdStr()
              << " ( "
              << ::st_ComputeNodeInfo_get_name( context.ptrSelectedNodeInfo() )
              << " / "
              << ::st_ComputeNodeInfo_get_platform( context.ptrSelectedNodeInfo() )
              << " ) \r\n"
              << "Selected NUM_PARTICLES = "
              << std::setw( 10 ) << NUM_PARTICLES << "\r\n"
              << "Selected NUM_TURNS     = "
              << std::setw( 10 ) << NUM_TURNS << "\r\n\r\n"
              << std::endl;

    /* ---------------------------------------------------------------------- */
    /* Prepare the buffers: */
    /* ---------------------------------------------------------------------- */

    st::Buffer lhc_particle_dump( ::st_PATH_TO_BBSIMPLE_PARTICLES_DUMP );
    st::Buffer lhc_beam_elements_buffer( ::st_PATH_TO_BBSIMPLE_BEAM_ELEMENTS );
    st::Buffer pb( buf_size_t{ 1u << 24u } );

    st::Particles* particles = pb.createNew< st::Particles >( NUM_PARTICLES );
    st::Particles const* input_particles =
        st::Particles::FromBuffer( lhc_particle_dump, 0u );

    buf_size_t const num_input_particles = input_particles->getNumParticles();

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < NUM_PARTICLES ; ++ii )
    {
        buf_size_t const jj = ii % num_input_particles;
        particles->copySingle( input_particles, jj, ii );
    }

    st::ClArgument particles_arg( pb, &context );
    st::ClArgument beam_elements_arg( lhc_beam_elements_buffer, &context );

    /* --------------------------------------------------------------------- */
    /* Perform tracking over NUM_TURNS */
    /* --------------------------------------------------------------------- */

    context.track( particles_arg, beam_elements_arg, NUM_TURNS );

    /* --------------------------------------------------------------------- */
    /* Read particle data back to pb Buffer */
    /* --------------------------------------------------------------------- */

    particles_arg.read( pb );

    return 0;
}

/* end: examples/cxx/track_lhc_no_bb_opencl.cpp */
