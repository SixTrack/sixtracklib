#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include <iostream>
#include <string>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.hpp"

int main( int argc, char* argv[] )
{
    namespace st = sixtrack;

    using size_t      = st::Buffer::size_type;
    using track_job_t = st::TrackJobCpu;

    st::Buffer input_pb;
    st::Buffer pb;
    st::Buffer eb;

    st::Particles* particles = nullptr;

    int NUM_PARTICLES                  = 0;
    int NUM_TURNS                      = 10;
    int NUM_TURNS_ELEM_BY_ELEM         = 3;
    int OUTPUT_SKIP                    = 1;
    int track_status                   = 0;

    /* -------------------------------------------------------------------- */
    /* Read command line parameters */

    if( argc < 3 )
    {
        std::cout << "Usage: " << argv[ 0 ]
                  << " PATH_TO_PARTICLES PATH_TO_BEAM_ELEMENTS" << std::endl;
        return 0;
    }

    if( ( argc >= 3 ) &&
        ( argv[ 1 ] != nullptr ) && ( std::strlen( argv[ 1 ] ) > 0u ) &&
        ( argv[ 2 ] != nullptr ) && ( std::strlen( argv[ 2 ] ) > 0u ) )
    {
        input_pb.readFromFile( argv[ 1 ] );
        eb.readFromFile( argv[ 2 ] );

        NUM_PARTICLES = st::Particles::FromBuffer(
            input_pb, 0u )->getNumParticles();
    }

    /* --------------------------------------------------------------------- */
    /* Prepare input and tracking data from run-time parameters: */

    std::printf("%-30s = %10d\n","NUM_PARTICLES",NUM_PARTICLES);
    std::printf("%-30s = %10d\n","NUM_TURNS",NUM_TURNS);
    std::printf("%-30s = %10d\n","NUM_TURNS_ELEM_BY_ELEM",NUM_TURNS_ELEM_BY_ELEM);
    std::printf("%-30s = %10d\n","OUTPUT_SKIP", OUTPUT_SKIP);

    if( NUM_PARTICLES >= 0 )
    {
        particles = pb.createNew< st::Particles >(
            st::Particles::FromBuffer( input_pb, 0u )->getNumParticles() );
    }

    if( ( eb.getNumObjects() > size_t{ 0 } ) &&
        ( NUM_TURNS > NUM_TURNS_ELEM_BY_ELEM ) )
    {
        st::BeamMonitor* beam_monitor = eb.createNew< st::BeamMonitor >();
        beam_monitor->setNumStores( NUM_TURNS - NUM_TURNS_ELEM_BY_ELEM );
        beam_monitor->setStart( NUM_TURNS_ELEM_BY_ELEM );
        beam_monitor->setSkip( OUTPUT_SKIP );
        beam_monitor->setIsRolling( true );
    }

    /* ********************************************************************* */
    /* ****               PERFORM TRACKING OPERATIONS                ******* */
    /* ********************************************************************* */

    if( ( particles != nullptr ) && ( NUM_PARTICLES > 0 ) && ( NUM_TURNS > 0 ) )
    {
        /* Create the track_job */
        track_job_t job( pb, eb, nullptr, NUM_TURNS_ELEM_BY_ELEM );

        if( NUM_TURNS_ELEM_BY_ELEM > 0 )
        {
            track_status |= st::trackElemByElem( job, NUM_TURNS_ELEM_BY_ELEM );
        }

        if( NUM_TURNS > NUM_TURNS_ELEM_BY_ELEM )
        {
            track_status |= st::trackUntil( job, NUM_TURNS );
        }

        /* ****************************************************************** */
        /* ****               PERFORM OUTPUT OPERATIONS                ****** */
        /* ****************************************************************** */

        /* NOTE: for the CPU Track Job, collect (currently) performs no
         * operations. Since this *might* change in the future, it's
         * mandated to always call NS(TrackJobCpu_collect)() before
         * accessing the particles, the beam elements or the
         * output buffer */

        st::collect( job );

        if( track_status == 0 )
        {
            st::Buffer* out_buffer = nullptr;

            if( job.hasOutputBuffer() )
            {
                out_buffer = job.ptrOutputBuffer();
            }

            if( job.hasElemByElemOutput() )
            {
                ::st_ElemByElemConfig const* elem_by_elem_config =
                    job.ptrElemByElemConfig();

                size_t const out_offset = job.elemByElemOutputBufferOffset();

                st_Particles* elem_by_elem_output = st::Particles::FromBuffer(
                    *out_buffer, out_offset );

                /* ::st_Particles_print_out(
                       elem_by_elem_output->getCApiPtr() ); */

                ( void )elem_by_elem_config;
                ( void )elem_by_elem_output;
            }

            if( job.hasBeamMonitorOutput() )
            {
                st_size_t const bemon_start_index =
                    job.beamMonitorsOutputBufferOffset();

                st_size_t const bemon_stop_index =
                    bemon_start_index + job.numBeamMonitors();

                st_size_t ii = bemon_start_index;

                for(  ; ii < bemon_stop_index ; ++ii )
                {
                    st::Particles* out_particles = st::Particles::FromBuffer(
                        *out_buffer, ii );

                    /* ::st_Particles_print_out(
                           out_particles->getCApiPtr() ); */

                    ( void )out_particles;
                }
            }
        }
    }

    return 0;
}

/* end: examples/cxx/track_job_cpu.cpp */
