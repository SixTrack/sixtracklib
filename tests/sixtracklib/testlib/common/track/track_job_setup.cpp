#include "sixtracklib/testlib/common/track/track_job_setup.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/track/track_job_base.hpp"
#include "sixtracklib/common/track/track_job_ctrl_arg_base.hpp"
#include "sixtracklib/common/track/track_job_nodectrl_arg_base.hpp"

namespace st      = SIXTRL_CXX_NAMESPACE;
namespace st_test = SIXTRL_CXX_NAMESPACE::tests;

namespace SIXTRL_CXX_NAMESPACE
{
    namespace tests
    {
        bool TestTrackJob_setup_no_required_output(
            st::TrackJobBaseNew& SIXTRL_RESTRICT_REF job,
            st::Buffer const& SIXTRL_RESTRICT_REF particles_buffer,
            st::Buffer::size_type const num_psets,
            st::Buffer::size_type const* SIXTRL_RESTRICT pset_indices_begin,
            st::Buffer const& SIXTRL_RESTRICT_REF beam_elements_buffer,
            st::Buffer* SIXTRL_RESTRICT ptr_output_buffer )
        {
            using buffer_t = st::Buffer;
            using size_t   = buffer_t::size_type;

            bool success = (
                ( ( ( ptr_output_buffer == nullptr ) &&
                    ( !job.hasOutputBuffer() ) && ( !job.ownsOutputBuffer() ) ) ||
                  ( ( ptr_output_buffer != nullptr ) &&
                    (  job.hasOutputBuffer() ) &&
                    ( !job.ownsOutputBuffer() ) ) ) &&
                ( !job.hasBeamMonitorOutput() ) &&
                (  job.numBeamMonitors() == size_t{ 0 } ) );

            if( success )
            {
                success = ( ( job.beamMonitorIndicesBegin() == nullptr ) &&
                            ( job.beamMonitorIndicesEnd()   == nullptr ) );
            }

            if( success )
            {
                success = ( (  job.numParticleSets() == num_psets ) &&
                    (  job.particleSetIndicesBegin() != nullptr ) &&
                    (  job.particleSetIndicesEnd()   != nullptr ) &&+
                    ( std::equal( job.particleSetIndicesBegin(),
                                  job.particleSetIndicesEnd(),
                                  pset_indices_begin ) ) );
            }

            if( success )
            {
                success = ( ( !job.hasElemByElemOutput() ) &&
                    ( job.ptrElemByElemConfig() == nullptr ) );
            }

            if( success )
            {
                if( ptr_output_buffer != nullptr )
                {
                    success = ( ( job.ptrOutputBuffer() == ptr_output_buffer ) &&
                        ( job.hasOutputBuffer() ) &&
                        ( !job.ownsOutputBuffer() ) );
                }
                else
                {
                    success = ( ( job.ptrOutputBuffer() == nullptr ) &&
                        ( !job.hasOutputBuffer() ) &&
                        ( !job.ownsOutputBuffer() ) );
                }
            }

            if( success )
            {
                success = (
                    ( job.ptrParticlesBuffer() == &particles_buffer ) &&
                    ( job.ptrBeamElementsBuffer() == &beam_elements_buffer ) );
            }

            return success;
        }


    }
}

/* end: tests/sixtracklib/testlib/common/track/track_job_setup.cpp */
