#ifndef SIXTRACK_TESTLIB_COMMON_PARTICLES_PARTICLES_ADDR_CXX_HPP__
#define SIXTRACK_TESTLIB_COMMON_PARTICLES_PARTICLES_ADDR_CXX_HPP__

#include <cstddef>
#include <cstdlib>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/control/controller_base.hpp"
#include "sixtracklib/common/control/argument_base.hpp"

namespace SIXTRL_CXX_NAMESPACE
{
    namespace tests
    {
        SIXTRL_HOST_FN SIXTRL_CXX_NAMESPACE::arch_status_t
        TestParticlesAddr_prepare_buffers(
            SIXTRL_CXX_NAMESPACE::Buffer& SIXTRL_RESTRICT_REF paddr_buffer,
            SIXTRL_CXX_NAMESPACE::Buffer& SIXTRL_RESTRICT_REF particles_buffer,
            SIXTRL_CXX_NAMESPACE::buffer_size_t const num_elements,
            SIXTRL_CXX_NAMESPACE::buffer_size_t const min_num_particles,
            SIXTRL_CXX_NAMESPACE::buffer_size_t const max_num_particles,
            double const probablity_for_non_particles );

        SIXTRL_HOST_FN SIXTRL_CXX_NAMESPACE::arch_status_t
        TestParticlesAddr_verify_structure(
            SIXTRL_CXX_NAMESPACE::Buffer const&
                SIXTRL_RESTRICT_REF paddr_buffer,
            SIXTRL_CXX_NAMESPACE::Buffer const&
                SIXTRL_RESTRICT_REF particles_buffer );

        SIXTRL_HOST_FN SIXTRL_CXX_NAMESPACE::arch_status_t
        TestParticlesAddr_verify_addresses(
            SIXTRL_CXX_NAMESPACE::Buffer const&
                SIXTRL_RESTRICT_REF paddr_buffer,
            SIXTRL_CXX_NAMESPACE::Buffer const&
                SIXTRL_RESTRICT_REF particles_buffer );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool TestParticlesAddr_prepare_ctrl_args_test(
            SIXTRL_CXX_NAMESPACE::ControllerBase* SIXTRL_RESTRICT ctrl,
            SIXTRL_CXX_NAMESPACE::ArgumentBase* SIXTRL_RESTRICT paddr_arg,
            SIXTRL_CXX_NAMESPACE::Buffer& SIXTRL_RESTRICT_REF paddr_buffer,
            SIXTRL_CXX_NAMESPACE::ArgumentBase* SIXTRL_RESTRICT particles_arg,
            SIXTRL_CXX_NAMESPACE::Buffer& SIXTRL_RESTRICT_REF particles_buffer,
            SIXTRL_CXX_NAMESPACE::ArgumentBase* SIXTRL_RESTRICT result_arg );
        
        SIXTRL_HOST_FN bool TestParticlesAddr_verify_ctrl_args_test(
            SIXTRL_CXX_NAMESPACE::ControllerBase* SIXTRL_RESTRICT ctrl,
            SIXTRL_CXX_NAMESPACE::ArgumentBase* SIXTRL_RESTRICT paddr_arg,
            SIXTRL_CXX_NAMESPACE::Buffer& SIXTRL_RESTRICT_REF paddr_buffer,
            SIXTRL_CXX_NAMESPACE::ArgumentBase* SIXTRL_RESTRICT particles_arg,
            SIXTRL_CXX_NAMESPACE::Buffer& SIXTRL_RESTRICT_REF particles_buffer,
            SIXTRL_CXX_NAMESPACE::ArgumentBase* SIXTRL_RESTRICT result_arg );
    }
}

#endif /* SIXTRACK_TESTLIB_COMMON_PARTICLES_PARTICLES_ADDR_CXX_HPP__ */

/* end: tests/sixtracklib/testlib/common/particles/particles_addr.hpp */
