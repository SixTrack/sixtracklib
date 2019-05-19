#include "sixtracklib/cuda/wrappers/track_job_wrappers.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include <gtest/gtest.h>
#include <cuda_runtime_api.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles/particles_addr.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/be_drift.h"
#include "sixtracklib/cuda/controller.h"
#include "sixtracklib/cuda/argument.h"
#include "sixtracklib/cuda/control/kernel_config.h"
#include "sixtracklib/cuda/control/node_info.h"

namespace SIXTRL_CXX_NAMESPACE
{
    namespace tests
    {
        bool run_test(
            std::string const& node_id_str,
            ::NS(buffer_size_t) const num_elements,
            ::NS(buffer_size_t) const num_blocks,
            ::NS(buffer_size_t) const num_threads_per_block,
            ::NS(buffer_size_t) const min_num_particles =
                ::NS(buffer_size_t{ 1 },
            ::NS(buffer_size_t) const max_num_particles =
                ::NS(buffer_size_t{ 2 },
            double const probabilty_to_not_particles = double{ 0 },
            ::NS(buffer_size_t) const seed = ::NS(buffer_size_t){ 20190517 } );
    }
}

TEST( C99_CudaWrappersExtractParticleAddrTest, SingleParticleSetTest )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    ASSERT_TRUE( st::tests::run_test( "", 1, 1,    1   ) );
    ASSERT_TRUE( st::tests::run_test( "", 1, 2048, 1   ) );
    ASSERT_TRUE( st::tests::run_test( "", 1, 1,    256 ) );
}

TEST( C99_CudaWrappersExtractParticleAddrTest, MultiParticleSetTest )
{
    ASSERT_TRUE( st::tests::run_test( "", 2048, 1,    1   ) );
    ASSERT_TRUE( st::tests::run_test( "", 2048, 2048, 1   ) );
    ASSERT_TRUE( st::tests::run_test( "", 2048, 1,    256 ) );
    ASSERT_TRUE( st::tests::run_test( "", 2048, 8,    256 ) );
    ASSERT_TRUE( st::tests::run_test( "", 2048, 16,   256 ) );
}

TEST( C99_CudaWrappersExtractParticleAddrTest, MultiParticleSetWithNonParticles )
{
    ASSERT_TRUE( st::tests::run_test( "", 2048, 1,    1  , 1, 2, 0.1 ) );
    ASSERT_TRUE( st::tests::run_test( "", 2048, 2048, 1  , 1, 2, 0.1 ) );
    ASSERT_TRUE( st::tests::run_test( "", 2048, 1,    256, 1, 2, 0.1 ) );
    ASSERT_TRUE( st::tests::run_test( "", 2048, 8,    256, 1, 2, 0.1 ) );
    ASSERT_TRUE( st::tests::run_test( "", 2048, 16,   256, 1, 2, 0.1 ) );
}

namespace SIXTRL_CXX_NAMESPACE
{
    namespace tests
    {
        bool run_test(
            ::NS(CudaController)* SIXTRL_RESTRICT ctrl,
            ::NS(CudaKernelConfig)* SIXTRL_RESTRICT ptr_kernel_config,
            ::NS(buffer_size_t) const num_elements,
            ::NS(buffer_size_t) const min_num_particles,
            ::NS(buffer_size_t) const max_num_particles,
            double const probabilty_to_not_particles,
            ::NS(buffer_size_t) const seed = ::NS(buffer_size_t){ 20190517 } )
        {
            using particles_t        = ::NS(Particles);
            using particles_addr_t   = ::NS(ParticlesAddr);
            using c_buffer_t         = ::NS(Buffer);

            using cuda_ctrl_t        = ::NS(CudaController);
            using cuda_arg_t         = ::NS(CudaArgument);
            using cuda_kernel_conf_t = ::NS(CudaKernelConfig);
            using kernel_id_t        = ::NS(ctrl_kernel_id_t);
            using size_t             = ::NS(buffer_size_t);
            using status_t           = ::NS(arch_status_t);
            using result_register_t  = ::NS(arch_debugging_t);
            using address_t          = ::NS(buffer_addr_t);
            using addr_diff_t        = ::NS(buffer_addr_diff_t);

            bool success = false;

            status_t status = ::NS(ARCH_STATUS_SUCCESS);

            if( ( ctrl == nullptr ) ||
                ( ptr_kernel_config == nullptr ) ||
                ( ptr_kernel_config->needsUpdate() ) )
            {
                return success;
            }

            /* -------------------------------------------------------------------- */
            /* Prepare the cmp particles and particles addr buffer; the former is
             * used to verify the results of the test */

            c_buffer_t* particles_buffer  = ::NS(Buffer_new)( size_t{ 0 } );
            c_buffer_t* cmp_paddr_buffer  = ::NS(Buffer_new)( size_t{ 0 } );

            SIXTRL_ASSERT( cmp_paddr_buffer != nullptr );
            SIXTRL_ASSERT( particles_buffer != nullptr );

            size_t const slot_size =
                ::NS(Buffer_get_slot_size)( particles_buffer );

            SIXTRL_ASSERT( slot_size > size_t{ 0 } );

            size_t constexpr prng_seed = size_t{ 20190517 };

            size_t constexpr num_psets           = size_t{ 1 };
            size_t constexpr min_num_particles   = size_t{ 100 };
            size_t constexpr max_num_particles   = size_t{ 200 };

            /* Enforce that we always have  a NS(Particles) instance in this test */
            double const probabilty_to_not_particles = double{ 0.0 };

            status = ::NS(TestParticlesAddr_prepare_buffers)( particles_buffer,
                cmp_paddr_buffer, num_psets, min_num_particles, max_num_particles,
                    probabilty_to_not_particles, prng_seed );

            SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

            status = ::NS(ParticlesAddr_buffer_store_all_addresses)(
                cmp_paddr_buffer, particles_buffer );

            SIXRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

            status = ::NS(TestParticlesAddr_verify_structure)(
                cmp_paddr_buffer, particles_buffer );

            SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

            status = ::NS(TestParticlesAddr_verify_addresses)(
                cmp_paddr_buffer, particles_buffer );

            SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

            /* --------------------------------------------------------------------- */
            /* Prepare the actual paddr_buffer that is used for the test */

            c_buffer_t* paddr_buffer = ::NS(Buffer_new)( size_t{ 0 } );

            SIXTRL_ASSERT( paddr_buffer != nullptr );

            status = ::NS(ParticlesAddr_prepare_buffer_based_on_particles_buffer)(
                paddr_buffer, particles_buffer );

            SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

            /* -------------------------------------------------------------------- */
            /* Init the Cuda controller and arguments for the addresses
             * and the particles */

            success = true;

            cuda_arg_t* particles_arg = ::NS(CudaArgument_new)( ctrl );
            SIXTRL_ASSERT( particles_arg != nullptr );

            cuda_arg_t* addresses_arg = ::NS(CudaArgument_new)( ctrl );
            SIXTRL_ASSERT( addresses_arg != nullptr );

            result_register_t result_register = ::NS(ARCH_DEBUGGING_GENERAL_FAILURE);

            cuda_arg_t* result_register_arg =
            ::NS(CudaArgument_new_raw_argument)(
                &result_register, sizeof( result_register ), ctrl );

            SIXTRL_ASSERT( result_register_arg != nullptr );

            /* Send the particles and the addresses buffer to the node */

            status = ::NS(Argument_send_buffer)( particles_arg, particles_buffer );
            SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

            status = ::NS(Argument_send_buffer)( addresses_arg, paddr_buffer );
            SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

            /* reset the particles address buffer at the host side,
             * so that we can be certain about the success of the operation */

            ::NS(Buffer_clear)( paddr_buffer, true );
            ::NS(Buffer_reset)( paddr_buffer );

            SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)( paddr_buffer ) ==
                size_t{ 0 } );

            /* ===================================================================== */
            /* !!! START OF THE ACTUAL TEST !!! */

            ::NS(Particles_buffer_store_all_addresses_cuda_wrapper)(
                ptr_kernel_config, addresses_arg, particles_arg,
                    result_register_arg );

            status = ::NS(Argument_receive_buffer)( paddr_buffer );
            SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS );
            SIXTRL_ASSERT( !::NS(Buffer_needs_remapping)( paddr_buffer ) );

            status = ::NS(Argument_receive_raw_arg)(
                &result_register, sizeof( result_register ) );
            SIXTRL_ASSERT( status == :NS(ARCH_STATUS_SUCCESS) );

            if( result_register != ::NS(ARCH_DEBUGGING_REGISTER_EMPTY) )
            {
                success = false;
            }

            success &= ( ::NS(TestParticlesAddr_verify_structure)( paddr_buffer,
               particles_buffer ) == ::NS(ARCH_STATUS_SUCCESS) );

            if( success )
            {
                status = ::NS(Argument_receive_buffer_without_remap)(
                    particles_arg, particles_buffer );
                SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

                address_t const device_begin_addr =
                ::NS(ManagedBuffer_get_stored_begin_addr)(
                    particles_buffer, slot_size );

                address_t const host_begin_addr =
                ::NS(ManagedBuffer_get_buffer_begin_addr)(
                    particles_buffer, slot_size );

                /* addr_offset = host_begin_addr - device_begin_addr -->
                 * host_begin_addr = device_begin_addr + addr_offset */

                addr_diff_t const addr_offset =
                    host_begin_addr - device_begin_addr;



            }

            /* Cleanup: */

            ::NS(Argument_delete)( particles_arg );
            ::NS(Argument_delete)( addresses_arg );
            ::NS(Argument_delete)( result_register_arg );

            ::NS(Buffer_delete)( paddr_buffer );
            ::NS(Buffer_delete)( particles_buffer );
            ::NS(Buffer_delete)( cmp_paddr_buffer );

            ::NS(Controller_delete)( ctrl );

            return success;
        }
    }
}

/* end: tests/sixtracklib/cuda/wrappers/test_track_job_wrappers_c99.cpp */
