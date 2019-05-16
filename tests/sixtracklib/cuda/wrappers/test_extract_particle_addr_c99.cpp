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


TEST( C99_CudaWrappersExtractParticleAddrTest, StoreAllParticleAddressesFromBuffer )
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

    status_t status = ::NS(ARCH_STATUS_SUCCESS);

    /* -------------------------------------------------------------------- */
    /* Prepare the particles buffer */

    c_buffer_t* minimal_particles_buffer = ::NS(Buffer_new)( size_t{ 0 } );
    c_buffer_t* particles_buffer  = ::NS(Buffer_new)( size_t{ 0 } );
    c_buffer_t* paddresses_buffer = ::NS(Buffer_new)( size_t{ 0 } );
    c_buffer_t* cmp_paddresses_buffer = ::NS(Buffer_new)( size_t{ 0 } );

    size_t constexpr num_psets           = size_t{ 2048 };
    size_t constexpr drift_index         = size_t{ 42 };
    size_t constexpr start_num_particles = size_t{ 10 };
    size_t constexpr num_particles_mult  = size_t{ 1 };

    size_t num_present_psets = size_t{ 0 };

    for( size_t ii = size_t{ 0 } ; ii < num_psets ; ++ii )
    {
        if( ii != drift_index )
        {
            size_t const num_particles =
                num_particles_mult * ii + start_num_particles;

            particles_t* p = ::NS(Buffer_new)(
                particles_buffer, num_particles );

            SIXTRL_ASSERT( p != nullptr );

            ++num_present_psets;
        }
        else
        {
            double const length = static_cast< double >( ii );
            ::NS(Drift)* drift = ::NS(Drift_add)(
                particles_buffer, length );

            SIXTRL_ASSERT( drift != nullptr );
        }
    }

    SIXTRL_ASSERT( NS(Buffer_get_num_of_objects)(
        particles_buffer ) == num_psets );

    SIXTRL_ASSERT( NS(Particles_buffer_get_num_particle_blocks)(
        particles_buffer ) == num_present_psets );

    particles_t* p = ::NS(Particles_add_copy)(
        minimal_particles_buffer, ::NS(Particles_buffer_get_const_particles)(
            particles_buffer, size_t{ 0 } ) );

    SIXTRL_ASSERT( p != nullptr );

    /* -------------------------------------------------------------------- */
    /* Init the Cuda controller and arguments for the addresses
     * and the particles */

    cuda_ctrl_t* ctrl = ::NS(CudaController_create)();
    SIXTRL_ASSERT( ctrl != nullptr );

    cuda_arg_t* particles_arg = ::NS(CudaArgument_new)( ctrl );
    SIXTRL_ASSERT( particles_arg != nullptr );

    cuda_arg_t* addresses_arg = ::NS(CudaArgument_new)( ctrl );
    SIXTRL_ASSERT( addresses_arg != nullptr );

    result_register_t result_register = ::NS(ARCH_DEBUGGING_GENERAL_FAILURE);

    cuda_arg_t* result_register_arg = ::NS(CudaArgument_new_raw_argument)(
        &result_register, sizeof( result_register ), ctrl );

    SIXTRL_ASSERT( result_register_arg != nullptr );

    /* ********************************************************************* */
    /* The minimal example: plarticles buffer with a single particle set:    */

    status = ::NS(ParticlesAddr_prepare_buffer_based_on_particles_buffer)(
        paddresses_buffer, minimal_particles_buffer );

    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );
    SIXTRL_ASSERT( NS(Buffer_get_num_of_objects)( paddresses_buffer ) ==
                   NS(Buffer_get_num_of_objects)( minimal_particles_buffer ) );

    status = ::NS(ParticlesAddr_prepare_buffer_based_on_particles_buffer)(
        cmp_paddresses_buffer, minimal_particles_buffer );

    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );
    SIXTRL_ASSERT( NS(Buffer_get_num_of_objects)( cmp_paddresses_buffer ) ==
               NS(Buffer_get_num_of_objects)( minimal_particles_buffer ) );

    status = ::NS(ParticlesAddr_buffer_store_all_addresses)(
        cmp_paddresses_buffer, minimal_particles_buffer );

    SIXRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

    /* Send the particles and the addresses buffer to the node */

    status = ::NS(Argument_send_buffer)(
        particles_arg, minimal_particles_buffer );

    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

    status = ::NS(Argument_send_buffer)(
        addresses_arg, paddresses_buffer );

    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

    /* reset the particles address buffer at the host side,
     * so that we can be certain about the success of the operation */

    ::NS(Buffer_clear)( paddresses_buffer, true );
    ::NS(Buffer_reset)( paddresses_buffer );

    SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)( paddresses_buffer ) ==
        size_t{ 0 } );

    /* Setup the kernel config to run the wrapper: */

    size_t const num_kernel_args = size_t{ 4 };
    std::string kernel_name = SIXTRL_C99_NAMESPACE_PREFIX_STR;
    kernel_name += "Particles_buffer_store_all_addresses_cuda_debug";

    size_t const work_items_dim = size_t{ 1 };
    size_t num_work_items  = size_t{ 1 };

    size_t const work_group_dim = size_t{ 1 };
    size_t work_group_size = size_t{ 1 };

    kernel_config_t* ptr_minimal_kernel = ::NS(CudaKernelConfig_new_detailed)(
        ctrl, num_kernel_args, work_items_dim, &num_work_items, work_group_dim,
            &work_group_size, kernel_name.c_str() );

    SIXTRL_ASSERT( ptr_minimal_kernel != nullptr );
    SIXTRL_ASSERT( !::NS(KernelConfig_needs_update)( ptr_minimal_kernel ) );

    /* ===================================================================== */
    /* !!! START OF THE ACTUAL TEST !!! */

    ::NS(Particles_buffer_store_all_addresses_cuda_wrapper)(
        ptr_minimal_kernel, addresses_arg, particles_arg, result_register_arg );

    status = ::NS(Argument_receive_buffer)( paddresses_buffer );
    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS );
    SIXTRL_ASSERT( !::NS(Buffer_needs_remapping)( paddresses_buffer ) );

    status = ::NS(Argument_receive_raw_arg)(
        &result_register, sizeof( result_register ) );

    ASSERT_TRUE( result_register == ::NS(ARCH_DEBUGGING_REGISTER_EMPTY) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( paddresses_buffer ) ==
         ::NS(Buffer_get_num_of_objects)( minimal_particles_buffer ) );

    if( ::NS(Buffer_get_num_of_objects)( paddresses_buffer ) ==
        ::NS(Buffer_get_num_of_objects)( cmp_paddresses_buffer ) )
    {
        buf_size_t const nn =
            ::NS(Buffer_get_num_of_objects)( paddresses_buffer );

        for( size_t ii = size_t{ 0 } ; ii < nn ; ++ii )
        {
            particles_addr_t const* paddr =
                ::NS(ParticlesAddr_buffer_get_const_particle_addr)(
                    paddresses_buffer, ii );

            particles_addr_t const* cmp_paddr =
                ::NS(ParticlesAddr_buffer_get_const_particle_addr)(
                    cmp_paddresses_buffer, ii );

            particles_t const* p = ::NS(Particles_buffer_get_const_particles)(
                minimal_particles_buffer, ii );

            ASSERT_TRUE( paddr != nullptr );
            ASSERT_TRUE( cmp_paddr != nullptr );
            ASSERT_TRUE( cmp_paddr->num_particles == paddr->num_particles );

            ASSERT_TRUE(
                ( ( p != nullptr ) && ( ::NS(Particles_get_num_of_particles)(
                    p ) == paddr->num_particles ) ) ||
                ( ( p == nullptr ) && ( paddr->num_particles == size_t{ 0 } ) )
            );
        }
    }

    /* -------------------------------------------------------------------- */
    /* Full example, only one thread / one block : */

    status = ::NS(ParticlesAddr_prepare_buffer_based_on_particles_buffer)(
        paddresses_buffer, particles_buffer );
    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS ) );

    status = ::NS(ParticlesAddr_prepare_buffer_based_on_particles_buffer)(
        cmp_paddresses_buffer, particles_buffer );
    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

    status = ::NS(ParticlesAddr_buffer_store_all_addresses)(
        cmp_paddresses_buffer, particles_buffer );
    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

    status = ::NS(Argument_send_buffer)( addresses_arg, paddresses_buffer );
    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

    result_register = ::NS(ARCH_DEBUGGING_GENERAL_FAILURE);

    status = ::NS(Argument_send_raw_argument)( result_register_arg,
       &result_register, sizeof( result_register ) );
    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );

    SIXTRL_ASSERT( ::NS(CudaController_is_managed_cobject_buffer_remapped)(
        ctrl, ::NS(CudaArgument_get_cuda_arg_buffer)( particles_arg ),
            ::NS(Buffer_get_slot_size)( particles_buffer ) ) );

    ::NS(Particles_buffer_store_all_addresses_cuda_wrapper)(
        ptr_minimal_kernel, addresses_arg, particles_arg, result_register_arg );

    status = ::NS(Argument_receive_buffer)( paddresses_buffer );
    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS );
    SIXTRL_ASSERT( !::NS(Buffer_needs_remapping)( paddresses_buffer ) );

    status = ::NS(Argument_receive_raw_arg)(
        result_register_arg, &result_register, sizeof( result_register ) );
    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );

    ASSERT_TRUE( result_register == ::NS(ARCH_DEBUGGING_REGISTER_EMPTY) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( paddresses_buffer ) ==
         ::NS(Buffer_get_num_of_objects)( minimal_particles_buffer ) );

    if( ::NS(Buffer_get_num_of_objects)( paddresses_buffer ) ==
        ::NS(Buffer_get_num_of_objects)( cmp_paddresses_buffer ) )
    {
        buf_size_t const nn =
            ::NS(Buffer_get_num_of_objects)( paddresses_buffer );

        for( size_t ii = size_t{ 0 } ; ii < nn ; ++ii )
        {
            particles_addr_t const* paddr =
                ::NS(ParticlesAddr_buffer_get_const_particle_addr)(
                    paddresses_buffer, ii );

            particles_addr_t const* cmp_paddr =
                ::NS(ParticlesAddr_buffer_get_const_particle_addr)(
                    cmp_paddresses_buffer, ii );

            particles_t const* p = ::NS(Particles_buffer_get_const_particles)(
                minimal_particles_buffer, ii );

            ASSERT_TRUE( paddr != nullptr );
            ASSERT_TRUE( cmp_paddr != nullptr );
            ASSERT_TRUE( cmp_paddr->num_particles == paddr->num_particles );

            ASSERT_TRUE(
                ( ( p != nullptr ) && ( ::NS(Particles_get_num_of_particles)(
                    p ) == paddr->num_particles ) ) ||
                ( ( p == nullptr ) && ( paddr->num_particles == size_t{ 0 } ) )
            );
        }
    }

    /* -------------------------------------------------------------------- */
    /* Full example, as many blocks as there are particle sets : */

    status = ::NS(ParticlesAddr_prepare_buffer_based_on_particles_buffer)(
        paddresses_buffer, particles_buffer );
    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS ) );

    status = ::NS(Argument_send_buffer)( addresses_arg, paddresses_buffer );
    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

    result_register = ::NS(ARCH_DEBUGGING_GENERAL_FAILURE);

    status = ::NS(Argument_send_raw_argument)( result_register_arg,
       &result_register, sizeof( result_register ) );
    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );


    ptr_minimal_kernel->setNumWorkItems( num_psets );
    ptr_minimal_kernel->setWorkGourpSize( size_t{ 1 } );
    ptr_minimal_kernel->update();

    SIXTRL_ASSERT( !ptr_minimal_kernel->needsUpdate() );

    SIXTRL_ASSERT( ::NS(CudaController_is_managed_cobject_buffer_remapped)(
        ctrl, ::NS(CudaArgument_get_cuda_arg_buffer)( particles_arg ),
            ::NS(Buffer_get_slot_size)( particles_buffer ) ) );

    ::NS(Particles_buffer_store_all_addresses_cuda_wrapper)(
        ptr_minimal_kernel, addresses_arg, particles_arg, result_register_arg );

    status = ::NS(Argument_receive_buffer)( paddresses_buffer );
    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS );
    SIXTRL_ASSERT( !::NS(Buffer_needs_remapping)( paddresses_buffer ) );

    status = ::NS(Argument_receive_raw_arg)(
        result_register_arg, &result_register, sizeof( result_register ) );
    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );

    ASSERT_TRUE( result_register == ::NS(ARCH_DEBUGGING_REGISTER_EMPTY) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( paddresses_buffer ) ==
         ::NS(Buffer_get_num_of_objects)( minimal_particles_buffer ) );

    if( ::NS(Buffer_get_num_of_objects)( paddresses_buffer ) ==
        ::NS(Buffer_get_num_of_objects)( cmp_paddresses_buffer ) )
    {
        buf_size_t const nn =
            ::NS(Buffer_get_num_of_objects)( paddresses_buffer );

        for( size_t ii = size_t{ 0 } ; ii < nn ; ++ii )
        {
            particles_addr_t const* paddr =
                ::NS(ParticlesAddr_buffer_get_const_particle_addr)(
                    paddresses_buffer, ii );

            particles_addr_t const* cmp_paddr =
                ::NS(ParticlesAddr_buffer_get_const_particle_addr)(
                    cmp_paddresses_buffer, ii );

            particles_t const* p = ::NS(Particles_buffer_get_const_particles)(
                minimal_particles_buffer, ii );

            ASSERT_TRUE( paddr != nullptr );
            ASSERT_TRUE( cmp_paddr != nullptr );
            ASSERT_TRUE( cmp_paddr->num_particles == paddr->num_particles );

            ASSERT_TRUE(
                ( ( p != nullptr ) && ( ::NS(Particles_get_num_of_particles)(
                    p ) == paddr->num_particles ) ) ||
                ( ( p == nullptr ) && ( paddr->num_particles == size_t{ 0 } ) )
            );
        }
    }

    /* -------------------------------------------------------------------- */
    /* Full example, as many blocks as there are particle sets : */

    status = ::NS(ParticlesAddr_prepare_buffer_based_on_particles_buffer)(
        paddresses_buffer, particles_buffer );
    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS ) );

    status = ::NS(Argument_send_buffer)( addresses_arg, paddresses_buffer );
    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

    result_register = ::NS(ARCH_DEBUGGING_GENERAL_FAILURE);

    status = ::NS(Argument_send_raw_argument)( result_register_arg,
       &result_register, sizeof( result_register ) );
    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );


    ptr_minimal_kernel->setNumWorkItems( num_psets );
    ptr_minimal_kernel->setWorkGourpSize( size_t{ 1 } );
    ptr_minimal_kernel->update();

    SIXTRL_ASSERT( !ptr_minimal_kernel->needsUpdate() );

    SIXTRL_ASSERT( ::NS(CudaController_is_managed_cobject_buffer_remapped)(
        ctrl, ::NS(CudaArgument_get_cuda_arg_buffer)( particles_arg ),
            ::NS(Buffer_get_slot_size)( particles_buffer ) ) );

    ::NS(Particles_buffer_store_all_addresses_cuda_wrapper)(
        ptr_minimal_kernel, addresses_arg, particles_arg, result_register_arg );

    status = ::NS(Argument_receive_buffer)( paddresses_buffer );
    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS );
    SIXTRL_ASSERT( !::NS(Buffer_needs_remapping)( paddresses_buffer ) );

    status = ::NS(Argument_receive_raw_arg)(
        result_register_arg, &result_register, sizeof( result_register ) );
    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );

    ASSERT_TRUE( result_register == ::NS(ARCH_DEBUGGING_REGISTER_EMPTY) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( paddresses_buffer ) ==
         ::NS(Buffer_get_num_of_objects)( minimal_particles_buffer ) );

    if( ::NS(Buffer_get_num_of_objects)( paddresses_buffer ) ==
        ::NS(Buffer_get_num_of_objects)( cmp_paddresses_buffer ) )
    {
        buf_size_t const nn =
            ::NS(Buffer_get_num_of_objects)( paddresses_buffer );

        for( size_t ii = size_t{ 0 } ; ii < nn ; ++ii )
        {
            particles_addr_t const* paddr =
                ::NS(ParticlesAddr_buffer_get_const_particle_addr)(
                    paddresses_buffer, ii );

            particles_addr_t const* cmp_paddr =
                ::NS(ParticlesAddr_buffer_get_const_particle_addr)(
                    cmp_paddresses_buffer, ii );

            particles_t const* p = ::NS(Particles_buffer_get_const_particles)(
                minimal_particles_buffer, ii );

            ASSERT_TRUE( paddr != nullptr );
            ASSERT_TRUE( cmp_paddr != nullptr );
            ASSERT_TRUE( cmp_paddr->num_particles == paddr->num_particles );

            ASSERT_TRUE(
                ( ( p != nullptr ) && ( ::NS(Particles_get_num_of_particles)(
                    p ) == paddr->num_particles ) ) ||
                ( ( p == nullptr ) && ( paddr->num_particles == size_t{ 0 } ) )
            );
        }
    }


    /* -------------------------------------------------------------------- */
    /* Cleanup: */

    ::NS(Buffer_delete)( minimal_particles_buffer );
    ::NS(Buffer_delete)( particles_buffer );

    ::NS(Buffer_delete)( paddresses_buffer );
    ::NS(Buffer_delete)( cmp_paddresses_buffer );

    ::NS(Argument_delete)( particles_arg );
    ::NS(Argument_delete)( addresses_arg );
    ::NS(Argument_delete)( result_register_arg );

    ::NS(Controller_delete)( ctrl );
}

/* end: tests/sixtracklib/cuda/wrappers/test_track_job_wrappers_c99.cpp */
