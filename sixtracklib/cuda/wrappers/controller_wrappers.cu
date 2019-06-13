#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/wrappers/controller_wrappers.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cuda_runtime_api.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/control/definitions.h"
//     #include "sixtracklib/common/control/kernel_config_base.h"

    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/control/kernel_config.h"
    #include "sixtracklib/cuda/kernels/managed_buffer_remap.cuh"
    #include "sixtracklib/cuda/argument.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

void NS(Buffer_remap_cuda_wrapper)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(cuda_arg_buffer_t) SIXTRL_RESTRICT managed_buffer_begin,
    NS(buffer_size_t) const slot_size )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_managed_buffer_t;

    dim3 const* ptr_blocks =
        NS(CudaKernelConfig_get_ptr_const_blocks)( kernel_config );

    dim3 const* ptr_threads =
        NS(CudaKernelConfig_get_ptr_const_threads_per_block)( kernel_config );

    /* kernel config */

    SIXTRL_ASSERT( ptr_blocks  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ptr_threads != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( NS(KernelConfig_get_arch_id)( kernel_config ) ==
        NS(ARCHITECTURE_CUDA) );

    SIXTRL_ASSERT( !NS(KernelConfig_needs_update)( kernel_config ) );

    SIXTRL_ASSERT( managed_buffer_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( slot_size > ( NS(buffer_size_t) )0u );

    NS(ManagedBuffer_remap_cuda)<<< *ptr_blocks, *ptr_threads >>>(
        reinterpret_cast< ptr_managed_buffer_t >( managed_buffer_begin ),
        slot_size );

    ::cudaDeviceSynchronize();
}

void NS(Buffer_remap_cuda_debug_wrapper)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT conf,
    NS(cuda_arg_buffer_t) SIXTRL_RESTRICT managed_buffer_begin,
    NS(buffer_size_t) const slot_size,
    NS(cuda_arg_buffer_t) SIXTRL_RESTRICT dbg_register_arg )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_managed_buffer_t;
    typedef SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* ptr_dbg_register_t;

    dim3 const* ptr_blocks = SIXTRL_NULLPTR;
    dim3 const* ptr_threads = SIXTRL_NULLPTR;

    if( ( NS(KernelConfig_get_arch_id)( conf ) == NS(ARCHITECTURE_CUDA) ) &&
        ( !NS(KernelConfig_needs_update)( conf ) ) )
    {
        ptr_blocks = NS(CudaKernelConfig_get_ptr_const_blocks)( conf );

        ptr_threads =
            NS(CudaKernelConfig_get_ptr_const_threads_per_block)( conf );
    }

    if( ( ptr_blocks  != SIXTRL_NULLPTR ) &&
        ( ptr_threads != SIXTRL_NULLPTR ) &&
        ( managed_buffer_begin != SIXTRL_NULLPTR ) &&
        ( slot_size > ( NS(buffer_size_t) )0u ) )
    {
        NS(ManagedBuffer_remap_cuda_debug)<<< *ptr_blocks, *ptr_threads >>>(
            reinterpret_cast< ptr_managed_buffer_t >( managed_buffer_begin ),
            slot_size,
            reinterpret_cast< ptr_dbg_register_t >( dbg_register_arg ) );

        ::cudaDeviceSynchronize();
    }
}

bool NS(Buffer_is_remapped_cuda_wrapper)(
    NS(cuda_arg_buffer_t) SIXTRL_RESTRICT managed_buffer_begin,
    NS(buffer_size_t) const slot_size,
    NS(cuda_arg_buffer_t) SIXTRL_RESTRICT ptr_debug_register,
    NS(arch_status_t)* SIXTRL_RESTRICT ptr_status )
{
    typedef NS(arch_status_t) status_t;
    typedef NS(arch_debugging_t) debug_register_t;
    typedef SIXTRL_DATAPTR_DEC debug_register_t* ptr_dbg_register_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_managed_buffer_t;

    status_t local_status = NS(ARCH_STATUS_GENERAL_FAILURE);

    bool is_remapped = false;

    if( ( ptr_debug_register != SIXTRL_NULLPTR ) &&
        ( managed_buffer_begin != SIXTRL_NULLPTR ) &&
        ( slot_size > ( NS(buffer_size_t) )0u ) )
    {
        debug_register_t local_dbg = NS(ARCH_DEBUGGING_GENERAL_FAILURE);

        debug_register_t const needs_remapping_true  = ( debug_register_t )1;
        debug_register_t const needs_remapping_false = ( debug_register_t )2;

        ::cudaError_t err = ::cudaMemcpy( ptr_debug_register, &local_dbg,
            sizeof( local_dbg ), ::cudaMemcpyHostToDevice );

        if( err == ::cudaSuccess )
        {
            local_status = NS(ARCH_STATUS_SUCCESS);

            ptr_dbg_register_t dbg_register =
                reinterpret_cast< ptr_dbg_register_t >( ptr_debug_register );

            NS(ManagedBuffer_needs_remapping_cuda)<<< 1, 1 >>>(
                reinterpret_cast< ptr_managed_buffer_t >( managed_buffer_begin ),
                    slot_size, needs_remapping_true, needs_remapping_false,
                        dbg_register );

            ::cudaDeviceSynchronize();

            err = ::cudaMemcpy( &local_dbg, dbg_register, sizeof( local_dbg ),
                                ::cudaMemcpyDeviceToHost );

            if( err != ::cudaSuccess )
            {
                local_status = NS(ARCH_STATUS_GENERAL_FAILURE);
            }

            SIXTRL_ASSERT( ( local_dbg == needs_remapping_false ) ||
                           ( local_dbg == needs_remapping_true  ) );

            is_remapped = ( local_dbg == needs_remapping_false );
        }

    }

    if( ptr_status != SIXTRL_NULLPTR) *ptr_status = local_status;

    return is_remapped;
}

/* end: sixtracklib/cuda/wrappers/controller_wrappers.cu */
