#include "sixtracklib/cuda/wrappers/context_operations.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/buffer.h"

#include "sixtracklib/cuda/kernels/managed_buffer_remap.cuh"


NS(controller_status_t) NS(CudaContext_perform_send)(
    void* SIXTRL_RESTRICT destination,
    const void *const SIXTRL_RESTRICT source_begin,
    NS(controller_size_t) const source_length )
{
    typedef NS(controller_status_t) status_t;
    typedef NS(controller_size_t)   ctx_size_t;

    status_t status = NS(CONTROLLER_STATUS_GENERAL_FAILURE);

    if( ( destination != SIXTRL_NULLPTR ) &&
        ( source_begin != SIXTRL_NULLPTR ) &&
        ( source_length > ( ctx_size_t )0u ) )
    {
        cudaError_t const ret = cudaMemcpy(
            destination, source_begin, source_length, cudaMemcpyHostToDevice );

        if( ret == cudaSuccess )
        {
            status = NS(CONTROLLER_STATUS_SUCCESS);
        }
    }

    return status;
}

NS(controller_status_t) NS(CudaContext_perform_receive)(
    void* SIXTRL_RESTRICT destination,
    NS(controller_size_t) const destination_capacity,
    void* SIXTRL_RESTRICT source_begin,
    NS(controller_size_t) const source_length )
{
    typedef NS(controller_status_t) status_t;
    typedef NS(controller_size_t)   ctx_size_t;

    status_t status = NS(CONTROLLER_STATUS_GENERAL_FAILURE);

    if( ( source_begin  != SIXTRL_NULLPTR ) &&
        ( source_length > ( ctx_size_t )0u ) &&
        ( destination != SIXTRL_NULLPTR ) &&
        ( destination_capacity >= source_length ) )
    {
        cudaError_t const ret = cudaMemcpy(
            destination, source_begin, source_length, cudaMemcpyDeviceToHost );

        if( ret == cudaSuccess )
        {
            status = NS(CONTROLLER_STATUS_SUCCESS);
        }
    }

    return status;
}

NS(controller_status_t)
NS(CudaContext_perform_remap_send_cobject_buffer_on_grid)(
    void* SIXTRL_RESTRICT arg_buffer,
    NS(controller_size_t) const slot_size,
    NS(controller_size_t) const grid_num_blocks,
    NS(controller_size_t) const threads_per_block )
{
    typedef NS(controller_status_t) status_t;
    typedef NS(controller_size_t)   ctx_size_t;
    typedef NS(buffer_size_t)    buf_size_t;

    status_t status = NS(CONTROLLER_STATUS_GENERAL_FAILURE);

    if( ( arg_buffer != SIXTRL_NULLPTR ) && ( slot_size > ( buf_size_t )0u ) )
    {
        dim3 grid_dim;
        dim3 block_dim;

        grid_dim.x = grid_num_blocks;
        grid_dim.y = 1;
        grid_dim.z = 1;

        block_dim.x = threads_per_block;
        block_dim.y = 1;
        block_dim.z = 1;

        NS(ManagedBuffer_remap_cuda)<<< grid_dim, block_dim >>>(
            ( unsigned char* )arg_buffer, slot_size );

        status = ::NS(CONTROLLER_STATUS_SUCCESS);
    }

    return status;
}

NS(controller_status_t) NS(CudaContext_perform_remap_send_cobject_buffer)(
    void* SIXTRL_RESTRICT arg_buffer,
    NS(buffer_size_t) const slot_size )
{
    return NS(CudaContext_perform_remap_send_cobject_buffer_on_grid)(
        arg_buffer, slot_size, 1u, 1u );
}

/* end: sixtracklib/cuda/wrappers/context_operations.cu */
