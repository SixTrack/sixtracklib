#include "sixtracklib/cuda/wrappers/buffer_remap.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if !defined( NDEBUG )
    #include <stdio.h>
#endif /* !defined( NDEBUG ) */

#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include <cuda.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/cuda/kernels/managed_buffer_remap.cuh"

int NS(Buffer_remap_on_cuda_grid)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const num_blocks,
    NS(buffer_size_t) const num_threads_per_block )
{
    using managed_buffer_t = SIXTRL_BUFFER_DATAPTR_DEC unsigned char*;
    uintptr_t const begin_addr = ::NS(Buffer_get_data_begin_addr)( buffer );

    return NS(ManagedBuffer_remap_cuda)( reinterpret_cast< managed_buffer_t >(
        begin_addr ), ::NS(Buffer_get_slot_size)( buffer ),
            num_blocks, num_threads_per_block );
}

int NS(Buffer_remap_debug_on_cuda_grid)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC uint32_t* SIXTRL_RESTRICT ptr_success_flag,
    NS(buffer_size_t) const num_blocks,
    NS(buffer_size_t) const num_threads_per_block )
{
    using managed_buffer_t = SIXTRL_BUFFER_DATAPTR_DEC unsigned char*;
    uintptr_t const begin_addr = ::NS(Buffer_get_data_begin_addr)( buffer );

    return NS(ManagedBuffer_remap_cuda)( reinterpret_cast< managed_buffer_t >(
        begin_addr ), ptr_success_flag, ::NS(Buffer_get_slot_size)( buffer ),
            num_blocks, num_threads_per_block );
}

/* ------------------------------------------------------------------------- */

int NS(ManagedBuffer_remap_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const slot_size,
    NS(buffer_size_t) const num_blocks,
    NS(buffer_size_t) const num_threads_per_block )
{

}

int NS(ManagedBuffer_remap_debug_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const slot_size,
    SIXTRL_ARGPTR_DEC uint32_t* SIXTRL_RESTRICT ptr_success_flag,
    NS(buffer_size_t) const num_blocks,
    NS(buffer_size_t) const num_threads_per_block )
{

}

/* end: sixtracklib/cuda/wrappers/buffer_remap.cu */
