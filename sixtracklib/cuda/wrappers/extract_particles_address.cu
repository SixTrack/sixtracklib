#include "sixtracklib/cuda/wrappers/extract_particles_address.h"

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
#include "sixtracklib/cuda/kernels/extract_particles_address.cuh"

NS(controller_status_t) NS(Particles_extract_addresses_cuda_on_grid)(
    void* SIXTRL_RESTRICT addr_arg_buffer,
    void* SIXTRL_RESTRICT pb_arg_buffer,
    NS(buffer_size_t) const num_blocks,
    NS(buffer_size_t) const num_threads_per_block )
{
    NS(controller_status_t) status = -1;

    dim3 grid_dim;
    dim3 block_dim;

    grid_dim.x = num_blocks;
    grid_dim.y = 1;
    grid_dim.z = 1;

    block_dim.x = num_threads_per_block;
    block_dim.y = 1;
    block_dim.z = 1;

    status = ( NS(controller_status_t) )0u;

    NS(Particles_extract_addresses)<<< grid_dim, block_dim >>>(
        addr_arg_buffer, pb_arg_buffer, 8u );

    return status;
}

NS(controller_status_t) NS(Particles_extract_addresses_cuda)(
    void* SIXTRL_RESTRICT paddr_arg,
    void* SIXTRL_RESTRICT pbuffer_arg )
{
    return NS(Particles_extract_addresses_cuda_on_grid)(
        paddr_arg, pbuffer_arg, 1, 1 );
}


