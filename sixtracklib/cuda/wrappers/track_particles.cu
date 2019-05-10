#include "sixtracklib/cuda/wrappers/track_particles.h"

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
#include "sixtracklib/cuda/kernels/track_particles.cuh"

NS(ctrl_status_t) NS(Track_particles_line_cuda_on_grid)(
    void* SIXTRL_RESTRICT particles_buffer_arg,
    void* SIXTRL_RESTRICT beam_elements_buffer_arg,
    NS(buffer_size_t) const line_begin_idx,
    NS(buffer_size_t) const line_end_idx,
    bool const finish_turn,
    NS(buffer_size_t) const num_blocks,
    NS(buffer_size_t) const num_threads_per_block )
{
    NS(ctrl_status_t) status = NS(CONTROLLER_STATUS_GENERAL_FAILURE);

    dim3 grid_dim;
    dim3 block_dim;

    grid_dim.x = num_blocks;
    grid_dim.y = 1;
    grid_dim.z = 1;

    block_dim.x = num_threads_per_block;
    block_dim.y = 1;
    block_dim.z = 1;

    status = NS(CONTROLLER_STATUS_SUCCESS);

    NS(Track_particles_line)<<< grid_dim, block_dim >>>(
        reinterpret_cast< unsigned char* >( particles_buffer_arg ),
        reinterpret_cast< unsigned char *>( beam_elements_buffer_arg ),
        line_begin_idx, line_end_idx, finish_turn );

    return status;
}

NS(ctrl_status_t) NS(Track_particles_line_cuda)(
    void* SIXTRL_RESTRICT particles_buffer_arg,
    void* SIXTRL_RESTRICT beam_elements_buffer_arg,
    NS(buffer_size_t) const line_begin_idx,
    NS(buffer_size_t) const line_end_idx,
    bool const finish_turn )
{
    return NS(Track_particles_line_cuda_on_grid)(
        particles_buffer_arg, beam_elements_buffer_arg, line_begin_idx,
            line_end_idx, finish_turn, 64, 64 );
}

