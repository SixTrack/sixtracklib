#ifndef SIXTRACKLIB_CUDA_CUDA_ENV_H__
#define SIXTRACKLIB_CUDA_CUDA_ENV_H__

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/blocks.h"

SIXTRL_HOST_FN bool NS(Track_particles_on_cuda)(
    int const num_of_blocks, 
    int const num_threads_per_block,
    SIXTRL_UINT64_T const num_of_turns,
    NS(Blocks)* SIXTRL_RESTRICT particles_buffer,
    NS(Blocks)* SIXTRL_RESTRICT beam_elements,
    NS(Blocks)* SIXTRL_RESTRICT elem_by_elem_buffer );

#endif /* SIXTRACKLIB_CUDA_CUDA_ENV_H__ */

/* end: sixtracklib/sixtracklib/cuda/cuda_env.h */
