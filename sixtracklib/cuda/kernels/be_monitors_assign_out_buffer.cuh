#ifndef SIXTRACKLIB_CUDA_KERNELS_BE_MONITORS_ASSIGN_OUT_BUFFER_CUDA_CUH__
#define SIXTRACKLIB_CUDA_KERNELS_BE_MONITORS_ASSIGN_OUT_BUFFER_CUDA_CUH__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/particles/definitions.h"
    #include "sixtracklib/cuda/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

__global__ void NS(BeamMonitor_assign_out_buffer_from_offset_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elem_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT output_buffer,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t) const out_buffer_offset_index,
    NS(buffer_size_t) const slot_size );

__global__ void NS(BeamMonitor_assign_out_buffer_from_offset_cuda_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elem_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT output_buffer,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t) const out_buffer_offset_index,
    NS(buffer_size_t) const slot_size,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_dbg_register);

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_CUDA_KERNELS_BE_MONITORS_ASSIGN_OUT_BUFFER_CUDA_CUH__ */

/* end sixtracklib/cuda/kernels/be_monitors_assign_out_buffer.cuh */
