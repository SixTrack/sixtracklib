#ifndef SIXTRACKLIB_CUDA_KERNELS_ELEM_BY_ELEM_ASSIGN_OUT_BUFFER_CUDA_CUH__
#define SIXTRACKLIB_CUDA_KERNELS_ELEM_BY_ELEM_ASSIGN_OUT_BUFFER_CUDA_CUH__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/output/output_buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

__global__ void NS(ElemByElemConfig_assign_out_buffer_from_offset_cuda)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const out_buffer_offset_index,
    NS(buffer_size_t) const slot_size );

__global__ void NS(ElemByElemConfig_assign_out_buffer_from_offset_cuda_debug)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const out_buffer_offset_index,
    NS(buffer_size_t) const slot_size,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_dbg_register);

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_CUDA_KERNELS_ELEM_BY_ELEM_ASSIGN_OUT_BUFFER_CUDA_CUH__ */

/* end sixtracklib/cuda/kernels/elem_by_elem_assign_out_buffer.cuh */
