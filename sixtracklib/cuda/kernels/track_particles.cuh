#ifndef SIXTRACKLIB_CUDA_KERNELS_TRACK_PARTICLES_KERNELS_CUH__
#define SIXTRACKLIB_CUDA_KERNELS_TRACK_PARTICLES_KERNELS_CUH__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/particles/definitions.h"
    #include "sixtracklib/common/track/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/cuda/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

__global__ void NS(Track_particles_until_turn_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer_begin,
    NS(buffer_size_t) const pset_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT be_buffer_begin,
    NS(buffer_size_t) const until_turn,
    NS(buffer_size_t) const slot_size );

__global__ void NS(Track_particles_until_turn_cuda_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer_begin,
    NS(buffer_size_t) const pset_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT be_buffer_begin,
    NS(buffer_size_t) const until_turn,
    NS(buffer_size_t) const slot_size,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_dbg_register);

/* ------------------------------------------------------------------------- */

__global__ void NS(Track_track_elem_by_elem_until_turn_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer_begin,
    NS(buffer_size_t) const pset_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT be_buffer_begin,
    SIXTRL_DATAPTR_DEC const NS(ElemByElemConfig) *const SIXTRL_RESTRICT conf,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    NS(buffer_size_t) const slot_size );

__global__ void NS(Track_track_elem_by_elem_until_turn_cuda_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer_begin,
    NS(buffer_size_t) const pset_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT be_buffer_begin,
    SIXTRL_DATAPTR_DEC const NS(ElemByElemConfig) *const SIXTRL_RESTRICT conf,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    NS(buffer_size_t) const slot_size,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_dbg_register);

/* ------------------------------------------------------------------------- */

__global__ void NS(Track_particles_line_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer_begin,
    NS(buffer_size_t) const pset_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT be_buffer_begin,
    NS(buffer_size_t) const belem_begin_id,
    NS(buffer_size_t) const belem_end_id,
    bool const finish_turn,
    NS(buffer_size_t) const slot_size );

__global__ void NS(Track_particles_line_cuda_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer_begin,
    NS(buffer_size_t) const pset_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT be_buffer_begin,
    NS(buffer_size_t) const belem_begin_id,
    NS(buffer_size_t) const belem_end_id,
    bool const finish_turn, NS(buffer_size_t) const slot_size,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_dbg_register);

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_CUDA_KERNELS_TRACK_PARTICLES_KERNELS_CUH__ */

/* end sixtracklib/cuda/kernels/track_particles_kernels.cuh */
