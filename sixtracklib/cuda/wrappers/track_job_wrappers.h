#ifndef SIXTRACKLIB_CUDA_WRAPPERS_TRACK_JOB_WRAPPERS_C99_H__
#define SIXTRACKLIB_CUDA_WRAPPERS_TRACK_JOB_WRAPPERS_C99_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/particles/definitions.h"
    #include "sixtracklib/common/track/definitions.h"

    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/argument.h"
    #include "sixtracklib/cuda/control/kernel_config.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN
void NS(Track_particles_until_turn_cuda_wrapper)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT particles_arg,
    NS(buffer_size_t) const pset_index,
    NS(CudaArgument)* SIXTRL_RESTRICT beam_elements_arg,
    NS(buffer_size_t) const until_turn,
    NS(CudaArgument)* SIXTRL_RESTRICT debug_register_arg );

SIXTRL_EXTERN SIXTRL_HOST_FN
void NS(Track_particles_elem_by_elem_until_turn_cuda_wrapper)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT particles_arg,
    NS(buffer_size_t) const pset_index,
    NS(CudaArgument)* SIXTRL_RESTRICT beam_elements_arg,
    NS(CudaArgument)* SIXTRL_RESTRICT config_buffer_arg,
    NS(buffer_size_t) const elem_by_elem_config_index,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    NS(CudaArgument)* SIXTRL_RESTRICT debug_register_arg );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Track_particles_line_cuda_wrapper)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT particles_arg,
    NS(buffer_size_t) const pset_index,
    NS(CudaArgument)* SIXTRL_RESTRICT beam_elements_arg,
    NS(buffer_size_t) const be_begin_idx, NS(buffer_size_t) const be_end_idx,
    bool const finish_turn,
    NS(CudaArgument)* SIXTRL_RESTRICT debug_register_arg );

SIXTRL_EXTERN SIXTRL_HOST_FN
void NS(BeamMonitor_assign_out_buffer_from_offset_cuda_wrapper)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT beam_elements_arg,
    NS(CudaArgument)* SIXTRL_RESTRICT output_buffer_arg,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t) const out_buffer_offset_index,
    NS(CudaArgument)* SIXTRL_RESTRICT debug_register_arg );

SIXTRL_EXTERN SIXTRL_HOST_FN
void NS(ElemByElemConfig_assign_out_buffer_from_offset_cuda_wrapper)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT config_buffer_arg,
    NS(buffer_size_t) const elem_by_elem_config_index,
    NS(CudaArgument)* SIXTRL_RESTRICT output_buffer_arg,
    NS(buffer_size_t) const out_buffer_offset_index,
    NS(CudaArgument)* SIXTRL_RESTRICT debug_register_arg );

SIXTRL_EXTERN SIXTRL_HOST_FN
void NS(Particles_buffer_store_all_addresses_cuda_wrapper)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT particles_addresses_arg,
    NS(CudaArgument)* SIXTRL_RESTRICT particles_arg,
    NS(CudaArgument)* SIXTRL_RESTRICT debug_flag_arg );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_CUDA_WRAPPERS_TRACK_JOB_WRAPPERS_C99_H__ */

/* end: sixtracklib/cuda/wrappers/track_job_wrappers.h */
