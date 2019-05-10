#ifndef SIXTRACKLIB_CUDA_WRAPPERS_TRACK_JOB_WRAPPERS_C99_H__
#define SIXTRACKLIB_CUDA_WRAPPERS_TRACK_JOB_WRAPPERS_C99_H__

#if !defined( SIXTRL_NO_INCLUDE )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/particles/definitions.h"
    #include "sixtracklib/common/track/definitions.h"

    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/argument.h"
    #include "sixtracklib/cuda/control/kernel_config.h"
#endif /* !defined( SIXTRL_NO_INCLUDE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t) NS(CudaTrack_track_until_turn)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT particles_arg,
    NS(CudaArgument)* SIXTRL_RESTRICT beam_elements_arg,
    NS(buffer_size_t) const until_turn,
    NS(CudaArgument)* SIXTRL_RESTRICT debug_flag_arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(CudaTrack_track_elem_by_elem_until_turn)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT particles_arg,
    NS(CudaArgument)* SIXTRL_RESTRICT beam_elements_arg,
    NS(CudaArgument)* SIXTRL_RESTRICT output_arg,
    NS(CudaArgument)* SIXTRL_RESTRICT elem_by_elem_config_arg,
    NS(buffer_size_t) const out_buffer_offset_index,
    NS(buffer_size_t) const until_turn,
    NS(CudaArgument)* SIXTRL_RESTRICT debug_flag_arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t) NS(CudaTrack_track_line)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT particles_arg,
    NS(CudaArgument)* SIXTRL_RESTRICT beam_elements_arg,
    NS(buffer_size_t) const be_begin_idx,
    NS(buffer_size_t) const be_end_idx,
    bool const finish_turn,
    NS(CudaArgument)* SIXTRL_RESTRICT debug_flag_arg );

SIXTRL_EXTERN SIXTRL_HOST_FN
NS(ctrl_status_t) NS(Cuda_assign_output_to_be_monitors)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT beam_elements_arg,
    NS(CudaArgument)* SIXTRL_RESTRICT output_arg,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t) const out_buffer_offset_index,
    NS(CudaArgument)* SIXTRL_RESTRICT debug_flag_arg );

SIXTRL_EXTERN SIXTRL_HOST_FN
NS(ctrl_status_t) NS(Cuda_fetch_particle_addresses)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT particles_addresses_arg,
    NS(CudaArgument)* SIXTRL_RESTRICT particles_arg,
    NS(CudaArgument)* SIXTRL_RESTRICT debug_flag_arg );

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_CUDA_WRAPPERS_TRACK_JOB_WRAPPERS_C99_H__ */

/* end: sixtracklib/cuda/wrappers/track_job_wrappers.h */