#ifndef SIXTRACKLIB_CUDA_WRAPPERS_CONTROLLER_WRAPPERS_C99_H__
#define SIXTRACKLIB_CUDA_WRAPPERS_CONTROLLER_WRAPPERS_C99_H__

#if !defined( SIXTRL_NO_INCLUDE )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/control/definitions.h"

    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/argument.h"
    #include "sixtracklib/cuda/control/kernel_config.h"
#endif /* !defined( SIXTRL_NO_INCLUDE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_EXTERN SIXTRL_HOST_FN
NS(ctrl_status_t) NS(CudaController_remap_cobjects_buffer_wrapper)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT buffer_arg,
    NS(buffer_size_t) const slot_size,
    NS(CudaArgument)* SIXTRL_RESTRICT debug_flag_arg );

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_CUDA_WRAPPERS_CONTROLLER_WRAPPERS_C99_H__ */

/* end: sixtracklib/cuda/wrappers/controller_wrappers.h */