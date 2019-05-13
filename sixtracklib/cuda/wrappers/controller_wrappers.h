#ifndef SIXTRACKLIB_CUDA_WRAPPERS_CONTROLLER_WRAPPERS_C99_H__
#define SIXTRACKLIB_CUDA_WRAPPERS_CONTROLLER_WRAPPERS_C99_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/control/definitions.h"

    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/control/kernel_config.h"
    #include "sixtracklib/cuda/argument.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Buffer_remap_cuda_wrapper)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT buffer_arg );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Buffer_remap_cuda_debug_wrapper)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT buffer_arg,
    NS(cuda_arg_buffer_t) SIXTRL_RESTRICT ptr_debug_register );

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_CUDA_WRAPPERS_CONTROLLER_WRAPPERS_C99_H__ */

/* end: sixtracklib/cuda/wrappers/controller_wrappers.h */