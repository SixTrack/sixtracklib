#ifndef SIXTRACKLIB_CUDA_CONTROL_KERNEL_CONFIG_C9999_H__
#define SIXTRACKLIB_CUDA_CONTROL_KERNEL_CONFIG_C9999_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/control/kernel_config.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cuda_runtime_api.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/kernel_config_base.h"
    #include "sixtracklib/cuda/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++ */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_FN dim3 const*
NS(CudaKernelConfig_get_ptr_const_blocks)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT cuda_kernel_config );

SIXTRL_EXTERN SIXTRL_FN dim3 const*
NS(CudaKernelConfig_get_ptr_const_threads_per_block)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT cuda_kernel_config );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_CONTROL_KERNEL_CONFIG_C9999_H__ */

/* end: sixtracklib/cuda/control/kernel_config.h */
