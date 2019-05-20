#ifndef SIXTRACKLIB_CUDA_CONTROL_KERNEL_CONFIG_C99_H__
#define SIXTRACKLIB_CUDA_CONTROL_KERNEL_CONFIG_C99_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/controller.hpp"
    #include "sixtracklib/cuda/control/kernel_config.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cuda_runtime_api.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/kernel_config_base.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++ */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaKernelConfig)* NS(CudaKernelConfig_new)(
    NS(ctrl_size_t) const num_kernel_args,
    char const* SIXTRL_RESTRICT kernel_name );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaKernelConfig)* 
NS(CudaKernelConfig_new_detailed)(
    char const* SIXTRL_RESTRICT kernel_name, 
    NS(ctrl_size_t) const num_kernel_args,
    NS(ctrl_size_t) const block_dimension,
    NS(ctrl_size_t) const threads_per_block_dimension,
    NS(ctrl_size_t) const shared_mem_per_block, 
    NS(ctrl_size_t) const max_block_size_limit, 
    NS(ctrl_size_t) const warp_size,
    char const* SIXTRL_RESTRICT config_str );
    
/* ------------------------------------------------------------------------ */
    
SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_size_t)
NS(CudaKernelConfig_total_num_blocks)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_size_t)
NS(CudaKernelConfig_total_num_threads_per_block)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_size_t)
NS(CudaKernelConfig_total_num_threads)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config );

/* ------------------------------------------------------------------------ */

SIXTRL_EXTERN SIXTRL_HOST_FN dim3 const*
NS(CudaKernelConfig_get_ptr_const_blocks)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT cuda_kernel_config );

SIXTRL_EXTERN SIXTRL_HOST_FN dim3 const*
NS(CudaKernelConfig_get_ptr_const_threads_per_block)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT cuda_kernel_config );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_CONTROL_KERNEL_CONFIG_C99_H__ */

/* end: sixtracklib/cuda/control/kernel_config.h */
