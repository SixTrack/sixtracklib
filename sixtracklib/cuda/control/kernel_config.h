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
    #include "sixtracklib/cuda/controller.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++ */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaKernelConfig)*
NS(CudaKernelConfig_new)(
    NS(CudaController)* SIXTRL_RESTRICT controller,
    NS(ctrl_size_t) const num_kernel_args,
    char const* SIXTRL_RESTRICT kernel_name );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaKernelConfig)*
NS(CudaKernelConfig_new_detailed)(
    NS(CudaController)* SIXTRL_RESTRICT controller,
    NS(ctrl_size_t) const num_kernel_args,
    NS(ctrl_size_t) const work_items_dim,
    NS(ctrl_size_t) const* SIXTRL_RESTRICT num_work_items,
    NS(ctrl_size_t) const work_groups_dim,
    NS(ctrl_size_t) const* SIXTRL_RESTRICT workg_group_sizes,
    char const* SIXTRL_RESTRICT kernel_name );

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

#endif /* SIXTRACKLIB_CUDA_CONTROL_KERNEL_CONFIG_C9999_H__ */

/* end: sixtracklib/cuda/control/kernel_config.h */
