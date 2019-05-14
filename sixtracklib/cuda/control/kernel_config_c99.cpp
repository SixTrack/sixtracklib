#include "sixtracklib/cuda/control/kernel_config.h"

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cuda_runtime_api.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/cuda/definitions.h"
#include "sixtracklib/cuda/control/kernel_config.hpp"

::dim3 const* NS(CudaKernelConfig_get_ptr_const_blocks)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT cuda_kernel_config )
{
    return ( cuda_kernel_config != nullptr )
        ? cuda_kernel_config->ptrBlocks() : nullptr;
}

::dim3 const* NS(CudaKernelConfig_get_ptr_const_threads_per_block)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT cuda_kernel_config )
{
    return ( cuda_kernel_config != nullptr )
        ? cuda_kernel_config->ptrThreadsPerBlock() : nullptr;
}

#endif /* C++, Host */

/* end: sixtracklib/cuda/control/kernel_config_c99.cpp */
