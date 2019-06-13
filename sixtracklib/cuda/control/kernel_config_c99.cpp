#include "sixtracklib/cuda/control/kernel_config.h"

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cuda_runtime_api.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/cuda/definitions.h"
#include "sixtracklib/cuda/controller.hpp"
#include "sixtracklib/cuda/control/kernel_config.hpp"
#include "sixtracklib/cuda/control/node_info.hpp"
#include "sixtracklib/cuda/controller.hpp"
#include "sixtracklib/cuda/controller.h"

namespace st = SIXTRL_CXX_NAMESPACE;

::NS(CudaKernelConfig)* NS(CudaKernelConfig_new)(
    NS(ctrl_size_t) const num_kernel_args,
    char const* SIXTRL_RESTRICT kernel_name )
{
    return new st::CudaKernelConfig( kernel_name, num_kernel_args );
}


::NS(CudaKernelConfig)* NS(CudaKernelConfig_new_detailed)(
    char const* SIXTRL_RESTRICT kernel_name,
    NS(ctrl_size_t) const num_kernel_args,
    NS(ctrl_size_t) const block_dimensions,
    NS(ctrl_size_t) const threads_per_block_dimensions,
    NS(ctrl_size_t) const shared_mem_per_block, 
    NS(ctrl_size_t) const max_block_size_limit, 
    NS(ctrl_size_t) const warp_size,
    char const* SIXTRL_RESTRICT config_str )
{
    using size_t = st::CudaKernelConfig::size_type;
    
    return new st::CudaKernelConfig( kernel_name, num_kernel_args, 
        block_dimensions, threads_per_block_dimensions, shared_mem_per_block, 
            max_block_size_limit, warp_size, config_str );
}

::NS(ctrl_size_t) NS(CudaKernelConfig_total_num_blocks)(
    const ::NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config )
{
    return ( kernel_config != nullptr )
        ? kernel_config->totalNumBlocks() : ::NS(ctrl_size_t){ 0 };
}

/* ------------------------------------------------------------------------- */

::NS(ctrl_size_t) NS(CudaKernelConfig_total_num_threads_per_block)(
    const ::NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config )
{
    return ( kernel_config != nullptr )
        ? kernel_config->totalNumThreadsPerBlock()
        : ::NS(ctrl_size_t){ 0 };
}

SIXTRL_EXTERN SIXTRL_HOST_FN ::NS(ctrl_size_t)
NS(CudaKernelConfig_total_num_threads)(
    const ::NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config )
{
    return( kernel_config != nullptr )
        ? kernel_config->totalNumThreads() : ::NS(ctrl_size_t){ 0 };
}

/* ------------------------------------------------------------------------- */

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
