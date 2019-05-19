#include "sixtracklib/cuda/control/node_info.h"

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/control/node_info.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cuda_runtime_api.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/node_info.h"
    #include "sixtracklib/cuda/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

namespace st = SIXTRL_CXX_NAMESPACE;

::NS(arch_size_t) NS(CudaNodeInfo_get_warp_size)(
    const ::NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return ( node_info != nullptr ) ? node_info->warpSize()
        : st::CudaNodeInfo::DEFAULT_WARP_SIZE;
}

::NS(arch_size_t) NS(CudaNodeInfo_get_compute_capability)(
    const ::NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return ( node_info != nullptr )
        ? node_info->computeCapability() : ::NS(arch_size_t){ 0 };
}

::NS(arch_size_t) NS(CudaNodeInfo_get_num_multiprocessors)(
    const ::NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return ( node_info )
        ? node_info->numMultiprocessors() : ::NS(arch_size_t){ 0 };
}

::NS(arch_size_t) NS(CudaNodeInfo_get_max_threads_per_block)(
    const ::NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return ( node_info != nullptr )
        ? node_info->maxThreadsPerBlock() : ::NS(arch_size_t){ 0 };
}

::NS(arch_size_t) NS(CudaNodeInfo_get_max_threads_per_multiprocessor)(
    const ::NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return ( node_info != nullptr )
        ? node_info->maxThreadsPerMultiprocessor() : ::NS(arch_size_t){ 0 };
}

#endif /* C++, Host */

/* end: sixtracklib/cuda/control/node_info_c99.cpp */
