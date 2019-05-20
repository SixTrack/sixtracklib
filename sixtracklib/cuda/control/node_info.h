#ifndef SIXTRACKLIB_CUDA_CONTROL_NODE_INFO_C99_H__
#define SIXTRACKLIB_CUDA_CONTROL_NODE_INFO_C99_H__

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

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++ */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_size_t) NS(CudaNodeInfo_get_warp_size)(
    const NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_size_t)
NS(CudaNodeInfo_get_compute_capability)(
    const NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_size_t)
NS(CudaNodeInfo_get_num_multiprocessors)(
    const NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_size_t)
NS(CudaNodeInfo_get_max_threads_per_block)(
    const NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_size_t)
NS(CudaNodeInfo_get_max_threads_per_multiprocessor)(
    const NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info );

/* ------------------------------------------------------------------------ */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaNodeInfo) const*
NS(NodeInfo_as_const_cuda_node_info)(
    NS(NodeInfoBase) const* SIXTRL_RESTRICT node_info_base );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaNodeInfo)*
NS(NodeInfo_as_cuda_node_info)( 
    NS(NodeInfoBase)* SIXTRL_RESTRICT node_info_base );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++ */

#endif /* SIXTRACKLIB_CUDA_CONTROL_NODE_INFO_C99_H__ */
/* end: sixtracklib/cuda/control/node_info.h */
