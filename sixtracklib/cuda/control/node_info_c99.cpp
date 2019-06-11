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

::NS(CudaNodeInfo)* NS(CudaNodeInfo_new)( 
    ::NS(cuda_dev_index_t) const cuda_device_index )
{
    ::NS(CudaNodeInfo)* ptr_cuda_node_info = nullptr;
    
    if( cuda_device_index >= ::NS(cuda_dev_index_t){ 0 } )
    {
        ::cudaDeviceProp device_properties;
        ::cudaError_t err = ::cudaGetDeviceProperties( 
            &device_properties, cuda_device_index );
    
        if( err == ::cudaSuccess )
        {
            ptr_cuda_node_info = new st::CudaNodeInfo( 
                cuda_device_index, device_properties );
        }
    }
    
    return ptr_cuda_node_info; 
}

::NS(CudaNodeInfo)* NS(CudaNodeInfo_new_detailed)( 
    ::NS(cuda_dev_index_t) const cuda_device_index, 
    ::NS(node_platform_id_t) const platform_id, 
    ::NS(node_device_id_t) const device_id, 
    ::NS(node_index_t) const node_index, 
    bool const is_default_node, bool const is_selected_node )
{
    ::NS(CudaNodeInfo)* ptr_cuda_node_info = nullptr;
    
    if( cuda_device_index >= ::NS(cuda_dev_index_t){ 0 } )
    {
        ::cudaDeviceProp device_properties;
        ::cudaError_t err = ::cudaGetDeviceProperties( 
            &device_properties, cuda_device_index );
    
        if( err == ::cudaSuccess )
        {
            ptr_cuda_node_info = new st::CudaNodeInfo( 
                cuda_device_index, device_properties, platform_id, device_id,
                    node_index, is_default_node, is_selected_node );
        }
    }
    
    return ptr_cuda_node_info; 
}

/* ------------------------------------------------------------------------- */

::NS(cuda_dev_index_t) NS(CudaNodeInfo_get_cuda_device_index)(
    const ::NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return ( ( node_info != nullptr ) && ( node_info->hasCudaDeviceIndex() ) )
        ? node_info->cudaDeviceIndex() : ::NS(cuda_dev_index_t){ -1 };
}

char const* NS(CudaNodeInfo_get_pci_bus_id_str)(
    const ::NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return ( ( node_info != nullptr ) && ( node_info->hasPciBusId() ) )
        ? node_info->ptrPciBusIdStr() : nullptr;
}

/* ------------------------------------------------------------------------- */

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

/* ------------------------------------------------------------------------- */

::NS(CudaNodeInfo) const* NS(NodeInfo_as_const_cuda_node_info)(
    ::NS(NodeInfoBase) const* SIXTRL_RESTRICT node_info_base )
{
    return st::NodeInfo_as_cuda_node_info( node_info_base );
}

::NS(CudaNodeInfo)* NS(NodeInfo_as_cuda_node_info)( 
    ::NS(NodeInfoBase)* SIXTRL_RESTRICT node_info_base )
{
    return st::NodeInfo_as_cuda_node_info( node_info_base );
}

#endif /* C++, Host */

/* end: sixtracklib/cuda/control/node_info_c99.cpp */
