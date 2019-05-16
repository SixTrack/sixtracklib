#include "sixtracklib/cuda/control/kernel_config.h"

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cuda_runtime_api.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/cuda/definitions.h"
#include "sixtracklib/cuda/controller.hpp"
#include "sixtracklib/cuda/control/kernel_config.hpp"
#include "sixtracklib/cuda/control/node_info.hpp"

NS(CudaKernelConfig)* NS(CudaKernelConfig_new)(
    NS(CudaController)* SIXTRL_RESTRICT controller,
    NS(ctrl_size_t) const num_kernel_args,
    char const* SIXTRL_RESTRICT kernel_name )
{
    using controller_t = SIXTRL_CXX_NAMESPACE::CudaController;
    using size_t  = controller_t::size_type;

    return NS(CudaKernelConfig_new_detailed)(
        controller, num_kernel_args, size_t{ 1 }, nullptr,
            size_t{ 1 }, nullptr, kernel_name );
}


::NS(CudaKernelConfig)* NS(CudaKernelConfig_new_detailed)(
    NS(CudaController)* SIXTRL_RESTRICT controller,
    NS(ctrl_size_t) const num_kernel_args,
    NS(ctrl_size_t) const work_items_dim,
    NS(ctrl_size_t) const* SIXTRL_RESTRICT num_work_items,
    NS(ctrl_size_t) const work_groups_dim,
    NS(ctrl_size_t) const* SIXTRL_RESTRICT work_group_sizes,
    char const* SIXTRL_RESTRICT kernel_name )
{
    using controller_t        = SIXTRL_CXX_NAMESPACE::CudaController;
    using node_info_t         = controller_t::node_info_t;
    using node_index_t        = controller_t::node_index_t;
    using kernel_id_t         = controller_t::kernel_id_t;
    using kernel_config_t     = SIXTRL_CXX_NAMESPACE::CudaKernelConfig;
    using size_t              = kernel_config_t::size_type;

    node_index_t selected_node_index = node_info_t::UNDEFINED_INDEX;
    node_info_t ptr_selected_node_info = nullptr;

    kernel_config_t ptr_kernel_config  = nullptr;

    if( ( controller != nullptr ) && ( controller->hasSelectedNode() ) )
    {
        selected_node_index = controller->selectedNodeIndex();

        if( selected_node_index != controller_t::UNDEFINED_INDEX )
        {
            ptr_selected_node_info =
                controller->ptrNodeInfo( selected_node_index );
        }
    }

    size_t const warp_size = ( ptr_selected_node_info != nullptr )
        ? ptr_selected_node_info->warpSize() : node_info_t::DEFAULT_WARP_SIZE;

    kernel_config_t kernel_config( kernel_name, num_kernel_args,
        work_items_dim, work_groups_dim, size_t{ 0 }, size_t{ 0 }, warp_size );

    success = true;

    if( num_work_items != nullptr )
    {
        success &= kernel_config.setNumWorkItems(
            work_items_dim, num_work_items );
    }

    if( warp_size > size_t{ 0 } )
    {
        size_t multiples[] = { warp_size, warp_size, warp_size };

        success &= kernel_config.setPreferredWorkGroupMultiple(
            work_group_dim, &multiples );
    }

    if( work_group_sizes != nullptr )
    {
        success &= kernel_config.setWorkGroupSizes(
            work_group_dim, work_group_sizes );
    }

    if( success )
    {
        success &= kernel_config.update();
    }

    if( controller != nullptr )
    {
        kernel_id_t const kernel_id = controller->addCudaKernelConfig(
            kernel_config );

        ptr_kernel_config = controller->ptrKernelConfig( kernel_id );
    }

    return ptr_kernel_config;
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
