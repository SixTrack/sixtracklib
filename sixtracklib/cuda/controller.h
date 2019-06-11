#ifndef SIXTRACKLIB_CUDA_CONTROLLER_H__
#define SIXTRACKLIB_CUDA_CONTROLLER_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/controller.h"
    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/controller.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, Host */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaController)*
NS(CudaController_create)( void );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaController)*
NS(CudaController_new)( const char *const SIXTRL_RESTRICT config_str );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaController)*
NS(CudaController_new_from_node_id)( 
    const NS(NodeId) *const SIXTRL_RESTRICT node_id, 
    const char *const SIXTRL_RESTRICT config_str );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaController)*
NS(CudaController_new_from_node_index)( 
    const NS(node_index_t) *const SIXTRL_RESTRICT node_id, 
    const char *const SIXTRL_RESTRICT config_str );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaController)*
NS(CudaController_new_from_platform_id_and_device_id)(
    NS(node_platform_id_t) const platform_id, 
    NS(node_device_id_t) const device_id,  
    const char *const SIXTRL_RESTRICT config_str );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaController)*
NS(CudaController_new_from_cuda_device_index)(
    int const cuda_device_index,
    const char *const SIXTRL_RESTRICT config_str );

/* ------------------------------------------------------------------------ */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(CudaController_select_node_by_cuda_device_index)(
    NS(CudaController)* SIXTRL_RESTRICT ctrl, int cuda_device_index );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(CudaController_select_node_by_cuda_pci_bus_id)(
    NS(CudaController)* SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT cuda_pci_bus_id );

/* ------------------------------------------------------------------------ */

 SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaNodeInfo) const*
 NS(CudaController_get_ptr_node_info_by_index)(
     const NS(CudaController) *const SIXTRL_RESTRICT ctrl,
     NS(ctrl_size_t) const index );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaNodeInfo) const*
 NS(CudaController_get_ptr_node_info_by_platform_id_and_device_id)(
     const NS(CudaController) *const SIXTRL_RESTRICT ctrl,
     NS(node_platform_id_t) const platform_idx,
     NS(node_device_id_t) const device_idx );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaNodeInfo) const*
 NS(CudaController_get_ptr_node_info_by_node_id)(
     const NS(CudaController) *const SIXTRL_RESTRICT ctrl,
     const NS(NodeId) *const SIXTRL_RESTRICT node_id );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaNodeInfo) const*
NS(CudaController_get_ptr_node_info)(
     const NS(CudaController) *const SIXTRL_RESTRICT ctrl,
     char const* SIXTRL_RESTRICT node_id_str );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaNodeInfo) const*
 NS(CudaController_get_ptr_node_info_by_cuda_dev_index)(
     const NS(CudaController) *const SIXTRL_RESTRICT ctrl,
     int const cuda_device_index );
 
 SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaNodeInfo) const*
 NS(CudaController_get_ptr_node_info_by_pci_bus_id)(
     const NS(CudaController) *const SIXTRL_RESTRICT ctrl, 
     char const* SIXTRL_RESTRICT cuda_pci_bus_id );

/* ------------------------------------------------------------------------ */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_kernel_id_t)
NS(CudaController_add_kernel_config)(
    NS(CudaController)* SIXTRL_RESTRICT ctrl,
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT cuda_kernel_config );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_kernel_id_t)
NS(CudaController_add_kernel_config_detailed)(
    NS(CudaController)* SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT kernel_name,
    NS(ctrl_size_t) const num_arguments,
    NS(ctrl_size_t) const grid_dim,
    NS(ctrl_size_t) const shared_mem_per_block,
    NS(ctrl_size_t) const max_blocks_limit,
    char const* SIXTRL_RESTRICT config_str );

/* ------------------------------------------------------------------------ */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaKernelConfig)*
NS(CudaController_get_ptr_kernel_config)(
    NS(CudaController)* SIXTRL_RESTRICT ctrl,
    NS(ctrl_kernel_id_t) const kernel_id );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaKernelConfig)*
NS(CudaController_get_ptr_kernel_config_by_kernel_name)(
    NS(CudaController)* SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT kernel_name );

/* ------------------------------------------------------------------------ */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(CudaController_remap_managed_cobject_buffer)(
    NS(CudaController)* SIXTRL_RESTRICT ctrl,
    NS(cuda_arg_buffer_t) SIXTRL_RESTRICT managed_buffer_begin,
    NS(arch_size_t) const slot_size );

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(CudaController_is_managed_cobject_buffer_remapped)(
    NS(CudaController)* SIXTRL_RESTRICT ctrl,
    NS(cuda_arg_buffer_t) SIXTRL_RESTRICT managed_buffer_begin,
    NS(arch_size_t) const slot_size );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(CudaController_send_memory)(
    NS(CudaController)* SIXTRL_RESTRICT ctrl,
    NS(cuda_arg_buffer_t) SIXTRL_RESTRICT destination,
    void const* SIXTRL_RESTRICT source, NS(arch_size_t) const source_length );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(CudaController_receive_memory)(
    NS(CudaController)* SIXTRL_RESTRICT ctrl,
    void* SIXTRL_RESTRICT destination,
    NS(cuda_const_arg_buffer_t) SIXTRL_RESTRICT source,
    NS(arch_size_t) const source_length );

/* ------------------------------------------------------------------------ */



#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_CONTROLLER_H__ */

/* end: sixtracklib/cuda/controller.h */
