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

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(CudaController_select_node_by_cuda_device_index)(
    NS(CudaController)* SIXTRL_RESTRICT ctrl, int cuda_device_index );

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(CudaController_select_node_by_cuda_pci_bus_id)(
    NS(CudaController)* SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT cuda_pci_bus_id );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(CudaController_delete)(
    NS(CudaController)* SIXTRL_RESTRICT ctrl );

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

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_CONTROLLER_H__ */

/* end: sixtracklib/cuda/controller.h */
