#ifndef SIXTRACKLIB_CUDA_INTERNAL_CONTROLLER_BASE_H__
#define SIXTRACKLIB_CUDA_INTERNAL_CONTROLLER_BASE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/internal/controller_base.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
extern "C" {
#endif /* C++, Host */

#if !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(CudaControllerBase_delete)(
    NS(CudaControllerBase)* SIXTRL_RESTRICT ctrl );

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(CudaController_select_node_by_cuda_device_index)(
    NS(CudaControllerBase)* SIXTRL_RESTRICT ctrl, int cuda_device_index );

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(CudaController_select_node_by_cuda_pci_bus_id)(
    NS(CudaControllerBase)* SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT cuda_pci_bus_id );

#endif /* !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_INTERNAL_CONTROLLER_BASE_H__ */

/* end: sixtracklib/cuda/internal/controller_base.h */
