#ifndef SIXTRACKLIB_CUDA_CONTROL_DEFAULT_KERNEL_CONFIGS_H__
#define SIXTRACKLIB_CUDA_CONTROL_DEFAULT_KERNEL_CONFIGS_H__

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/kernel_config_base.h"

#include "sixtracklib/cuda/definitions.h"
#include "sixtracklib/cuda/control/kernel_config.h"
#include "sixtracklib/cuda/control/node_info.h"

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++ */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(CudaKernelConfig_configure_track_until_turn_kernel)(
    NS(CudaKernelConfig)* SIXTRL_RESTRICT kernel_config,
    const NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info,
    NS(buffer_size_t) const total_num_particles_to_track );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(CudaKernelConfig_configure_track_elem_by_elem_until_turn_kernel)(
    NS(CudaKernelConfig)* SIXTRL_RESTRICT kernel_config,
    const NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info,
    NS(buffer_size_t) const total_num_particles_to_track );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(CudaKernelConfig_configure_track_elem_by_elem_until_turn_kernel)(
    NS(CudaKernelConfig)* SIXTRL_RESTRICT kernel_config,
    const NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info,
    NS(buffer_size_t) const total_num_particles_to_track );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(CudaKernelConfig_configure_fetch_particles_addresses_kernel)(
    NS(CudaKernelConfig)* SIXTRL_RESTRICT kernel_config,
    const NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info,
    NS(buffer_size_t) const num_particle_sets );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(CudaKernelConfig_configure_assign_output_to_beam_monitors_kernel)(
    NS(CudaKernelConfig)* SIXTRL_RESTRICT kernel_config,
    const NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info,
    NS(buffer_size_t) const num_beam_monitors );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(CudaKernelConfig_configure_assign_output_to_elem_by_elem_config_kernel)(
    NS(CudaKernelConfig)* SIXTRL_RESTRICT kernel_config,
    const NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info );



#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++ */

#endif /* SIXTRACKLIB_CUDA_CONTROL_DEFAULT_KERNEL_CONFIGS_H__*/
/* end: sixtracklib/cuda/control/default_kernel_config.h */

