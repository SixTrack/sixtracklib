#include "sixtracklib/cuda/control/default_kernel_config.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/kernel_config_base.h"
#include "sixtracklib/cuda/definitions.h"
#include "sixtracklib/cuda/control/kernel_config.h"
#include "sixtracklib/cuda/control/node_info.h"

SIXTRL_STATIC SIXTRL_HOST_FN NS(arch_status_t)
NS(CudaKernelConfig_configure_generic_track_kernel)(
    NS(CudaKernelConfig)* SIXTRL_RESTRICT kernel_config,
    const NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info,
    NS(buffer_size_t) const total_num_particles_to_track );


NS(arch_status_t) NS(CudaKernelConfig_configure_track_until_turn_kernel)(
    NS(CudaKernelConfig)* SIXTRL_RESTRICT kernel_config,
    const NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info,
    NS(buffer_size_t) const total_num_particles_to_track )
{
    return NS(CudaKernelConfig_configure_generic_track_kernel)(
        kernel_config, node_info, total_num_particles_to_track );
}

NS(arch_status_t)
NS(CudaKernelConfig_configure_track_elem_by_elem_until_turn_kernel)(
    NS(CudaKernelConfig)* SIXTRL_RESTRICT kernel_config,
    const NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info,
    NS(buffer_size_t) const total_num_particles_to_track )
{
    return NS(CudaKernelConfig_configure_generic_track_kernel)(
        kernel_config, node_info, total_num_particles_to_track );
}

NS(arch_status_t) NS(CudaKernelConfig_configure_track_line_kernel)(
    NS(CudaKernelConfig)* SIXTRL_RESTRICT kernel_config,
    const NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info,
    NS(buffer_size_t) const total_num_particles_to_track )
{
    return NS(CudaKernelConfig_configure_generic_track_kernel)(
        kernel_config, node_info, total_num_particles_to_track );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(arch_status_t) NS(CudaKernelConfig_configure_fetch_particles_addresses_kernel)(
    NS(CudaKernelConfig)* SIXTRL_RESTRICT kernel_config,
    const NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info,
    NS(buffer_size_t) const num_particle_sets )
{
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);

     if( ( kernel_config != SIXTRL_NULLPTR ) &&
        ( node_info != SIXTRL_NULLPTR ) )
    {
        NS(buffer_size_t) const warp_size =
            NS(CudaNodeInfo_get_warp_size)( node_info );

        NS(buffer_size_t) threads_per_block = ( NS(buffer_size_t) )128;


        NS(buffer_size_t) num_blocks = num_particle_sets / threads_per_block;

        if( ( num_blocks * threads_per_block ) < num_particle_sets )
        {
            ++num_blocks;
        }

        if( ( NS(KernelConfig_set_num_work_items_1d)(
                kernel_config, num_blocks ) ) &&
            ( NS(KernelConfig_set_work_group_sizes_1d)(
                kernel_config, threads_per_block ) ) &&
            ( NS(KernelConfig_set_preferred_work_group_multiple_1d)(
                kernel_config, warp_size ) ) )
        {
            if( NS(KernelConfig_update)( kernel_config ) )
            {
                status = NS(ARCH_STATUS_SUCCESS);
            }
        }
    }

    return status;
}

NS(arch_status_t)
NS(CudaKernelConfig_configure_assign_output_to_beam_monitors_kernel)(
    NS(CudaKernelConfig)* SIXTRL_RESTRICT kernel_config,
    const NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info,
    NS(buffer_size_t) const num_beam_monitors )
{
    ( void )num_beam_monitors;

    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);

     if( ( kernel_config != SIXTRL_NULLPTR ) &&
         ( node_info != SIXTRL_NULLPTR ) )
    {
        NS(buffer_size_t) const threads_per_block = ( NS(buffer_size_t) )1u;
        NS(buffer_size_t) const num_blocks = ( NS(buffer_size_t) )1u;
        
        NS(buffer_size_t) const warp_size = 
            NS(CudaNodeInfo_get_warp_size)( node_info );

        if( ( NS(KernelConfig_set_num_work_items_1d)(
                kernel_config, num_blocks ) ) &&
            ( NS(KernelConfig_set_work_group_sizes_1d)(
                kernel_config, threads_per_block ) ) &&
            ( NS(KernelConfig_set_preferred_work_group_multiple_1d)(
                kernel_config, warp_size ) ) )
        {
            if( NS(KernelConfig_update)( kernel_config ) )
            {
                status = NS(ARCH_STATUS_SUCCESS);
            }
        }
    }

    return status;
}

NS(arch_status_t)
NS(CudaKernelConfig_configure_assign_output_to_elem_by_elem_config_kernel)(
    NS(CudaKernelConfig)* SIXTRL_RESTRICT kernel_config,
    const NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);

     if( ( kernel_config != SIXTRL_NULLPTR ) &&
         ( node_info != SIXTRL_NULLPTR ) )
    {
        NS(buffer_size_t) const threads_per_block = ( NS(buffer_size_t) )1u;
        NS(buffer_size_t) const num_blocks = ( NS(buffer_size_t) )1u;
        
        NS(buffer_size_t) const warp_size = 
            NS(CudaNodeInfo_get_warp_size)( node_info );

        if( ( NS(KernelConfig_set_num_work_items_1d)(
                kernel_config, num_blocks ) ) &&
            ( NS(KernelConfig_set_work_group_sizes_1d)(
                kernel_config, threads_per_block ) ) &&
            ( NS(KernelConfig_set_preferred_work_group_multiple_1d)(
                kernel_config, warp_size ) ) )
        {
            if( NS(KernelConfig_update)( kernel_config ) )
            {
                status = NS(ARCH_STATUS_SUCCESS);
            }
        }
    }

    return status;
}

/* ========================================================================= */

NS(arch_status_t) NS(CudaKernelConfig_configure_generic_track_kernel)(
    NS(CudaKernelConfig)* SIXTRL_RESTRICT kernel_config,
    const NS(CudaNodeInfo) *const SIXTRL_RESTRICT node_info,
    NS(buffer_size_t) const total_num_particles_to_track )
{
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);

     if( ( kernel_config != SIXTRL_NULLPTR ) &&
        ( node_info != SIXTRL_NULLPTR ) )
    {
        NS(buffer_size_t) const warp_size =
            NS(CudaNodeInfo_get_warp_size)( node_info );

        NS(buffer_size_t) threads_per_block = ( NS(buffer_size_t) )128;


        NS(buffer_size_t) num_blocks =
            total_num_particles_to_track / threads_per_block;

        if( ( num_blocks * threads_per_block ) < total_num_particles_to_track )
        {
            ++num_blocks;
        }

        if( ( NS(KernelConfig_set_num_work_items_1d)(
                kernel_config, num_blocks ) ) &&
            ( NS(KernelConfig_set_work_group_sizes_1d)(
                kernel_config, threads_per_block ) ) &&
            ( NS(KernelConfig_set_preferred_work_group_multiple_1d)(
                kernel_config, warp_size ) ) )
        {
            if( NS(KernelConfig_update)( kernel_config ) )
            {
                status = NS(ARCH_STATUS_SUCCESS);
            }
        }
    }

    return status;
}



/* end: sixtracklib/cuda/control/default_kernel_config.c */
