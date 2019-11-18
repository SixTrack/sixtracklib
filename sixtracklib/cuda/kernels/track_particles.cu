#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/kernels/track_particles.cuh"
#endif /* !defined( SIXTRL_NO_INCLUDES )     */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>

    #include <cuda_runtime_api.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/debug_register.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/track/definitions.h"
    #include "sixtracklib/common/track/track_kernel_impl.h"
    #include "sixtracklib/cuda/cuda_tools.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__global__ void NS(Track_particles_until_turn_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const part_set_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT belem_buffer,
    NS(buffer_size_t) const until_turn, NS(buffer_size_t) const slot_size )
{
    typedef NS(particle_num_elements_t) nelements_t;
    nelements_t const part_idx = NS(Cuda_get_1d_thread_id_in_kernel)();
    nelements_t const part_idx_stride =
        NS(Cuda_get_total_num_threads_in_kernel)();

    NS(Track_particles_until_turn_kernel_impl)( pbuffer, part_set_index,
        part_idx, part_idx_stride, belem_buffer, until_turn, slot_size );
}

__global__ void NS(Track_particles_until_turn_cuda_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const part_set_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT belem_buffer,
    NS(buffer_size_t) const until_turn,
    NS(buffer_size_t) const slot_size,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flags )
{
    typedef NS(particle_num_elements_t) nelements_t;
    nelements_t const part_idx = NS(Cuda_get_1d_thread_id_in_kernel)();
    nelements_t const part_idx_stride =
        NS(Cuda_get_total_num_threads_in_kernel)();

    NS(arch_debugging_t) status_flags = SIXTRL_ARCH_DEBUGGING_GENERAL_FAILURE;

    NS(Track_particles_until_turn_debug_kernel_impl)( pbuffer, part_set_index,
        part_idx, part_idx_stride, belem_buffer, until_turn, slot_size,
            &status_flags );

    NS(Cuda_collect_status_flag_value)( ptr_status_flags, status_flags );
}

/* ------------------------------------------------------------------------- */

__global__ void NS(Track_track_elem_by_elem_until_turn_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const part_set_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig) *const
        SIXTRL_RESTRICT elem_by_elem_config,
    NS(buffer_size_t) const until_turn, NS(buffer_size_t) const slot_size )
{
    typedef NS(particle_num_elements_t) nelements_t;
    nelements_t const part_idx = NS(Cuda_get_1d_thread_id_in_kernel)();
    nelements_t const part_idx_stride =
        NS(Cuda_get_total_num_threads_in_kernel)();

    NS(Track_particles_elem_by_elem_until_turn_kernel_impl)( pbuffer,
        part_set_index, part_idx, part_idx_stride, belem_buffer,
            elem_by_elem_config, until_turn, slot_size );
}

__global__ void NS(Track_track_elem_by_elem_until_turn_cuda_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const part_set_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig) *const
        SIXTRL_RESTRICT elem_by_elem_config,
    NS(buffer_size_t) const until_turn, NS(buffer_size_t) const slot_size,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flags )
{
    typedef NS(particle_num_elements_t) nelements_t;
    nelements_t const part_idx = NS(Cuda_get_1d_thread_id_in_kernel)();
    nelements_t const part_idx_stride =
        NS(Cuda_get_total_num_threads_in_kernel)();

    NS(arch_debugging_t) status_flags = SIXTRL_ARCH_DEBUGGING_GENERAL_FAILURE;

    NS(Track_particles_elem_by_elem_until_turn_debug_kernel_impl)( pbuffer,
        part_set_index, part_idx, part_idx_stride, belem_buffer,
            elem_by_elem_config, until_turn, slot_size, &status_flags );

    NS(Cuda_collect_status_flag_value)( ptr_status_flags, status_flags );
}

/* ------------------------------------------------------------------------- */

__global__ void NS(Track_particles_line_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const part_set_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT belem_buffer,
    NS(buffer_size_t) const belem_begin_id,
    NS(buffer_size_t) const belem_end_id,
    bool const finish_turn, NS(buffer_size_t) const slot_size )
{
    typedef NS(particle_num_elements_t) nelements_t;
    nelements_t const part_idx = NS(Cuda_get_1d_thread_id_in_kernel)();
    nelements_t const part_idx_stride =
        NS(Cuda_get_total_num_threads_in_kernel)();

    NS(Track_particles_line_kernel_impl)( pbuffer, part_set_index, part_idx,
        part_idx_stride, belem_buffer, belem_begin_id, belem_end_id,
            finish_turn, slot_size );
}

__global__ void NS(Track_particles_line_cuda_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const part_set_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT belem_buffer,
    NS(buffer_size_t) const belem_begin_id,
    NS(buffer_size_t) const belem_end_id,
    bool const finish_turn, NS(buffer_size_t) const slot_size,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flags )
{
    typedef NS(particle_num_elements_t) nelements_t;
    nelements_t const part_idx = NS(Cuda_get_1d_thread_id_in_kernel)();
    nelements_t const part_idx_stride =
        NS(Cuda_get_total_num_threads_in_kernel)();

    NS(arch_debugging_t) status_flags = SIXTRL_ARCH_DEBUGGING_GENERAL_FAILURE;

    NS(Track_particles_line_debug_kernel_impl)( pbuffer, part_set_index,
        part_idx, part_idx_stride, belem_buffer, belem_begin_id, belem_end_id,
            finish_turn, slot_size, &status_flags );

    NS(Cuda_collect_status_flag_value)( ptr_status_flags, status_flags );
}

/* end sixtracklib/cuda/kernels/track_particles_kernels.cu */
