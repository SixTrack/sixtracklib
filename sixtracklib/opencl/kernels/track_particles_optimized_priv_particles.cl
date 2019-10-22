#ifndef SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_OPTIMIZED_PRIV_PARTICLES_KERNEL_CL__
#define SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_OPTIMIZED_PRIV_PARTICLES_KERNEL_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/default_compile_options.h"

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/track/definitions.h"
    #include "sixtracklib/common/track/track_kernel_opt_impl.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__kernel void NS(Track_particles_until_turn_opt_pp_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    SIXTRL_UINT64_T const part_set_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_INT64_T const until_turn, SIXTRL_UINT64_T const slot_size )
{
    typedef NS(particle_num_elements_t) nelements_t;
    nelements_t const part_idx = ( nelements_t )get_global_id( 0 );
    nelements_t const part_idx_stride = ( nelements_t )get_global_size( 0 );

    NS(Track_particles_until_turn_opt_kernel_impl)( pbuffer, part_set_index,
        part_idx, part_idx_stride, belem_buffer, until_turn, slot_size );
}

__kernel void NS(Track_particles_elem_by_elem_opt_pp_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    SIXTRL_UINT64_T const part_set_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig) *const
        SIXTRL_RESTRICT elem_by_elem_config,
    SIXTRL_INT64_T const until_turn, SIXTRL_UINT64_T const slot_size )
{
    typedef NS(particle_num_elements_t) nelements_t;
    nelements_t const part_idx = ( nelements_t )get_global_id( 0 );
    nelements_t const part_idx_stride = ( nelements_t )get_global_size( 0 );

    NS(Track_particles_elem_by_elem_until_turn_opt_kernel_impl)( pbuffer,
        part_set_index, part_idx, part_idx_stride, belem_buffer,
            elem_by_elem_config, until_turn, slot_size );
}

__kernel void NS(Track_particles_line_opt_pp_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    SIXTRL_UINT64_T const part_set_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_UINT64_T const line_begin_idx, SIXTRL_UINT64_T const line_end_idx,
    SIXTRL_UINT64_T const finish_turn_value, SIXTRL_UINT64_T const slot_size )
{
    typedef NS(particle_num_elements_t) nelements_t;
    nelements_t const part_idx = ( nelements_t )get_global_id( 0 );
    nelements_t const part_idx_stride = ( nelements_t )get_global_size( 0 );

    NS(Track_particles_line_opt_kernel_impl)( pbuffer, part_set_index, part_idx,
        part_idx_stride, belem_buffer, line_begin_idx, line_end_idx,
            ( bool )( finish_turn_value == ( SIXTRL_UINT64_T )1u ), slot_size );
}

#endif /* SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_OPTIMIZED_PRIV_PARTICLES_KERNEL_CL__ */

/* end: sixtracklib/opencl/kernels/track_particles_optimized_priv_particles.cl */
