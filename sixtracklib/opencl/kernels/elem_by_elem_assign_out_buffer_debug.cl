#ifndef SIXTRACKLIB_OPENCL_KERNELS_ELEM_BY_ELEM_ASSIGN_OUT_BUFFER_DEBUG_CL__
#define SIXTRACKLIB_OPENCL_KERNELS_ELEM_BY_ELEM_ASSIGN_OUT_BUFFER_DEBUG_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/default_compile_options.h"

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/internal/particles_defines.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/output/elem_by_elem_kernel_impl.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__kernel void NS(ElemByElem_assign_out_buffer_from_offset_debug_opencl)(
     SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT elem_by_elem_config,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer,
    SIXTRL_UINT64_T const out_buffer_index_offset,
    SIXTRL_UINT64_T const slot_size,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flag )
{
    typedef NS(buffer_size_t) buf_size_t;

    if( ( buf_size_t )get_global_id( 0 ) == ( buf_size_t )0u )
    {
        NS(arch_debugging_t) dbg = SIXTRL_ARCH_DEBUGGING_GENERAL_FAILURE;
        NS(ElemByElemConfig_assign_output_buffer_debug_kernel_impl)(
            elem_by_elem_config, out_buffer, out_buffer_index_offset,
                slot_size, &dbg );

        if( ptr_status_flag != SIXTRL_NULLPTR ) *ptr_status_flag = dbg;
    }
}

#endif /* SIXTRACKLIB_OPENCL_KERNELS_ELEM_BY_ELEM_ASSIGN_OUT_BUFFER_DEBUG_CL__*/

/* end: sixtracklib/opencl/kernels/elem_by_elem_assign_out_buffer_debug.cl */
