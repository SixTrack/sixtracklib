#ifndef SIXTRACKLIB_OPENCL_KERNELS_ELEM_BY_ELEM_ASSIGN_OUT_BUFFER_CL__
#define SIXTRACKLIB_OPENCL_KERNELS_ELEM_BY_ELEM_ASSIGN_OUT_BUFFER_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/default_compile_options.h"

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/internal/particles_defines.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/output/elem_by_elem_kernel_impl.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__kernel void NS(ElemByElem_assign_out_buffer_from_offset_opencl)(
     SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer,
    SIXTRL_UINT64_T const out_buffer_index_offset,
    SIXTRL_UINT64_T const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    if( ( buf_size_t )get_global_id( 0 ) == ( buf_size_t )0u )
    {
        NS(ElemByElemConfig_assign_output_buffer_kernel_impl)(
            config, out_buffer, out_buffer_index_offset, slot_size );
    }
}

#endif /* SIXTRACKLIB_OPENCL_KERNELS_ELEM_BY_ELEM_ASSIGN_OUT_BUFFER_CL__ */
/* end: */
