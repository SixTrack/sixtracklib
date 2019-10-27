#ifndef SIXTRACKLIB_COMMON_OUTPUT_ELEM_BY_ELEM_KERNEL_IMPL_C99_H__
#define SIXTRACKLIB_COMMON_OUTPUT_ELEM_BY_ELEM_KERNEL_IMPL_C99_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/particles/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, Host */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(ElemByElemConfig_assign_output_buffer_kernel_impl)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT elem_by_elem_config,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) const output_buffer_index_offset,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(ElemByElemConfig_assign_output_buffer_debug_kernel_impl)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT elem_by_elem_config,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) const output_buffer_index_offset,
    NS(buffer_size_t) const slot_size,
    SIXTRL_ARGPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flags );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(ElemByElemConfig_set_output_buffer_addr_kernel_impl)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT elem_by_elem_config,
    SIXTRL_UINT64_T const out_buffer_addr );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(ElemByElemConfig_set_output_buffer_addr_debug_kernel_impl)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT elem_by_elem_config,
    SIXTRL_UINT64_T const out_buffer_addr,
    SIXTRL_ARGPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flags );

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

/* ************************************************************************* */
/* Inline functions implementation */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/control/debug_register.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/be_monitor/be_monitor.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, Host */

SIXTRL_INLINE NS(arch_status_t)
NS(ElemByElemConfig_assign_output_buffer_kernel_impl)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT elem_by_elem_config,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) const output_buffer_index_offset,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(particle_num_elements_t) nelements_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* out_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Particles) const* ptr_out_particles_t;

    nelements_t const required_num_out_particles =
        NS(ElemByElemConfig_get_out_store_num_particles)( elem_by_elem_config );

    uintptr_t out_addr = ( uintptr_t )0u;
    out_iter_t out_it  = NS(ManagedBuffer_get_const_objects_index_begin)(
        out_buffer, slot_size );

    SIXTRL_ASSERT( elem_by_elem_config != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( out_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( slot_size > ( NS(buffer_size_t) )0u );
    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
        out_buffer, slot_size ) );

    SIXTRL_ASSERT( out_it != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( output_buffer_index_offset <
        NS(ManagedBuffer_get_num_objects)( out_buffer, slot_size ) );

    out_it = out_it + output_buffer_index_offset;
    out_addr = NS(Object_get_begin_addr)( out_it );

    if( ( out_it != SIXTRL_NULLPTR ) &&
        ( NS(Object_get_type_id)( out_it ) == NS(OBJECT_TYPE_PARTICLE) ) &&
        ( out_addr != ( uintptr_t )0u ) &&
        ( NS(Particles_get_num_of_particles)( ( ptr_out_particles_t )out_addr )
            >= required_num_out_particles ) )
    {
        NS(ElemByElemConfig_set_output_store_address)(
            elem_by_elem_config, out_addr );
    }
    else
    {
        NS(ElemByElemConfig_set_output_store_address)(
            elem_by_elem_config, ( uintptr_t )0u );
    }

    return SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t)
NS(ElemByElemConfig_assign_output_buffer_debug_kernel_impl)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT elem_by_elem_config,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) const output_buffer_index_offset,
    NS(buffer_size_t) const slot_size,
    SIXTRL_ARGPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flags )
{
    typedef NS(particle_num_elements_t) nelements_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* out_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Particles) const* ptr_out_particles_t;

    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    NS(arch_debugging_t) flags = SIXTRL_ARCH_DEBUGGING_MIN_FLAG;

    NS(arch_debugging_t) const ELEM_BY_ELEM_CONF_ILLEGAL_FLAG   = flags;
    NS(arch_debugging_t) const SLOT_SIZE_ILLEGAL_FLAG           = flags <<  1u;
    NS(arch_debugging_t) const OUT_BUFFER_NULL_FLAG             = flags <<  2u;
    NS(arch_debugging_t) const OUT_INDEX_OFFSET_ILLEGAL_FLAG    = flags <<  3u;
    NS(arch_debugging_t) const OUT_PARTICLES_ILLEGAL_FLAG       = flags <<  4u;
    NS(arch_debugging_t) const OUT_BUFFER_REQUIRES_REMAP_FLAG   = flags <<  5u;

    flags = ( NS(arch_debugging_t) )0u;

    if( ( elem_by_elem_config != SIXTRL_NULLPTR ) &&
        ( out_buffer != SIXTRL_NULLPTR ) &&
        ( slot_size > ( NS(buffer_size_t) )0u ) &&
        ( !NS(ManagedBuffer_needs_remapping)( out_buffer, slot_size ) ) )
    {
        nelements_t const required_num_out_particles =
            NS(ElemByElemConfig_get_out_store_num_particles)(
                elem_by_elem_config );

        out_iter_t out_it  = NS(ManagedBuffer_get_const_objects_index_begin)(
            out_buffer, slot_size );

        if( ( out_it != SIXTRL_NULLPTR ) &&
            ( output_buffer_index_offset < NS(ManagedBuffer_get_num_objects)(
                out_buffer, slot_size ) ) )
        {
            uintptr_t out_addr = ( uintptr_t )0u;

            out_it = out_it + output_buffer_index_offset;
            out_addr = NS(Object_get_begin_addr)( out_it );

            if( ( out_addr != ( uintptr_t )0u ) &&
                ( NS(Object_get_type_id)( out_it ) ==
                    NS(OBJECT_TYPE_PARTICLE) ) &&
                ( NS(Particles_get_num_of_particles)(
                    ( ptr_out_particles_t )out_addr ) >=
                        required_num_out_particles ) )
            {
                NS(ElemByElemConfig_set_output_store_address)(
                    elem_by_elem_config, out_addr );

                status = SIXTRL_ARCH_STATUS_SUCCESS;
            }
            else if( NS(Object_get_type_id)( out_it ) ==
                    NS(OBJECT_TYPE_PARTICLE) )
            {
                NS(ElemByElemConfig_set_output_store_address)(
                    elem_by_elem_config, 0u );

                status = SIXTRL_ARCH_STATUS_SUCCESS;
            }
            else
            {
                flags |= OUT_PARTICLES_ILLEGAL_FLAG;
            }
        }
        else
        {
            if( output_buffer_index_offset >= NS(ManagedBuffer_get_num_objects)(
                out_buffer, slot_size ) )
                flags |= OUT_INDEX_OFFSET_ILLEGAL_FLAG;

            if( out_it == SIXTRL_NULLPTR ) flags |= OUT_PARTICLES_ILLEGAL_FLAG;
        }
    }
    else
    {
        if( elem_by_elem_config == SIXTRL_NULLPTR )
                flags |= ELEM_BY_ELEM_CONF_ILLEGAL_FLAG;

        if( slot_size == ( NS(buffer_size_t) )0u )
                flags |= SLOT_SIZE_ILLEGAL_FLAG;

        if( out_buffer == SIXTRL_NULLPTR ) flags |= OUT_BUFFER_NULL_FLAG;

        if( NS(ManagedBuffer_needs_remapping)( out_buffer, slot_size ) )
                flags |= OUT_BUFFER_REQUIRES_REMAP_FLAG;
    }

    if( status != SIXTRL_ARCH_STATUS_SUCCESS )
    {
        flags = NS(DebugReg_store_arch_status)( flags, status );
    }

    if( ptr_status_flags != SIXTRL_NULLPTR )
    {
        *ptr_status_flags = flags;
    }

    return status;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(arch_status_t)
NS(ElemByElemConfig_set_output_buffer_addr_kernel_impl)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT elem_by_elem_config,
    SIXTRL_UINT64_T const out_buffer_addr )
{
    typedef NS(particle_num_elements_t) nelements_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* out_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Particles) const* ptr_out_particles_t;

    SIXTRL_ASSERT( elem_by_elem_config != SIXTRL_NULLPTR );
    NS(ElemByElemConfig_set_output_store_address)(
        elem_by_elem_config, out_buffer_addr );

    return SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t)
NS(ElemByElemConfig_set_output_buffer_addr_debug_kernel_impl)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT elem_by_elem_config,
    SIXTRL_UINT64_T const out_buffer_addr,
    SIXTRL_ARGPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flags )
{
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    NS(arch_debugging_t) flags = ( NS(arch_debugging_t) )0u;

    NS(arch_debugging_t) const ELEM_BY_ELEM_CONF_ILLEGAL_FLAG =
        SIXTRL_ARCH_DEBUGGING_MIN_FLAG;

    if( elem_by_elem_config != SIXTRL_NULLPTR )
    {
        NS(ElemByElemConfig_set_output_store_address)(
            elem_by_elem_config, out_buffer_addr );

        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }
    else
    {
        flags = NS(DebugReg_store_arch_status)(
            ELEM_BY_ELEM_CONF_ILLEGAL_FLAG, status );
    }

    if( ptr_status_flags != SIXTRL_NULLPTR )
    {
        *ptr_status_flags = flags;
    }

    return status;
}

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_OUTPUT_ELEM_BY_ELEM_KERNEL_IMPL_C99_H__ */
/* end: sixtracklib/common/output/output_kernel_impl.h */

