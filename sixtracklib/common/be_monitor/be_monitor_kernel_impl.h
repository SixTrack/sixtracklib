#ifndef SIXTRACKLIB_COMMON_BE_MONITOR_BE_OUTPUT_KERNEL_IMPL_C99_H__
#define SIXTRACKLIB_COMMON_BE_MONITOR_BE_OUTPUT_KERNEL_IMPL_C99_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/particles/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, Host */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(BeamMonitor_assign_output_kernel_impl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) be_monitors_idx,
    NS(buffer_size_t) const be_monitors_idx_stride,
    NS(buffer_size_t) output_buffer_index_offset,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(BeamMonitor_assign_output_debug_kernel_impl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) be_monitors_idx,
    NS(buffer_size_t) const be_monitors_idx_stride,
    NS(buffer_size_t) output_buffer_index_offset,
    NS(buffer_size_t) const slot_size,
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
NS(BeamMonitor_assign_output_kernel_impl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) be_monitors_idx,
    NS(buffer_size_t) const be_monitors_idx_stride,
    NS(buffer_size_t) const output_buffer_index_offset,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* ptr_be_monitor_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* be_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* out_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Particles) const* ptr_out_particles_t;
    typedef NS(be_monitor_index_t) mon_index_t;
    typedef NS(be_monitor_turn_t) nturn_t;

    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_SUCCESS;
    buf_size_t const num_output_buffers = NS(ManagedBuffer_get_num_objects)(
        out_buffer, slot_size );

    be_iter_t be_it  = NS(ManagedBuffer_get_objects_index_begin)(
        be_buffer_begin, slot_size );

    be_iter_t belem_end = NS(ManagedBuffer_get_objects_index_end)(
        be_buffer_begin, slot_size );

    buf_size_t next_be_mon_idx = ( buf_size_t )0u;

    out_iter_t out_it  = NS(ManagedBuffer_get_objects_index_begin)(
        out_buffer, slot_size );

    out_iter_t out_end = NS(ManagedBuffer_get_objects_index_end)(
        out_buffer, slot_size );

    SIXTRL_ASSERT( belem_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( out_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( slot_size > ( buf_size_t )8u );
    SIXTRL_ASSERT( be_monitors_idx_stride > ( buf_size_t )0u );

    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
        belem_buffer, slot_size ) );

    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
        out_buffer, slot_size ) );

    SIXTRL_ASSERT( be_it  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( uintptr_t )be_end >= ( uintptr_t )be_it );

    SIXTRL_ASSERT( out_it != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)( out_buffer, slot_size )
        > output_buffer_index_offset );

    out_it = out_it + output_buffer_index_offset;

    for( ; be_it != be_end ; ++be_it )
    {
        if( NS(Object_get_type_id)( be_it ) == NS(OBJECT_TYPE_BEAM_MONITOR) )
        {
            uintptr_t const be_mon_addr =
                ( uintptr_t )NS(Object_get_begin_addr)( be_it );

            if( ( be_monitors_idx == next_be_mon_idx ) &&
                ( be_mon_addr != ( uintptr_t )0u ) )
            {
                ptr_be_monitor_t be_mon = ( ptr_be_monitor_t )be_mon_addr;
                nturn_t const nn = NS(BeamMonitor_get_num_stores)( be_mon );

                SIXTRL_ASSERT( be_mon != SIXTRL_NULLPTR );
                SIXTRL_ASSERT( out_it != out_end );

                if( nn > ( nturn_t )0u )
                {
                    ptr_out_particles_t out_particles = ( ptr_out_particles_t
                        )NS(BufferIndex_get_const_particles)( out_it );

                    SIXTRL_ASSERT( out_particles != SIXTRL_NULLPTR );

                    buf_size_t const num_stored_particles =
                        NS(Particles_get_num_of_particles)( particles );

                    mon_index_t const min_part_id =
                        NS(BeamMonitor_get_min_particle_id)( be_mon );

                    mon_index_t const max_part_id =
                        NS(BeamMonitor_get_max_particle_id)( be_mon );

                    buf_size_t const stored_particles_per_turn =
                        ( max_part_id >= min_part_id )
                        ? ( buf_size_t )( max_part_id - min_part_id  + 1u )
                        : ZERO;

                    if( ( nn > 0 ) && ( stored_particles_per_turn > ZERO ) &&
                        ( out_particles != SIXTRL_NULLPTR ) &&
                        ( ( stored_particles_per_turn * ( buf_size_t )nn ) <=
                            num_stored_particles ) )
                    {
                        NS(BeamMonitor_set_out_address)(
                            be_mon, NS(Object_get_begin_addr)( out_it ) );
                    }
                    else
                    {
                        NS(BeamMonitor_set_out_address)( be_mon, 0u );
                    }
                }
                else
                {
                    NS(BeamMonitor_set_out_address)( be_mon, 0u );
                }
            }

            if( be_monitors_idx == next_be_mon_idx )
            {
                be_monitors_idx += be_monitors_idx_stride;
            }

            ++next_be_mon_idx;
            ++out_it;

            if( ( out_it == out_end ) || ( NS(Object_get_type_id)( out_it ) !=
                    NS(OBJECT_TYPE_PARTICLE) ) )
            {
                status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
                break;
            }
        }
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t)
NS(BeamMonitor_assign_output_debug_kernel_impl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) const output_buffer_index_offset,
    NS(buffer_size_t) const slot_size,
    SIXTRL_ARGPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flags )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* ptr_be_monitor_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* be_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* out_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Particles) const* ptr_out_particles_t;
    typedef NS(be_monitor_index_t) mon_index_t;
    typedef NS(be_monitor_turn_t) nturn_t;

    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_SUCCESS;
    NS(arch_debugging_t) flags = SIXTRL_ARCH_DEBUGGING_MIN_FLAG;

    NS(arch_debugging_t) const BE_MONITOR_IDX_ILLEGAL_FLAG      = flags;
    NS(arch_debugging_t) const OUT_BUFFER_NULL_FLAG             = flags <<  1u;
    NS(arch_debugging_t) const BELEM_BUFFER_NULL_FLAG           = flags <<  2u;
    NS(arch_debugging_t) const SLOT_SIZE_ILLEGAL_FLAG           = flags <<  3u;
    NS(arch_debugging_t) const OFFSET_INDEX_ILLEGAL_FLAG        = flags <<  4u;
    NS(arch_debugging_t) const OUT_BUFFER_REQUIRES_REMAP_FLAG   = flags <<  5u;
    NS(arch_debugging_t) const BELEM_BUFFER_REQUIRES_REMAP_FLAG = flags <<  6u;
    NS(arch_debugging_t) const BEAM_MONITOR_ILLEGAL_FLAG        = flags <<  7u;
    NS(arch_debugging_t) const BEAM_ELEM_LINE_ILLEGAL_FLAG      = flags <<  8u;
    NS(arch_debugging_t) const OUT_PARTICLES_ILLEGAL_FLAG       = flags <<  8u;

    flags = ( NS(arch_debugging_t) )0u;

    if( ( belem_buffer != SIXTRL_NULLPTR ) &&
        ( out_buffer != SIXTRL_NULLPTR ) && ( slot_size > ( buf_size_t )0u ) &&
        ( be_monitors_idx_stride > ( buf_size_t )0u ) )
    {
        buf_size_t const num_output_buffers =
            NS(ManagedBuffer_get_num_objects)( out_buffer, slot_size );

        be_iter_t be_it  = NS(ManagedBuffer_get_objects_index_begin)(
            be_buffer_begin, slot_size );

        be_iter_t belem_end = NS(ManagedBuffer_get_objects_index_end)(
            be_buffer_begin, slot_size );

        buf_size_t next_be_mon_idx = ( buf_size_t )0u;

        out_iter_t out_it  = NS(ManagedBuffer_get_objects_index_begin)(
            out_buffer, slot_size );

        out_iter_t out_end = NS(ManagedBuffer_get_objects_index_end)(
            out_buffer, slot_size );

        if( ( !NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) ) &&
            ( !NS(ManagedBuffer_needs_remapping)( out_buffer, slot_size ) ) &&
            ( be_it != SIXTRL_NULLPTR ) &&
            ( ( uintptr_t )be_end >= ( uintptr_t )be_it ) &&
            ( NS(ManagedBuffer_get_num_objects)( out_buffer, slot_size ) >
                output_buffer_index_offset ) )
        {
            out_it = out_it = output_buffer_index_offset;

            for( ; be_it != be_end ; ++be_it )
            {
                if( NS(Object_get_type_id)( be_it ) ==
                    NS(OBJECT_TYPE_BEAM_MONITOR) )
                {
                    uintptr_t const be_mon_addr =
                        ( uintptr_t )NS(Object_get_begin_addr)( be_it );

                    if( ( be_monitors_idx == next_be_mon_idx ) &&
                        ( be_mon_addr != ( uintptr_t )0u ) )
                    {
                        ptr_be_monitor_t be_mon =
                            ( ptr_be_monitor_t )be_mon_addr;

                        nturn_t const nn =
                            NS(BeamMonitor_get_num_stores)( be_mon );

                        if( nn > ( nturn_t )0u )
                        {
                            ptr_out_particles_t out_particles = ( ptr_out_particles_t
                                )NS(BufferIndex_get_const_particles)( out_it );

                            SIXTRL_ASSERT( out_particles != SIXTRL_NULLPTR );

                            buf_size_t const num_stored_particles =
                                NS(Particles_get_num_of_particles)( particles );

                            mon_index_t const min_part_id =
                                NS(BeamMonitor_get_min_particle_id)( be_mon );

                            mon_index_t const max_part_id =
                                NS(BeamMonitor_get_max_particle_id)( be_mon );

                            buf_size_t const stored_particles_per_turn =
                                ( max_part_id >= min_part_id )
                                ? ( buf_size_t )(
                                    max_part_id - min_part_id  + 1u )
                                : ZERO;

                            if( ( stored_particles_per_turn > ZERO ) &&
                                ( nn > 0 ) &&
                                ( out_particles != SIXTRL_NULLPTR ) &&
                                ( ( stored_particles_per_turn *
                                    ( buf_size_t )nn ) <= num_stored_particles ) )
                            {
                                NS(BeamMonitor_set_out_address)( be_mon,
                                    NS(Object_get_begin_addr)( out_it ) );
                            }
                            else
                            {
                                NS(BeamMonitor_set_out_address)( be_mon, 0u );
                            }
                        }
                        else
                        {
                            NS(BeamMonitor_set_out_address)( be_mon, 0u );
                        }
                    }

                    if( be_monitors_idx == next_be_mon_idx )
                    {
                        be_monitors_idx += be_monitors_idx_stride;
                    }

                    ++next_be_mon_idx;
                    ++out_it;

                    if( ( out_it == out_end ) ||
                        ( NS(Object_get_type_id)( out_it ) !=
                            NS(OBJECT_TYPE_PARTICLE) ) )
                    {
                        status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
                        break;
                    }
                }
            }
        }
        else
        {
            if( NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) )
                    flags |= BELEM_BUFFER_REQUIRES_REMAP_FLAG;

            if( NS(ManagedBuffer_needs_remapping)( out_buffer, slot_size ) )
                    flags |= OUT_BUFFER_REQUIRES_REMAP_FLAG;

            if( ( be_it == SIXTRL_NULLPTR ) ||
                ( ( uintptr_t )be_end >= ( uintptr_t )be_it ) )
                flags |= BEAM_ELEM_LINE_ILLEGAL_FLAG;

            if( NS(ManagedBuffer_get_num_objects)( out_buffer, slot_size ) <=
                    output_buffer_index_offset )
                flags |= OUT_PARTICLES_ILLEGAL_FLAG;
        }
    }
    else
    {
        if( belem_buffer == SIXTRL_NULLPTR ) flags |= BELEM_BUFFER_NULL_FLAG;
        if( out_buffer   == SIXTRL_NULLPTR ) flags |= OUT_BUFFER_NULL_FLAG;
        if( slot_size >   ( buf_size_t )0u ) flags |= SLOT_SIZE_ILLEGAL_FLAG;
        if( be_monitors_idx_stride,
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

#endif /* SIXTRACKLIB_COMMON_BE_MONITOR_BE_OUTPUT_KERNEL_IMPL_C99_H__ */
/* end: sixtracklib/common/be_monitor/be_monitor_kernel_impl.h */
