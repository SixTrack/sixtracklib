#ifndef SIXTRACKLIB_COMMON_BE_MONITOR_OUTPUT_BUFFER_H__
#define SIXTRACKLIB_COMMON_BE_MONITOR_OUTPUT_BUFFER_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/be_monitor/be_monitor.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(BeamMonitor_prepare_output_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(BeamMonitor_prepare_output_buffer_for_particle_sets)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT pb,
        NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT indices_begin,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(BeamMonitor_prepare_output_buffer_detailed)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(particle_index_t)  const min_particle_id,
    NS(particle_index_t)  const max_particle_id,
    NS(particle_index_t)  const min_turn_id,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(BeamMonitor_calculate_output_buffer_params)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
    NS(buffer_size_t) const output_buffer_slot_size );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(BeamMonitor_calculate_output_buffer_params_for_particle_sets)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT pb,
    NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
    NS(buffer_size_t) const output_buffer_slot_size );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(BeamMonitor_calculate_output_buffer_params_detailed)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
    NS(buffer_size_t) const output_buffer_slot_size );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(BeamMonitor_assign_output_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t) const until_turn_elem_by_elem );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(BeamMonitor_assign_output_buffer_from_offset)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t) const out_buffer_index_offset );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(BeamMonitor_setup_for_particles_all)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p );

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(BeamMonitor_setup_for_particles)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p );

SIXTRL_FN SIXTRL_STATIC int
NS(BeamMonitor_setup_managed_buffer_for_particles_all)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT belements,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC int NS(BeamMonitor_assign_managed_output_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t) const out_buffer_index_offset,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC int NS(BeamMonitor_assign_managed_output_buffer_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t) const out_buffer_index_offset,
    NS(buffer_size_t) const slot_size,
    SIXTRL_ARGPTR_DEC NS(ctrl_debug_flag_t)* ptr_debug_flag );

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* *****          Implementation of C inline functions                   *** */
/* ************************************************************************* */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE int NS(BeamMonitor_setup_for_particles)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p )
{
    int success = -1;

    typedef NS(particle_index_t)  index_t;
    typedef NS(be_monitor_addr_t) addr_t;

    index_t min_id = ( index_t )0;
    index_t max_id = ( index_t )-1;

    if( ( monitor != SIXTRL_NULLPTR ) &&
        ( 0 == NS(Particles_get_min_max_particle_id_value_no_duplicate_check)(
            p, &min_id, &max_id ) ) )
    {
        SIXTRL_STATIC_VAR index_t const ZERO = ( index_t )0u;

        if( ( min_id >= ZERO ) && ( max_id >= min_id ) )
        {
            NS(BeamMonitor_set_min_particle_id)( monitor, min_id );
            NS(BeamMonitor_set_max_particle_id)( monitor, max_id );
            NS(BeamMonitor_set_out_address)( monitor, ( addr_t )0u );

            success = 0;
        }
    }

    return success;
}

SIXTRL_INLINE int NS(BeamMonitor_setup_managed_buffer_for_particles_all)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT belements,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t) const slot_size )
{
    int success = -1;

    typedef NS(particle_index_t)        index_t;
    typedef NS(be_monitor_addr_t)       addr_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* ptr_obj_t;
    typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*    ptr_beam_monitor_t;

    ptr_obj_t be_it = NS(ManagedBuffer_get_objects_index_begin)(
            belements, slot_size );

    ptr_obj_t be_end = NS(ManagedBuffer_get_objects_index_end)(
            belements, slot_size );

    index_t min_particle_id = ( index_t )0u;
    index_t max_particle_id = ( index_t )-1;

    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
        belements, slot_size ) );

    if( ( be_it != SIXTRL_NULLPTR ) && ( be_end != SIXTRL_NULLPTR ) &&
        ( slot_size != ( NS(buffer_size_t) )0u ) && ( 0 ==
            NS(Particles_get_min_max_particle_id_value_no_duplicate_check)(
                p, &min_particle_id, &max_particle_id ) ) )
    {
        for( ; be_it != be_end ; ++be_it )
        {
            if( NS(Object_get_type_id)( be_it ) ==
                NS(OBJECT_TYPE_BEAM_MONITOR) )
            {
                ptr_beam_monitor_t monitor = ( ptr_beam_monitor_t
                    )NS(Object_get_begin_ptr)( be_it );

                NS(BeamMonitor_set_min_particle_id)( monitor, min_particle_id );
                NS(BeamMonitor_set_max_particle_id)( monitor, max_particle_id );
                NS(BeamMonitor_set_out_address)( monitor, ( addr_t )0 );
            }
        }

        success = 0;
    }

    return success;
}

SIXTRL_INLINE int NS(BeamMonitor_assign_managed_output_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t) const out_buffer_index_offset,
    NS(buffer_size_t) const slot_size )
{
    int success = -1;

    typedef NS(buffer_size_t)                               buf_size_t;
    typedef NS(be_monitor_turn_t)                           nturn_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*        ptr_obj_t;
    typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*           ptr_beam_monitor_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Particles) const*  ptr_particles_t;
    typedef NS(be_monitor_index_t)                          mon_index_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    ptr_obj_t out_it = NS(ManagedBuffer_get_objects_index_begin)(
        out_buffer, slot_size );

    ptr_obj_t out_end = NS(ManagedBuffer_get_objects_index_end)(
        out_buffer, slot_size );

    ptr_obj_t be_it  = NS(ManagedBuffer_get_objects_index_begin)(
        beam_elements, slot_size );

    ptr_obj_t be_end = NS(ManagedBuffer_get_objects_index_end)(
        beam_elements, slot_size );

    nturn_t const first_turn_id = ( nturn_t )min_turn_id;

    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
        out_buffer, slot_size ) );

    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
        beam_elements, slot_size ) );

    SIXTRL_ASSERT( out_buffer_index_offset <
        NS(ManagedBuffer_get_num_objects)( out_buffer, slot_size ) );

    SIXTRL_ASSERT( be_it  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_end != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( out_it  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( out_end != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( min_turn_id >= ( NS(particle_index_t) )0u );

    out_it = out_it + out_buffer_index_offset;

    if( ( uintptr_t )out_end >= ( uintptr_t )out_it )
    {
        success = 0;
    }
    else
    {
        return success;
    }

    for( ; be_it != be_end ; ++be_it )
    {
        NS(object_type_id_t) const type_id = NS(Object_get_type_id)( be_it );
        uintptr_t const addr = ( uintptr_t )NS(Object_get_begin_addr)( be_it );

        if( ( type_id == NS(OBJECT_TYPE_BEAM_MONITOR) ) && ( addr != ZERO ) )
        {
            ptr_beam_monitor_t monitor = ( ptr_beam_monitor_t )addr;

            nturn_t const    nn = NS(BeamMonitor_get_num_stores)( monitor );
            nturn_t const start = NS(BeamMonitor_get_start)( monitor );

            if( ( nn > ( nturn_t )0u ) &&
                ( start >= first_turn_id ) && ( out_it != out_end ) )
            {
                ptr_particles_t particles = ( ptr_particles_t
                    )NS(BufferIndex_get_const_particles)( out_it );

                buf_size_t const num_stored_particles =
                    NS(Particles_get_num_of_particles)( particles );

                mon_index_t const min_particle_id =
                    NS(BeamMonitor_get_min_particle_id)( monitor );

                mon_index_t const max_particle_id =
                    NS(BeamMonitor_get_max_particle_id)( monitor );

                buf_size_t const stored_particles_per_turn =
                    ( max_particle_id >= min_particle_id )
                        ? ( buf_size_t )(
                            max_particle_id - min_particle_id  + 1u )
                        : ZERO;

                if( ( nn > 0 ) && ( stored_particles_per_turn > ZERO ) &&
                    ( particles != SIXTRL_NULLPTR ) &&
                    ( ( stored_particles_per_turn * ( buf_size_t )nn ) <=
                        num_stored_particles ) )
                {
                    NS(BeamMonitor_set_out_address)(
                        monitor, NS(Object_get_begin_addr)( out_it++ ) );
                }
                else if( ( nn > 0 ) && ( stored_particles_per_turn > ZERO ) )
                {
                    success = -1;
                    break;
                }
            }
            else if( out_it == out_end )
            {
                success = -1;
                break;
            }
        }
    }

    return success;
}


SIXTRL_INLINE int NS(BeamMonitor_assign_managed_output_buffer_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT output_buffer,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t) const out_buffer_index_offset,
    NS(buffer_size_t) const slot_size,
    SIXTRL_ARGPTR_DEC NS(ctrl_debug_flag_t)* ptr_debug_flag )
{
    typedef NS(ctrl_debug_flag_t)   flag_t;
    typedef NS(buffer_size_t)               buf_size_t;

    flag_t debug_flag = ( flag_t )0u;

    int const success = NS(BeamMonitor_assign_managed_output_buffer)(
        beam_elements, output_buffer, min_turn_id, out_buffer_index_offset,
            slot_size );

    if( ( success != 0 ) && ( ptr_debug_flag != SIXTRL_NULLPTR ) )
    {
        if( slot_size == ( buf_size_t )0u )   debug_flag |= ( flag_t )0x0001;
        if( beam_elements == SIXTRL_NULLPTR ) debug_flag |= ( flag_t )0x0002;

        if( NS(ManagedBuffer_needs_remapping)( beam_elements, slot_size ) )
            debug_flag |= ( flag_t )0x0004;

        if( output_buffer == SIXTRL_NULLPTR ) debug_flag |= ( flag_t )0x0008;

        if( NS(ManagedBuffer_needs_remapping)( output_buffer, slot_size ) )
            debug_flag |= ( flag_t )0x0010;

        if( min_turn_id < ( NS(particle_index_t) )0 )
            debug_flag |= ( flag_t )0x0020;

        if( NS(ManagedBuffer_get_num_objects)( output_buffer, slot_size ) <
                out_buffer_index_offset )
        {
            debug_flag |= ( flag_t )0x0040;
        }

        if( debug_flag == ( flag_t )0u ) debug_flag |= ( flag_t )0x0080;
        *ptr_debug_flag |= debug_flag;
    }

    return success;
}

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_BE_MONITOR_OUTPUT_BUFFER_H__ */

/* end: sixtracklib/common/be_monitor/output.h */
