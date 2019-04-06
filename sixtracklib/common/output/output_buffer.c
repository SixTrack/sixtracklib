#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/output/output_buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/beam_elements.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
    #include "sixtracklib/common/buffer/buffer_garbage.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/be_monitor/be_monitor.h"
    #include "sixtracklib/common/be_monitor/output_buffer.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/output/elem_by_elem_output_buffer.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */


SIXTRL_HOST_FN NS(output_buffer_flag_t)
NS(OutputBuffer_required_for_tracking)(
    const NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
    const NS(Buffer) *const SIXTRL_RESTRICT belem_buffer,
    NS(buffer_size_t) const dump_elem_by_elem_turns )
{
    typedef NS(buffer_size_t)           buf_size_t;
    typedef NS(object_type_id_t)        type_id_t;
    typedef NS(Object) const*           obj_iter_t;
    typedef NS(buffer_addr_t)           address_t;
    typedef NS(BeamMonitor) const*      ptr_monitor_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR address_t const ADDR_ZERO = ( address_t )0u;

    NS(output_buffer_flag_t) flags = NS(OUTPUT_BUFFER_NONE);

    obj_iter_t be_it = NS(Buffer_get_const_objects_begin)( belem_buffer );

    buf_size_t const num_belems =
        NS(Buffer_get_num_of_objects)( belem_buffer );

    SIXTRL_ASSERT( NS(BeamElements_is_beam_elements_buffer)( belem_buffer ) );

    SIXTRL_ASSERT( NS(Buffer_is_particles_buffer)( particles_buffer ) );
    SIXTRL_ASSERT( NS(Particles_buffer_get_total_num_of_particles)(
        particles_buffer ) > ( NS(particle_num_elements_t) )0u );

    if( dump_elem_by_elem_turns > ZERO )
    {
        flags |= NS(OUTPUT_BUFFER_ELEM_BY_ELEM);
    }

    if( ( be_it != SIXTRL_NULLPTR ) && ( num_belems > ZERO ) )
    {
        buf_size_t const be_monitor_size = sizeof( NS(BeamMonitor) );

        obj_iter_t be_end = be_it - ( ptrdiff_t )1;
        be_it = be_it + ( ptrdiff_t )( num_belems - ( buf_size_t )1u );

        for( ; be_it != be_end ; --be_it )
        {
            buf_size_t const obj_size = NS(Object_get_size)( be_it );
            address_t  const addr = NS(Object_get_begin_addr)( be_it );
            type_id_t  const type = NS(Object_get_type_id)( be_it );

            if( ( addr != ADDR_ZERO ) && ( obj_size >= be_monitor_size ) &&
                ( type == NS(OBJECT_TYPE_BEAM_MONITOR) ) )
            {
                ptr_monitor_t be_monitor = ( ptr_monitor_t )( uintptr_t )addr;

                if( ( be_monitor != SIXTRL_NULLPTR ) &&
                    ( NS(BeamMonitor_get_num_stores)( be_monitor ) > 0 ) )
                {
                    flags |= NS(OUTPUT_BUFFER_BEAM_MONITORS);
                    break;
                }
            }
        }
    }

    return flags;
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(OutputBuffer_prepare)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t) const dump_elem_by_elem_turns,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_elem_by_elem_out_index_offset,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_beam_monitor_out_index_offset,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id )
{
    NS(particle_index_t) const start_elem_id = ( NS(particle_index_t) )0u;
    NS(particle_index_t) min_part_id, max_part_id, min_elem_id, max_elem_id,
                         min_turn_id, max_turn_id;

    int success = NS(OutputBuffer_get_min_max_attributes)( p, belements,
        &min_part_id, &max_part_id, &min_elem_id, &max_elem_id, &min_turn_id,
            &max_turn_id, SIXTRL_NULLPTR, start_elem_id );

    if( success == 0 )
    {
        success = NS(OutputBuffer_prepare_detailed)( belements, output_buffer,
            min_part_id, max_part_id, min_elem_id, max_elem_id, min_turn_id,
                max_turn_id, dump_elem_by_elem_turns,
                    ptr_elem_by_elem_out_index_offset,
                        ptr_beam_monitor_out_index_offset, SIXTRL_NULLPTR );

        if( ( success == 0 ) && ( ptr_min_turn_id != SIXTRL_NULLPTR ) )
        {
            *ptr_min_turn_id = min_turn_id;
        }
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(OutputBuffer_prepare_for_particle_sets)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t) const dump_elem_by_elem_turns,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_elem_by_elem_out_index_offset,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_beam_monitor_out_index_offset,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id )
{
    NS(particle_index_t) const start_elem_id = ( NS(particle_index_t) )0u;
    NS(particle_index_t) min_part_id, max_part_id, min_elem_id, max_elem_id,
                         min_turn_id, max_turn_id;

    int success = NS(OutputBuffer_get_min_max_attributes_on_particle_sets)(
        pbuffer, num_particle_sets, indices_begin, belements_buffer,
        &min_part_id, &max_part_id, &min_elem_id, &max_elem_id, &min_turn_id,
            &max_turn_id, SIXTRL_NULLPTR, start_elem_id );

    if( success == 0 )
    {
        success = NS(OutputBuffer_prepare_detailed)( belements_buffer,
            output_buffer, min_part_id, max_part_id, min_elem_id, max_elem_id,
                min_turn_id, max_turn_id, dump_elem_by_elem_turns,
                    ptr_elem_by_elem_out_index_offset,
                        ptr_beam_monitor_out_index_offset, SIXTRL_NULLPTR );

        if( ( success == 0 ) && ( ptr_min_turn_id != SIXTRL_NULLPTR ) )
        {
            *ptr_min_turn_id = min_turn_id;
        }
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(OutputBuffer_prepare_detailed)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(particle_index_t)     const min_part_id,
    NS(particle_index_t)     const max_part_id,
    NS(particle_index_t)     const min_elem_id,
    NS(particle_index_t)     const max_elem_id,
    NS(particle_index_t)     const min_turn_id,
    NS(particle_index_t)     const max_turn_id,
    NS(buffer_size_t)        const dump_elem_by_elem_turns,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_elem_by_elem_out_index_offset,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_beam_monitor_out_index_offset,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT
        ptr_max_elem_by_elem_turn_id )
{
    int success = -1;

    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(particle_index_t)    index_t;

    if( ( belements != SIXTRL_NULLPTR ) &&
        ( !NS(Buffer_needs_remapping)( belements ) ) &&
        ( output_buffer != SIXTRL_NULLPTR ) &&
        ( !NS(Buffer_needs_remapping)( output_buffer ) ) )
    {
        buf_size_t elem_by_elem_index_offset = ( buf_size_t )0u;
        buf_size_t beam_monitor_index_offset = ( buf_size_t )0u;
        index_t    max_elem_by_elem_turn_id  = min_turn_id;

        success = 0;

        if( dump_elem_by_elem_turns > ( buf_size_t )0u )
        {
            max_elem_by_elem_turn_id += dump_elem_by_elem_turns;

            if( max_elem_by_elem_turn_id < max_turn_id )
            {
                max_elem_by_elem_turn_id = max_turn_id;
            }

            success = NS(ElemByElemConfig_prepare_output_buffer_detailed)(
                output_buffer, min_part_id, max_part_id,
                min_elem_id, max_elem_id, min_turn_id,
                max_elem_by_elem_turn_id, &elem_by_elem_index_offset );
        }

        if( success == 0 )
        {
            success = NS(BeamMonitor_prepare_output_buffer_detailed)(
                belements, output_buffer, min_part_id, max_part_id,
                min_turn_id, &beam_monitor_index_offset );
        }

        if( success == 0 )
        {
            if( ptr_elem_by_elem_out_index_offset != SIXTRL_NULLPTR )
            {
                *ptr_elem_by_elem_out_index_offset = elem_by_elem_index_offset;
            }

            if( ptr_beam_monitor_out_index_offset != SIXTRL_NULLPTR )
            {
                *ptr_beam_monitor_out_index_offset = beam_monitor_index_offset;
            }

            if( ptr_max_elem_by_elem_turn_id != SIXTRL_NULLPTR )
            {
                *ptr_max_elem_by_elem_turn_id = max_elem_by_elem_turn_id;
            }
        }
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_HOST_FN int NS(OutputBuffer_calculate_output_buffer_params)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t) const dump_elem_by_elem_turns,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
    NS(buffer_size_t) const out_buffer_slot_size )
{
    NS(particle_index_t) const start_elem_id = ( NS(particle_index_t) )0u;
    NS(particle_index_t) min_part_id, max_part_id, min_elem_id, max_elem_id,
                         min_turn_id, max_turn_id;

    int success = NS(OutputBuffer_get_min_max_attributes)( p, belements,
        &min_part_id, &max_part_id, &min_elem_id, &max_elem_id, &min_turn_id,
            &max_turn_id, SIXTRL_NULLPTR, start_elem_id );

    if( success == 0 )
    {
        success = NS(OutputBuffer_calculate_output_buffer_params_detailed)(
            belements, min_part_id, max_part_id, min_elem_id, max_elem_id,
                min_turn_id, max_turn_id, dump_elem_by_elem_turns,
                    ptr_num_objects, ptr_num_slots, ptr_num_data_ptrs,
                        ptr_num_garbage, out_buffer_slot_size );
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_HOST_FN int
NS(OutputBuffer_calculate_output_buffer_params_for_particles_sets)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t) const dump_elem_by_elem_turns,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
    NS(buffer_size_t) const out_buffer_slot_size )
{
    int success = -1;

    NS(particle_index_t) const start_elem_id = ( NS(particle_index_t) )0u;
    NS(particle_index_t) min_part_id, max_part_id, min_elem_id, max_elem_id,
                         min_turn_id, max_turn_id;

    if( 0 == NS(OutputBuffer_get_min_max_attributes_on_particle_sets)(
            pbuffer, num_particle_sets, indices_begin, belements, &min_part_id,
                &max_part_id, &min_elem_id, &max_elem_id, &min_turn_id,
                    &max_turn_id, SIXTRL_NULLPTR, start_elem_id ) )
    {
        success = NS(OutputBuffer_calculate_output_buffer_params_detailed)(
            belements, min_part_id, max_part_id, min_elem_id, max_elem_id,
                min_turn_id, max_turn_id, dump_elem_by_elem_turns,
                    ptr_num_objects, ptr_num_slots, ptr_num_data_ptrs,
                        ptr_num_garbage, out_buffer_slot_size );
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_HOST_FN int NS(OutputBuffer_calculate_output_buffer_params_detailed)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(particle_index_t) const min_part_id,
    NS(particle_index_t) const max_part_id,
    NS(particle_index_t) const min_elem_id,
    NS(particle_index_t) const max_elem_id,
    NS(particle_index_t) const min_turn_id,
    NS(particle_index_t) const max_turn_id,
    NS(buffer_size_t) const dump_elem_by_elem_turns,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
    NS(buffer_size_t) const out_buffer_slot_size )
{
    int success = -1;

    typedef NS(buffer_size_t) buf_size_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    buf_size_t elem_by_elem_num_slots     = ZERO;
    buf_size_t elem_by_elem_num_objects   = ZERO;
    buf_size_t elem_by_elem_num_data_ptrs = ZERO;
    buf_size_t elem_by_elem_num_garbage   = ZERO;

    buf_size_t beam_monitor_num_slots     = ZERO;
    buf_size_t beam_monitor_num_objects   = ZERO;
    buf_size_t beam_monitor_num_data_ptrs = ZERO;
    buf_size_t beam_monitor_num_garbage   = ZERO;

    SIXTRL_ASSERT( belements != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( belements ) );

    if( dump_elem_by_elem_turns > ZERO )
    {
        success = NS(ElemByElemConfig_calculate_output_buffer_params_detailed)(
            min_part_id, max_part_id, min_elem_id, max_elem_id, min_turn_id,
                max_turn_id, dump_elem_by_elem_turns, &elem_by_elem_num_objects,
                    &elem_by_elem_num_slots, &elem_by_elem_num_data_ptrs,
                        &elem_by_elem_num_garbage, out_buffer_slot_size );
    }
    else
    {
        success = 0;
    }

    if( success == 0 )
    {
        success = NS(BeamMonitor_calculate_output_buffer_params_detailed)(
            belements, min_part_id, max_part_id, min_turn_id,
                &beam_monitor_num_objects,   &beam_monitor_num_slots,
                    &beam_monitor_num_data_ptrs, &beam_monitor_num_garbage,
                        out_buffer_slot_size );
    }

    if( success == 0 )
    {
        if( ptr_num_objects != SIXTRL_NULLPTR )
        {
            *ptr_num_objects = elem_by_elem_num_objects +
                               beam_monitor_num_objects;
        }

        if( ptr_num_slots != SIXTRL_NULLPTR )
        {
            *ptr_num_slots = elem_by_elem_num_slots + beam_monitor_num_slots;
        }

        if( ptr_num_data_ptrs != SIXTRL_NULLPTR )
        {
            *ptr_num_data_ptrs = elem_by_elem_num_data_ptrs +
                                 beam_monitor_num_data_ptrs;
        }

        if( ptr_num_garbage != SIXTRL_NULLPTR )
        {
            *ptr_num_garbage = elem_by_elem_num_garbage +
                               beam_monitor_num_garbage;
        }
    }

    return success;
}

/* end: sixtracklib/common/output/be_monitor.c */
