#include "sixtracklib/common/be_monitor/be_monitor.h"

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

NS(arch_status_t) NS(BeamMonitor_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( monitor ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( offsets != SIXTRL_NULLPTR ) && ( slot_size > ZERO ) &&
        ( max_num_offsets > ( buf_size_t )0u ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, offsets, max_num_offsets, ZERO );
    }

    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

NS(arch_status_t) NS(BeamMonitor_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( monitor ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( sizes != SIXTRL_NULLPTR ) && ( slot_size > ZERO ) &&
        ( max_num_sizes > ( buf_size_t )0u ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, sizes, max_num_sizes, ZERO );
    }

    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

NS(arch_status_t) NS(BeamMonitor_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( monitor ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( counts != SIXTRL_NULLPTR ) && ( slot_size > ZERO ) &&
        ( max_num_counts > ( buf_size_t )0u ) )
    {
        SIXTRACKLIB_SET_VALUES( buf_size_t, counts, max_num_counts, ZERO );
    }

    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(object_type_id_t) NS(BeamMonitor_type_id_ext)( void ) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_type_id)();
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(BeamMonitor_are_present_in_obj_index_range_ext)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_end
) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_are_present_in_obj_index_range)( obj_begin, obj_end );
}

bool NS(BeamMonitor_are_present_in_managed_buffer_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_are_present_in_managed_buffer)(
        buffer_begin, slot_size );
}

bool NS(BeamMonitor_are_present_in_buffer_ext)( SIXTRL_BUFFER_ARGPTR_DEC
    const NS(Buffer) *const SIXTRL_RESTRICT buffer ) SIXTRL_NOEXCEPT
{
   return NS(BeamMonitor_are_present_in_buffer)( buffer );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(buffer_size_t) NS(BeamMonitor_num_monitors_in_obj_index_range_ext)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_end
) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_num_monitors_in_obj_index_range)(
        obj_begin, obj_end );
}

NS(buffer_size_t) NS(BeamMonitor_num_monitors_in_managed_buffer_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_num_monitors_in_managed_buffer)(
        buffer_begin, slot_size );
}

NS(buffer_size_t) NS(BeamMonitor_num_monitors_in_buffer_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT
        buffer ) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_num_monitors_in_buffer)( buffer );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(buffer_size_t) NS(BeamMonitor_monitor_indices_from_obj_index_range_ext)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_end,
    NS(buffer_size_t) const start_index ) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_monitor_indices_from_obj_index_range)(
        indices_begin, max_num_of_indices, obj_begin, obj_end, start_index );
}

NS(buffer_size_t) NS(BeamMonitor_monitor_indices_from_managed_buffer_ext)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_monitor_indices_from_managed_buffer)( indices_begin,
        max_num_of_indices, buffer_begin, slot_size );
}

NS(buffer_size_t) NS(BeamMonitor_monitor_indices_from_buffer_ext)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer
) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_monitor_indices_from_buffer)(
        indices_begin, max_num_of_indices, buffer );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(arch_status_t) NS(BeamMonitor_reset_ext)( SIXTRL_BE_ARGPTR_DEC
    NS(BeamMonitor)* SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_reset)( monitor );
}

NS(arch_status_t) NS(BeamMonitor_reset_all_in_obj_index_range_ext)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT obj_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT obj_end
) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_reset_all_in_obj_index_range)( obj_begin, obj_end );
}

NS(arch_status_t) NS(BeamMonitor_reset_all_in_managed_buffer_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_reset_all_in_managed_buffer)(
        buffer_begin, slot_size );
}

NS(arch_status_t) NS(BeamMonitor_reset_all_in_buffer_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
        SIXTRL_RESTRICT buffer ) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_reset_all_in_buffer)( buffer );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(BeamMonitor_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        requ_dataptrs ) SIXTRL_NOEXCEPT
{
    NS(BeamMonitor) temp;
    NS(arch_status_t) const status = NS(BeamMonitor_clear)( &temp );
    NS(buffer_size_t) const ndataptrs = NS(BeamMonitor_num_dataptrs)( &temp );
    SIXTRL_ASSERT( ndataptrs == ( NS(buffer_size_t) )0u );

    return ( ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) &&
             ( ndataptrs == ( NS(buffer_size_t) )0 ) &&
             ( NS(Buffer_can_add_object)( buffer, sizeof( NS(BeamMonitor) ),
                ndataptrs, SIXTRL_NULLPTR, SIXTRL_NULLPTR, requ_objects,
                    requ_slots, requ_dataptrs ) ) );
}

SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* NS(BeamMonitor_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* added_elem = SIXTRL_NULLPTR;

    NS(BeamMonitor) temp;
    NS(arch_status_t) const status = NS(BeamMonitor_clear)( &temp );
    NS(buffer_size_t) const ndataptrs = NS(BeamMonitor_num_dataptrs)( &temp );

    if( ( buffer != SIXTRL_NULLPTR ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0 ) &&
        ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) )
    {
        added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* )( uintptr_t
            )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, &temp,
                sizeof( NS(BeamMonitor) ), NS(BeamMonitor_type_id)(), ndataptrs,
                    SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
    }

    return added_elem;
}

SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* NS(BeamMonitor_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(be_monitor_turn_t) const num_stores, NS(be_monitor_turn_t) const start,
    NS(be_monitor_turn_t) const skip,  NS(be_monitor_addr_t) const out_address,
    NS(be_monitor_index_t) const min_particle_id,
    NS(be_monitor_index_t) const max_particle_id,
    bool const is_rolling, bool const is_turn_ordered )
{
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* added_elem = SIXTRL_NULLPTR;
    NS(buffer_size_t) ndataptrs = ( NS(buffer_size_t) )0;

    NS(BeamMonitor) temp;
    NS(arch_status_t) status =
        NS(BeamMonitor_set_num_stores)( &temp, num_stores );
    status |= NS(BeamMonitor_set_start)( &temp, start );
    status |= NS(BeamMonitor_set_skip)( &temp, skip );
    status |= NS(BeamMonitor_set_out_address)( &temp, out_address );
    status |= NS(BeamMonitor_set_min_particle_id)( &temp, min_particle_id );
    status |= NS(BeamMonitor_set_max_particle_id)( &temp, max_particle_id );
    status |= NS(BeamMonitor_set_is_rolling)( &temp, is_rolling );
    status |= NS(BeamMonitor_set_is_turn_ordered)( &temp, is_turn_ordered );
    ndataptrs = NS(BeamMonitor_num_dataptrs)( &temp );

    if( ( buffer != SIXTRL_NULLPTR ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0 ) &&
        ( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS ) )
    {
        added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* )( uintptr_t
            )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, &temp,
                sizeof( NS(BeamMonitor) ), NS(BeamMonitor_type_id)(),
                    ndataptrs, SIXTRL_NULLPTR, SIXTRL_NULLPTR,
                        SIXTRL_NULLPTR ) );
    }

    return added_elem;
}

SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* NS(BeamMonitor_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT orig )
{
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* added_elem = SIXTRL_NULLPTR;
    NS(buffer_size_t) const ndataptrs = NS(BeamMonitor_num_dataptrs)( orig );

    if( ( buffer != SIXTRL_NULLPTR ) && ( orig != SIXTRL_NULLPTR ) &&
        ( ndataptrs == ( NS(buffer_size_t) )0u ) )
    {
        added_elem = ( SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* )( uintptr_t
        )NS(Object_get_begin_addr)( NS(Buffer_add_object)( buffer, orig,
            sizeof( NS(BeamMonitor) ), NS(BeamMonitor_type_id)(), ndataptrs,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
    }

    return added_elem;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(arch_status_t) NS(BeamMonitor_insert_end_of_turn_monitors)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(be_monitor_turn_t) const turn_by_turn_start,
    NS(be_monitor_turn_t) const num_turn_by_turn_turns,
    NS(be_monitor_turn_t) const target_num_turns,
    NS(be_monitor_turn_t) const skip_turns,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT prev_node )
{
    typedef SIXTRL_BUFFER_ARGPTR_DEC NS(Object)* ptr_obj_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor)* ptr_beam_monitor_t;
    typedef NS(be_monitor_turn_t) nturn_t;

    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    ptr_obj_t idx_begin = NS(Buffer_get_objects_begin)( buffer );
    ptr_obj_t idx_end   = NS(Buffer_get_objects_end)( buffer );

    SIXTRL_STATIC_VAR nturn_t const ONE_TURN = ( nturn_t )1u;
    SIXTRL_STATIC_VAR nturn_t const ZERO     = ( nturn_t )0u;

    /* HACK: Currently, there is no clean solution for "inserting" a node
     * at a specific position in a buffer. Thus, so far the only easy solution
     * is to add the beam monitors to the end.
     * Thus, if the prev_node != idx_end, we can currently not perform the
     * action and return false immediately. */
    if( prev_node != idx_end ) return status;

    if( ( buffer  != SIXTRL_NULLPTR ) && ( idx_begin != SIXTRL_NULLPTR ) &&
        ( idx_end != SIXTRL_NULLPTR ) &&
        ( ( intptr_t )( idx_end - idx_begin ) >= ( intptr_t )0 ) &&
        ( ( prev_node == SIXTRL_NULLPTR ) ||
          ( ( ( intptr_t )( prev_node - idx_begin ) >= ( intptr_t )0 ) &&
            ( ( intptr_t )( idx_end - prev_node   ) >= ( intptr_t )0 ) ) ) )
    {
        nturn_t const output_turn_start =
            turn_by_turn_start + num_turn_by_turn_turns;

        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;

        if( num_turn_by_turn_turns > ZERO )
        {
            uintptr_t const offset = ( uintptr_t )( prev_node - idx_begin );

            /* TODO: Change the call to NS(BeamElement_new)() below to the
             * appropiate insert function and remove the ASSERT statement
             * as soon as an insert API is available for NS(Buffer)!
             * Inspect also the update of prev_node below!!!
             */
            SIXTRL_ASSERT( prev_node == idx_end );

            ptr_beam_monitor_t mon = NS(BeamMonitor_new)( buffer );
            if( mon != SIXTRL_NULLPTR )
            {
                NS(BeamMonitor_set_num_stores)( mon, num_turn_by_turn_turns );
                NS(BeamMonitor_set_start)( mon, turn_by_turn_start );
                NS(BeamMonitor_set_skip)( mon, ONE_TURN );
                NS(BeamMonitor_set_is_rolling)( mon, false );

                idx_begin = NS(Buffer_get_objects_begin)( buffer );
                idx_end   = NS(Buffer_get_objects_end)( buffer );
                prev_node = idx_begin + ( offset + ( uintptr_t )1u );
            }
            else
            {
                status = ( NS(arch_status_t)
                    )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
            }
        }

        if( target_num_turns > ( output_turn_start ) )
        {
            /* TODO: Change the call to NS(BeamElement_new)() below to the
             * appropiate insert function and remove the ASSERT statement
             * as soon as an insert API is available for NS(Buffer)!
             * Inspect also the update of prev_node below!!!
             */
            SIXTRL_ASSERT( prev_node == idx_end );

            ptr_beam_monitor_t mon = NS(BeamMonitor_new)( buffer );
            if( mon != SIXTRL_NULLPTR )
            {
                nturn_t skip = skip_turns;
                nturn_t num_stores = target_num_turns - output_turn_start;

                if( skip < ONE_TURN ) skip = ONE_TURN;
                if( skip > ONE_TURN )
                {
                    nturn_t const remainder = num_stores % skip;
                    num_stores /= skip;

                    if( remainder != ZERO ) ++num_stores;
                }

                NS(BeamMonitor_set_num_stores)( mon, num_stores );
                NS(BeamMonitor_set_start)( mon, output_turn_start );
                NS(BeamMonitor_set_skip)( mon, skip );
                NS(BeamMonitor_set_is_rolling)( mon, true );
            }
            else
            {
                status = ( NS(arch_status_t)
                    )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
            }
        }
    }

    return status;
}

NS(arch_status_t) NS(BeamMonitor_insert_end_of_turn_monitors_at_pos)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements,
    NS(be_monitor_turn_t) const turn_by_turn_start,
    NS(be_monitor_turn_t) const num_turn_by_turn_turns,
    NS(be_monitor_turn_t) const target_num_turns,
    NS(be_monitor_turn_t) const skip_turns,
    NS(buffer_size_t) const insert_at_index )
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( insert_at_index == ( NS(buffer_size_t) )0u )
    {
        status = NS(BeamMonitor_insert_end_of_turn_monitors)(
            belements, turn_by_turn_start, num_turn_by_turn_turns,
            target_num_turns, skip_turns, SIXTRL_NULLPTR );
    }
    else if( insert_at_index <= NS(Buffer_get_num_of_objects)( belements ) )
    {
        SIXTRL_BUFFER_ARGPTR_DEC NS(Object)* prev_node =
            NS(Buffer_get_objects_begin)( belements );

        SIXTRL_ASSERT( prev_node != SIXTRL_NULLPTR );
        prev_node = prev_node + insert_at_index;

        status = NS(BeamMonitor_insert_end_of_turn_monitors)(
            belements, turn_by_turn_start, num_turn_by_turn_turns,
            target_num_turns, skip_turns, prev_node );
    }

    return status;
}
