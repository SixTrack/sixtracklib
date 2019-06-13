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

int NS(BeamMonitor_insert_end_of_turn_monitors_at_pos)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements,
    NS(be_monitor_turn_t) const turn_by_turn_start,
    NS(be_monitor_turn_t) const num_turn_by_turn_turns,
    NS(be_monitor_turn_t) const target_num_turns,
    NS(be_monitor_turn_t) const skip_turns,
    NS(buffer_size_t) const insert_at_index )
{
    if( insert_at_index == ( NS(buffer_size_t) )0u )
    {
        return NS(BeamMonitor_insert_end_of_turn_monitors)(
            belements, turn_by_turn_start, num_turn_by_turn_turns,
            target_num_turns, skip_turns, SIXTRL_NULLPTR );
    }
    else if( insert_at_index <= NS(Buffer_get_num_of_objects)( belements ) )
    {
        SIXTRL_BUFFER_ARGPTR_DEC NS(Object)* prev_node =
            NS(Buffer_get_objects_begin)( belements );

        SIXTRL_ASSERT( prev_node != SIXTRL_NULLPTR );
        prev_node = prev_node + insert_at_index;

        return NS(BeamMonitor_insert_end_of_turn_monitors)(
            belements, turn_by_turn_start, num_turn_by_turn_turns,
            target_num_turns, skip_turns, prev_node );
    }

    return -1;
}

int NS(BeamMonitor_insert_end_of_turn_monitors)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements,
    NS(be_monitor_turn_t) const turn_by_turn_start,
    NS(be_monitor_turn_t) const num_turn_by_turn_turns,
    NS(be_monitor_turn_t) const target_num_turns,
    NS(be_monitor_turn_t) const skip_turns,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT prev_node )
{
    typedef SIXTRL_BUFFER_ARGPTR_DEC NS(Object)*        ptr_obj_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor)*  ptr_beam_monitor_t;
    typedef NS(be_monitor_turn_t)                       nturn_t;

    int success = -1;

    ptr_obj_t idx_begin = NS(Buffer_get_objects_begin)( belements );
    ptr_obj_t idx_end   = NS(Buffer_get_objects_end)( belements );

    SIXTRL_STATIC_VAR nturn_t const ONE_TURN = ( nturn_t )1u;
    SIXTRL_STATIC_VAR nturn_t const ZERO     = ( nturn_t )0u;

    /* HACK: Currently, there is no clean solution for "inserting" a node
     * at a specific position in a buffer. Thus, so far the only easy solution
     * is to add the beam monitors to the end.
     * Thus, if the prev_node != idx_end, we can currently not perform the
     * action and return false immediately. */
    if( prev_node != idx_end )
    {
        return success;
    }

    if( ( belements != SIXTRL_NULLPTR ) &&
        ( idx_begin != SIXTRL_NULLPTR ) && ( idx_end != SIXTRL_NULLPTR ) &&
        ( ( intptr_t )( idx_end - idx_begin ) >= ( intptr_t )0 ) &&
        ( ( prev_node == SIXTRL_NULLPTR ) ||
          ( ( ( intptr_t )( prev_node - idx_begin ) >= ( intptr_t )0 ) &&
            ( ( intptr_t )( idx_end - prev_node   ) >= ( intptr_t )0 ) ) ) )
    {
        nturn_t const output_turn_start =
            turn_by_turn_start + num_turn_by_turn_turns;

        success = 0;

        if( num_turn_by_turn_turns > ZERO )
        {
            uintptr_t const offset = ( uintptr_t )( prev_node - idx_begin );

            /* TODO: Change the call to NS(BeamElement_new)() below to the
             * appropiate insert function and remove the ASSERT statement
             * as soon as an insert API is available for NS(Buffer)!
             * Inspect also the update of prev_node below!!!
             */
            SIXTRL_ASSERT( prev_node == idx_end );

            ptr_beam_monitor_t mon = NS(BeamMonitor_new)( belements );
            if( mon != SIXTRL_NULLPTR )
            {
                NS(BeamMonitor_set_num_stores)( mon, num_turn_by_turn_turns );
                NS(BeamMonitor_set_start)( mon, turn_by_turn_start );
                NS(BeamMonitor_set_skip)( mon, ONE_TURN );
                NS(BeamMonitor_set_is_rolling)( mon, false );

                idx_begin = NS(Buffer_get_objects_begin)( belements );
                idx_end   = NS(Buffer_get_objects_end)( belements );
                prev_node = idx_begin + ( offset + ( uintptr_t )1u );
            }
            else
            {
                success = -1;
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

            ptr_beam_monitor_t mon = NS(BeamMonitor_new)( belements );
            if( mon != SIXTRL_NULLPTR )
            {
                nturn_t skip = skip_turns;
                nturn_t num_stores = target_num_turns - output_turn_start;

                if( skip < ONE_TURN )
                {
                    skip = ONE_TURN;
                }

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
                success = -1;
            }
        }
    }

    return success;
}

/* end: sixtracklib/common/be_monitor/be_monitor.c */
