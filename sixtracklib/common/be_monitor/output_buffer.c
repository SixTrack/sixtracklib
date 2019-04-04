#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_monitor/output_buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
    #include "sixtracklib/common/buffer/buffer_garbage.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/be_monitor/be_monitor.h"
    #include "sixtracklib/common/output/output_buffer.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

int NS(BeamMonitor_calculate_output_buffer_params)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
    NS(buffer_size_t) const out_buffer_slot_size )
{
    NS(particle_index_t) min_part_id = ( NS(particle_index_t) )0u;
    NS(particle_index_t) max_part_id = ( NS(particle_index_t) )0u;
    NS(particle_index_t) min_turn_id = ( NS(particle_index_t) )0u;

    int success = NS(Particles_find_min_max_attributes)( p, &min_part_id,
        &max_part_id, SIXTRL_NULLPTR, SIXTRL_NULLPTR, &min_turn_id,
            SIXTRL_NULLPTR );

    if( success == 0 )
    {
        success = NS(BeamMonitor_calculate_output_buffer_params_detailed)(
            belements, min_part_id, max_part_id, min_turn_id, ptr_num_objects,
                ptr_num_slots, ptr_num_data_ptrs, ptr_num_garbage,
                    out_buffer_slot_size );
    }

    return success;
}

int NS(BeamMonitor_calculate_output_buffer_params_for_particle_sets)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT pb,
    NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
    NS(buffer_size_t) const out_buffer_slot_size )
{
    NS(particle_index_t) min_part_id = ( NS(particle_index_t) )0u;
    NS(particle_index_t) max_part_id = ( NS(particle_index_t) )0u;
    NS(particle_index_t) min_turn_id = ( NS(particle_index_t) )0u;

    int success = NS(Particles_buffer_get_min_max_attributes_of_particles_set)(
        pb, num_particle_sets, indices_begin, &min_part_id, &max_part_id,
            SIXTRL_NULLPTR, SIXTRL_NULLPTR, &min_turn_id, SIXTRL_NULLPTR );

    if( success == 0 )
    {
        success = NS(BeamMonitor_calculate_output_buffer_params_detailed)(
            belements, min_part_id, max_part_id, min_turn_id, ptr_num_objects,
                ptr_num_slots, ptr_num_data_ptrs, ptr_num_garbage,
                    out_buffer_slot_size );
    }

    return success;
}

int NS(BeamMonitor_calculate_output_buffer_params_detailed)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(particle_index_t) const min_part_id,
    NS(particle_index_t) const max_part_id,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
    NS(buffer_size_t) const out_buffer_slot_size )
{
    typedef NS(buffer_size_t)                               buf_size_t;
    typedef NS(particle_index_t)                            index_t;
    typedef NS(be_monitor_turn_t)                           nturn_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*  ptr_obj_t;
    typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*           ptr_beam_monitor_t;
    typedef NS(buffer_addr_t)                               address_t;
    typedef NS(object_type_id_t)                            type_id_t;

    int success = -1;

    SIXTRL_STATIC_VAR buf_size_t const ZERO      = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR index_t    const IZERO     = ( index_t )0u;
    SIXTRL_STATIC_VAR address_t  const ADDR_ZERO = ( address_t )0u;
    SIXTRL_STATIC_VAR nturn_t    const ONE_TURN  = ( nturn_t )1u;

    buf_size_t num_objects  = ZERO;
    buf_size_t num_slots    = ZERO;
    buf_size_t num_dataptrs = ZERO;
    buf_size_t num_garbage  = ZERO;

    nturn_t const first_turn_id = ( nturn_t )min_turn_id;

    SIXTRL_ASSERT( belements != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( belements ) );

    if( ( min_turn_id >= IZERO ) && ( min_part_id >= IZERO ) &&
        ( min_part_id <= max_part_id ) )
    {
        ptr_obj_t be_it  = NS(Buffer_get_const_objects_begin)( belements );
        ptr_obj_t be_end = NS(Buffer_get_const_objects_end)( belements );

        buf_size_t const num_particles_to_store = ( buf_size_t )(
            max_part_id - min_part_id + ( buf_size_t )1u );

        success = 0;

        for( ; be_it != be_end ; ++be_it )
        {
            type_id_t const type_id = NS(Object_get_type_id)( be_it );

            if( ( type_id == NS(OBJECT_TYPE_BEAM_MONITOR) ) &&
                ( NS(Object_get_begin_addr)( be_it ) > ADDR_ZERO ) )
            {
                ptr_beam_monitor_t monitor = ( ptr_beam_monitor_t )( uintptr_t
                    )NS(Object_get_begin_addr)( be_it );

                buf_size_t required_store_particles = ZERO;

                nturn_t const skip  = NS(BeamMonitor_get_skip)( monitor );
                nturn_t const start = NS(BeamMonitor_get_start)( monitor );
                nturn_t const num_stores =
                    NS(BeamMonitor_get_num_stores)( monitor );

                if( ( start >= first_turn_id ) ||
                    ( NS(BeamMonitor_is_rolling)( monitor ) ) )
                {
                    required_store_particles = num_particles_to_store *
                        ( buf_size_t )num_stores;
                }
                else if( ( start < first_turn_id ) && ( skip >= ONE_TURN ) )
                {
                    nturn_t const already_tracked = first_turn_id - start;
                    nturn_t const already_stored  =
                        ONE_TURN + already_tracked / skip;

                    if( num_stores > already_stored )
                    {
                        required_store_particles = num_particles_to_store *
                            ( buf_size_t)( num_stores - already_stored );
                    }
                }

                if( required_store_particles > ZERO )
                {
                    ++num_objects;

                    num_dataptrs += NS(Particles_get_num_dataptrs)(
                        SIXTRL_NULLPTR );

                    num_slots +=
                        NS(Particles_get_required_num_slots_on_managed_buffer)(
                            required_store_particles, out_buffer_slot_size );
                }
            }
        }
    }

    if( success == 0 )
    {
        if(  ptr_num_objects != SIXTRL_NULLPTR )
        {
            *ptr_num_objects  = num_objects;
        }

        if(  ptr_num_slots != SIXTRL_NULLPTR )
        {
            *ptr_num_slots = num_slots;
        }

        if(  ptr_num_data_ptrs != SIXTRL_NULLPTR )
        {
            *ptr_num_data_ptrs  = num_dataptrs;
        }

        if(  ptr_num_garbage != SIXTRL_NULLPTR )
        {
            *ptr_num_garbage  = num_garbage;
        }
    }

    return success;
}

int NS(BeamMonitor_prepare_output_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset )
{
    NS(particle_index_t) min_part_id = ( NS(particle_index_t) )0u;
    NS(particle_index_t) max_part_id = ( NS(particle_index_t) )0u;
    NS(particle_index_t) min_turn_id = ( NS(particle_index_t) )0u;

    int success = NS(Particles_get_min_max_attributes)( p, &min_part_id,
        &max_part_id, SIXTRL_NULLPTR, SIXTRL_NULLPTR, &min_turn_id,
            SIXTRL_NULLPTR );

    if( success == 0 )
    {
        success = NS(BeamMonitor_prepare_output_buffer_detailed)(
            belements, output_buffer, min_part_id, max_part_id,
                min_turn_id, ptr_index_offset );
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(BeamMonitor_prepare_output_buffer_for_particle_sets)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT indices_begin,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset )
{
    NS(particle_index_t) min_part_id = ( NS(particle_index_t) )0u;
    NS(particle_index_t) max_part_id = ( NS(particle_index_t) )0u;
    NS(particle_index_t) min_turn_id = ( NS(particle_index_t) )0u;

    int success = NS(Particles_buffer_get_min_max_attributes_of_particles_set)(
        pbuffer, num_particle_sets, indices_begin, &min_part_id, &max_part_id,
            SIXTRL_NULLPTR, SIXTRL_NULLPTR, &min_turn_id, SIXTRL_NULLPTR );

    if( success == 0 )
    {
        success = NS(BeamMonitor_prepare_output_buffer_detailed)(
            belements, output_buffer, min_part_id, max_part_id,
                min_turn_id, ptr_index_offset );
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(BeamMonitor_prepare_output_buffer_detailed)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(particle_index_t)  const min_part_id,
    NS(particle_index_t)  const max_part_id,
    NS(particle_index_t)  const min_turn_id,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset )
{
    int success = -1;

    typedef NS(buffer_size_t)                         buf_size_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*  ptr_obj_t;
    typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*     ptr_beam_monitor_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Particles)*  ptr_particles_t;
    typedef NS(buffer_addr_t)                         address_t;
    typedef NS(be_monitor_turn_t)                     nturn_t;
    typedef NS(particle_index_t)                      index_t;
    typedef NS(object_type_id_t)                      type_id_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO      = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR nturn_t    const TZERO     = ( nturn_t )0u;
    SIXTRL_STATIC_VAR index_t    const IZERO     = ( index_t )0u;
    SIXTRL_STATIC_VAR address_t  const ADDR_ZERO = ( address_t )0u;

    ptr_obj_t be_begin = NS(Buffer_get_objects_begin)( belements );
    ptr_obj_t be_end   = NS(Buffer_get_objects_end)( belements );

    buf_size_t const out_buffer_index_offset =
        NS(Buffer_get_num_of_objects)( output_buffer );

    buf_size_t num_beam_monitors = ZERO;
    buf_size_t num_particles_per_turn = ZERO;

    nturn_t const first_turn_id = ( nturn_t )min_turn_id;

    ptr_obj_t be_it = be_begin;

    SIXTRL_ASSERT( belements != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( belements ) );

    SIXTRL_ASSERT( output_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( output_buffer ) );

    if( ( min_turn_id < IZERO ) || ( min_part_id < IZERO ) ||
        ( max_part_id < min_part_id ) )
    {
        return success;
    }

    num_particles_per_turn = ( buf_size_t )(
        1u + max_part_id - min_part_id );

    SIXTRL_ASSERT( num_particles_per_turn > ZERO );
    success = 0;

    for( ; be_it != be_end ; ++be_it )
    {
        type_id_t const type_id = NS(Object_get_type_id)( be_it );

        if( ( type_id == NS(OBJECT_TYPE_BEAM_MONITOR) ) &&
            ( NS(Object_get_begin_addr)( be_it ) > ADDR_ZERO ) )
        {
            ptr_beam_monitor_t monitor = ( ptr_beam_monitor_t )( uintptr_t
                )NS(Object_get_begin_addr)( be_it );

            nturn_t const start = NS(BeamMonitor_get_start)( monitor );
            nturn_t const nstores = NS(BeamMonitor_get_num_stores)( monitor );

            if( ( start >= first_turn_id ) && ( nstores > TZERO ) )
            {
                buf_size_t const num_particles_to_store =
                    num_particles_per_turn * nstores;

                ptr_particles_t particles =
                    NS(Particles_new)( output_buffer, num_particles_to_store );

                if( particles != SIXTRL_NULLPTR )
                {
                    NS(BeamMonitor_set_min_particle_id)(
                        monitor, min_part_id );

                    NS(BeamMonitor_set_max_particle_id)(
                        monitor, max_part_id );

                    NS(BeamMonitor_set_out_address)(
                        monitor, ( address_t )0u );

                    ++num_beam_monitors;
                }
                else
                {
                    success = -1;
                    break;
                }
            }
        }
    }

    SIXTRL_ASSERT( ( success != 0 ) || ( NS(Buffer_get_num_of_objects)(
        output_buffer ) == ( num_beam_monitors + out_buffer_index_offset ) ) );

    if( ( ptr_index_offset != SIXTRL_NULLPTR ) && ( success == 0 ) )
    {
        *ptr_index_offset = out_buffer_index_offset;
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */


int NS(BeamMonitor_assign_output_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t) const num_elem_by_elem_turns  )
{
    int success = -1;

    typedef NS(buffer_size_t) buf_size_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR buf_size_t const ONE  = ( buf_size_t )1u;

    if( ( num_elem_by_elem_turns > ZERO ) &&
        ( NS(Buffer_get_num_of_objects)( belements_buffer ) > ONE ) )
    {
        success = NS(BeamMonitor_assign_output_buffer_from_offset)(
            belements_buffer, out_buffer, min_turn_id, ONE );
    }
    else if( num_elem_by_elem_turns == ZERO )
    {
        success = NS(BeamMonitor_assign_output_buffer_from_offset)(
            belements_buffer, out_buffer, min_turn_id, ZERO );
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(BeamMonitor_assign_output_buffer_from_offset)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t) const out_buffer_index_offset )
{
    return NS(BeamMonitor_assign_managed_output_buffer)(
        NS(Buffer_get_data_begin)( belements_buffer ),
        NS(Buffer_get_data_begin)( out_buffer ),
        min_turn_id, out_buffer_index_offset,
        NS(Buffer_get_slot_size)( out_buffer ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(BeamMonitor_setup_for_particles_all)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p )
{
    return NS(BeamMonitor_setup_managed_buffer_for_particles_all)(
        NS(Buffer_get_data_begin)( belements_buffer ),
        p, NS(Buffer_get_slot_size)( belements_buffer ) );
}

/* end: sixtracklib/common/be_monitor/output_buffer.c */
