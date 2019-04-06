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

int NS(ElemByElemConfig_calculate_output_buffer_params)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t) const dump_elem_by_elem_turns,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
    NS(buffer_size_t) const out_buffer_slot_size )
{
    NS(particle_index_t) const start_elem_id = (  NS(particle_index_t) )0u;
    NS(particle_index_t) min_part_id, max_part_id, min_elem_id, max_elem_id,
                         min_turn_id, max_turn_id;

    int success = NS(OutputBuffer_get_min_max_attributes)( p, belements,
        &min_part_id, &max_part_id, &min_elem_id, &max_elem_id, &min_turn_id,
            &max_turn_id, SIXTRL_NULLPTR, start_elem_id );

    if( success == 0 )
    {
        success = NS(ElemByElemConfig_calculate_output_buffer_params_detailed)(
            min_part_id, max_part_id, min_elem_id, max_elem_id, min_turn_id,
                max_turn_id, dump_elem_by_elem_turns, ptr_num_objects,
                    ptr_num_slots, ptr_num_data_ptrs, ptr_num_garbage,
                        out_buffer_slot_size );
    }

    return success;
}

int NS(ElemByElemConfig_calculate_output_buffer_param_for_particle_sets)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t) const dump_elem_by_elem_turns,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
    NS(buffer_size_t) const out_buffer_slot_size )
{
    NS(particle_index_t) const start_elem_id = (  NS(particle_index_t) )0u;
    NS(particle_index_t) min_part_id, max_part_id, min_elem_id, max_elem_id,
                         min_turn_id, max_turn_id;

    int success = NS(OutputBuffer_get_min_max_attributes_on_particle_sets)(
        pbuffer, num_particle_sets, indices_begin, belements,
        &min_part_id, &max_part_id, &min_elem_id, &max_elem_id, &min_turn_id,
            &max_turn_id, SIXTRL_NULLPTR, start_elem_id );

    if( success == 0 )
    {
        success = NS(ElemByElemConfig_calculate_output_buffer_params_detailed)(
            min_part_id, max_part_id, min_elem_id, max_elem_id, min_turn_id,
                max_turn_id, dump_elem_by_elem_turns, ptr_num_objects,
                    ptr_num_slots, ptr_num_data_ptrs, ptr_num_garbage,
                        out_buffer_slot_size );
    }

    return success;
}

int NS(ElemByElemConfig_calculate_output_buffer_params_detailed)(
    NS(particle_index_t) const min_part_id,
    NS(particle_index_t) const max_part_id,
    NS(particle_index_t) const min_elem_id,
    NS(particle_index_t) const max_elem_id,
    NS(particle_index_t) const min_turn_id,
    NS(particle_index_t) const max_turn_id,
    NS(buffer_size_t) const dump_elem_by_elem_turns,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
    NS(buffer_size_t) const out_buffer_slot_size )
{
    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(particle_index_t)    index_t;

    int success = -1;

    SIXTRL_STATIC_VAR buf_size_t const   ZERO = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR buf_size_t const    ONE = ( buf_size_t )1u;
    SIXTRL_STATIC_VAR index_t const INV_INDEX = ( index_t )-1;

    buf_size_t num_objects  = ZERO;
    buf_size_t num_slots    = ZERO;
    buf_size_t num_dataptrs = ZERO;
    buf_size_t num_garbage  = ZERO;

    bool const input_param_valid = ( ( min_part_id <= max_part_id ) &&
        ( min_turn_id  <= max_turn_id ) && ( min_part_id > INV_INDEX ) &&
        ( min_turn_id  > INV_INDEX ) );

    if( ( dump_elem_by_elem_turns > ZERO ) && ( input_param_valid ) )
    {
        index_t max_elem_by_elem_turn_id =
            min_turn_id + dump_elem_by_elem_turns;

        if( max_elem_by_elem_turn_id < max_turn_id )
        {
            max_elem_by_elem_turn_id = max_turn_id;
        }

        if( min_turn_id < max_elem_by_elem_turn_id )
        {
            buf_size_t const store_num_particles =
                NS(ElemByElemConfig_get_stored_num_particles_detailed)(
                    min_part_id, max_part_id, min_elem_id, max_elem_id,
                        min_turn_id, max_elem_by_elem_turn_id );

            if( store_num_particles > ZERO )
            {
                num_objects  = ONE;
                num_dataptrs = NS(Particles_get_num_dataptrs)( SIXTRL_NULLPTR );

                num_slots =
                    NS(Particles_get_required_num_slots_on_managed_buffer)(
                        store_num_particles, out_buffer_slot_size );
            }

            success = 0;
        }
    }
    else if( input_param_valid )
    {
        success = 0;
    }

    if( success == 0 )
    {
        if(  ptr_num_objects != SIXTRL_NULLPTR )
        {
            *ptr_num_objects  = num_objects;
        }

        if(  ptr_num_slots != SIXTRL_NULLPTR )
        {
            *ptr_num_slots  = num_slots;
        }

        if(  ptr_num_data_ptrs != SIXTRL_NULLPTR )
        {
            *ptr_num_data_ptrs = num_dataptrs;
        }

        if(  ptr_num_garbage != SIXTRL_NULLPTR )
        {
            *ptr_num_garbage  = num_garbage;
        }
    }

    return success;
}

int NS(ElemByElemConfig_prepare_output_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t) const dump_elem_by_elem_turns,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset )
{
    NS(particle_index_t) const start_elem_id = (  NS(particle_index_t) )0u;
    NS(particle_index_t) min_part_id, max_part_id, min_elem_id, max_elem_id,
                         min_turn_id, max_turn_id;

    int success = NS(OutputBuffer_get_min_max_attributes)( p, belements,
        &min_part_id, &max_part_id, &min_elem_id, &max_elem_id, &min_turn_id,
            &max_turn_id, SIXTRL_NULLPTR, start_elem_id );

    if( success == 0 )
    {
        NS(particle_index_t) const max_elem_by_elem_turn_id =
            ( max_turn_id + dump_elem_by_elem_turns ) -
                ( NS(particle_index_t) )1;

        SIXTRL_ASSERT( max_elem_by_elem_turn_id >= min_turn_id );

        success = NS(ElemByElemConfig_prepare_output_buffer_detailed)(
            output_buffer, min_part_id, max_part_id, min_elem_id,
                max_elem_id, min_turn_id, max_elem_by_elem_turn_id,
                    ptr_index_offset );
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(ElemByElemConfig_prepare_output_buffer_for_particle_sets)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t) const dump_elem_by_elem_turns,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset )
{
    NS(particle_index_t) const start_elem_id = (  NS(particle_index_t) )0u;
    NS(particle_index_t) min_part_id, max_part_id, min_elem_id, max_elem_id,
                         min_turn_id, max_turn_id;

    int success = NS(OutputBuffer_get_min_max_attributes_on_particle_sets)(
        pbuffer, num_particle_sets, indices_begin, belements,
        &min_part_id, &max_part_id, &min_elem_id, &max_elem_id, &min_turn_id,
            &max_turn_id, SIXTRL_NULLPTR, start_elem_id );

    if( success == 0 )
    {
        NS(particle_index_t) const max_elem_by_elem_turn_id =
            ( max_turn_id + dump_elem_by_elem_turns ) -
                ( NS(particle_index_t) )1;

        SIXTRL_ASSERT( max_elem_by_elem_turn_id >= min_turn_id );

        success = NS(ElemByElemConfig_prepare_output_buffer_detailed)(
            output_buffer, min_part_id, max_part_id, min_elem_id,
                max_elem_id, min_turn_id, max_elem_by_elem_turn_id,
                    ptr_index_offset );
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(ElemByElemConfig_prepare_output_buffer_detailed)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*SIXTRL_RESTRICT out_buffer,
    NS(particle_index_t) const min_part_id,
    NS(particle_index_t) const max_part_id,
    NS(particle_index_t) const min_elem_id,
    NS(particle_index_t) const max_elem_id,
    NS(particle_index_t) const min_turn_id,
    NS(particle_index_t) const max_elem_by_elem_turn_id,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset )
{
    int success = -1;

    typedef NS(buffer_size_t)    buf_size_t;
    typedef NS(particle_index_t) index_t;

    SIXTRL_STATIC_VAR index_t const ZERO = ( index_t )0u;

    buf_size_t const initial_out_buffer_index_offset =
        NS(Buffer_get_num_of_objects)( out_buffer );

    if( ( out_buffer != SIXTRL_NULLPTR ) &&
        ( !NS(Buffer_needs_remapping)( out_buffer ) ) &&
        ( min_part_id >= ZERO ) &&
        ( min_part_id <= max_part_id ) &&
        ( min_elem_id  >= ZERO ) && ( min_elem_id <= max_elem_id ) &&
        ( min_turn_id >= ZERO ) &&
        ( min_turn_id <= max_elem_by_elem_turn_id ) )
    {
        buf_size_t const max_num_objects =
            NS(Buffer_get_max_num_of_objects)( out_buffer );

        buf_size_t const max_num_slots =
            NS(Buffer_get_max_num_of_slots)( out_buffer );

        buf_size_t const max_num_dataptrs =
            NS(Buffer_get_max_num_of_dataptrs)( out_buffer );

        buf_size_t const max_num_garbage =
            NS(Buffer_get_max_num_of_garbage_ranges)( out_buffer );

        buf_size_t num_objects  = NS(Buffer_get_num_of_objects)( out_buffer );
        buf_size_t num_slots    = NS(Buffer_get_num_of_slots)( out_buffer );
        buf_size_t num_dataptrs = NS(Buffer_get_num_of_dataptrs)( out_buffer );

        buf_size_t const num_garbage =
            NS(Buffer_get_num_of_garbage_ranges)( out_buffer );

        buf_size_t const store_num_particles =
            NS(ElemByElemConfig_get_stored_num_particles_detailed)(
                min_part_id, max_part_id, min_elem_id, max_elem_id, min_turn_id,
                    max_elem_by_elem_turn_id );

        buf_size_t const delta_num_objects = ( buf_size_t )1u;
        buf_size_t const delta_num_slots   =
            NS(Particles_get_required_num_slots)( out_buffer,
                store_num_particles );

        buf_size_t const delta_num_dataptrs =
            NS(Particles_get_required_num_dataptrs)( out_buffer,
                store_num_particles );

        num_objects  += delta_num_objects;
        num_slots    += delta_num_slots;
        num_dataptrs += delta_num_dataptrs;

        if( ( delta_num_objects > ZERO ) &&
            ( num_objects <= max_num_objects ) &&
            ( delta_num_slots > ZERO ) && ( num_slots <= max_num_slots ) &&
            ( delta_num_dataptrs > ZERO ) &&
            ( num_dataptrs <= max_num_dataptrs ) &&
            ( num_garbage  <= max_num_garbage  ) )
        {
            success = 0;
        }
        else if( ( delta_num_objects  > ZERO ) &&
                 ( delta_num_slots    > ZERO ) &&
                 ( delta_num_dataptrs > ZERO ) &&
                 ( ( num_objects  >= max_num_objects  ) ||
                   ( num_slots    >= max_num_slots    ) ||
                   ( num_dataptrs >= max_num_dataptrs ) ) )
        {
            success = NS(Buffer_reserve)( out_buffer,
                num_objects, num_slots, num_dataptrs, num_garbage );
        }


        if( success == 0 )
        {
            success = ( SIXTRL_NULLPTR != NS(Particles_new)(
                out_buffer, store_num_particles ) ) ? 0 : -1;
        }

        if( ( ptr_index_offset != SIXTRL_NULLPTR ) && ( success == 0 ) )
        {
            SIXTRL_ASSERT( NS(Buffer_get_num_of_objects)( out_buffer ) ==
                ( initial_out_buffer_index_offset + ( buf_size_t )1u ) );

            *ptr_index_offset = initial_out_buffer_index_offset;
        }
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(ElemByElemConfig_prepare_output_buffer_from_conf)(
    SIXTRL_BE_ARGPTR_DEC struct NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*SIXTRL_RESTRICT output_buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset )
{
    int success = -1;

    SIXTRL_ASSERT( output_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( output_buffer ) );

    if( config != SIXTRL_NULLPTR )
    {
        success = NS(ElemByElemConfig_prepare_output_buffer_detailed)(
            output_buffer,
            NS(ElemByElemConfig_get_min_particle_id)( config ),
            NS(ElemByElemConfig_get_max_particle_id)( config ),
            NS(ElemByElemConfig_get_min_element_id)( config ),
            NS(ElemByElemConfig_get_max_element_id)( config ),
            NS(ElemByElemConfig_get_min_turn)( config ),
            NS(ElemByElemConfig_get_max_turn)( config ), ptr_index_offset );
    }

    return success;
}

/* end: sixtracklib/common/output/elem_by_elem_output_buffer.c */
