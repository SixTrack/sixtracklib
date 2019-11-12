#include "sixtracklib/common/output/elem_by_elem_config.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/output/output_buffer.h"
#include "sixtracklib/common/output/elem_by_elem_output_buffer.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"

bool NS(ElemByElemConfig_is_active_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    return NS(ElemByElemConfig_is_active)( config );
}

NS(particle_num_elements_t) NS(ElemByElemConfig_get_out_store_num_particles_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    return NS(ElemByElemConfig_get_out_store_num_particles)( config );
}

NS(particle_num_elements_t) NS(ElemByElemConfig_get_num_particles_to_store_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    return NS(ElemByElemConfig_get_num_particles_to_store)( config );
}

NS(particle_num_elements_t) NS(ElemByElemConfig_get_num_turns_to_store_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    return NS(ElemByElemConfig_get_num_turns_to_store)( config );
}

NS(particle_num_elements_t) NS(ElemByElemConfig_get_num_elements_to_store_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    return NS(ElemByElemConfig_get_num_elements_to_store)( config );
}

NS(particle_index_t) NS(ElemByElemConfig_get_min_particle_id_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    return NS(ElemByElemConfig_get_min_particle_id)( config );
}

NS(particle_index_t) NS(ElemByElemConfig_get_max_particle_id_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    return NS(ElemByElemConfig_get_max_particle_id)( config );
}

NS(particle_index_t) NS(ElemByElemConfig_get_min_element_id_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    return NS(ElemByElemConfig_get_min_element_id)( config );
}

NS(particle_index_t) NS(ElemByElemConfig_get_max_element_id_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    return NS(ElemByElemConfig_get_max_element_id)( config );
}

NS(particle_index_t) NS(ElemByElemConfig_get_min_turn_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    return NS(ElemByElemConfig_get_min_turn)( config );
}

NS(particle_index_t) NS(ElemByElemConfig_get_max_turn_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    return NS(ElemByElemConfig_get_max_turn)( config );
}

bool NS(ElemByElemConfig_is_rolling_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    return NS(ElemByElemConfig_is_rolling)( config );
}

NS(elem_by_elem_order_t) NS(ElemByElemConfig_get_order_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    return NS(ElemByElemConfig_get_order)( config );
}

NS(elem_by_elem_out_addr_t) NS(ElemByElemConfig_get_output_store_address_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    return NS(ElemByElemConfig_get_output_store_address)( config );
}

/* -------------------------------------------------------------------------- */

NS(particle_num_elements_t)
NS(ElemByElemConfig_get_particles_store_index_details_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    NS(particle_index_t) const particle_id,
    NS(particle_index_t) const at_element_id,
    NS(particle_index_t) const at_turn )
{
    return NS(ElemByElemConfig_get_particles_store_index_details)(
        config, particle_id, at_element_id, at_turn );
}

NS(particle_num_elements_t)
NS(ElemByElemConfig_get_particles_store_index_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles)
        *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index )
{
    return NS(ElemByElemConfig_get_particles_store_index)(
        config, particles, particle_index );
}

NS(particle_index_t)
NS(ElemByElemConfig_get_particle_id_from_store_index_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    NS(particle_num_elements_t) const out_store_index )
{
    return NS(ElemByElemConfig_get_particle_id_from_store_index)(
        config, out_store_index );
}

NS(particle_index_t)
NS(ElemByElemConfig_get_at_element_id_from_store_index_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    NS(particle_num_elements_t) const out_store_index )
{
    return NS(ElemByElemConfig_get_at_element_id_from_store_index)(
        config, out_store_index );
}

NS(particle_index_t)
NS(ElemByElemConfig_get_at_turn_from_store_index_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    NS(particle_num_elements_t) const out_store_index )
{
    return NS(ElemByElemConfig_get_at_turn_from_store_index)(
        config, out_store_index );
}

/* ------------------------------------------------------------------------ */

NS(arch_status_t) NS(ElemByElemConfig_init_detailed_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    NS(elem_by_elem_order_t) const order,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    bool const is_rolling_flag )
{
    return NS(ElemByElemConfig_init_detailed)(
        config, order, min_particle_id, max_particle_id, min_element_id,
            max_element_id, min_turn, max_turn, is_rolling_flag );
}

/* ------------------------------------------------------------------------- */

SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
NS(ElemByElemConfig_preset_ext)( SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config )
{
    return NS(ElemByElemConfig_preset)( config );
}

void NS(ElemByElemConfig_clear_ext)( SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config )
{
    NS(ElemByElemConfig_clear)( config );
}

void NS(ElemByElemConfig_set_order_ext)( SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    NS(elem_by_elem_order_t) const order )
{
    NS(ElemByElemConfig_set_order)( config, order );
}

void NS(ElemByElemConfig_set_is_rolling_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    bool const is_rolling_flag )
{
    NS(ElemByElemConfig_set_is_rolling)( config, is_rolling_flag );
}

void NS(ElemByElemConfig_set_output_store_address_ext)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    NS(elem_by_elem_out_addr_t) const out_address )
{
    NS(ElemByElemConfig_set_output_store_address)( config, out_address );
}

/* ------------------------------------------------------------------------- */

NS(ElemByElemConfig)* NS(ElemByElemConfig_create)( void )
{
    return NS(ElemByElemConfig_preset)( ( NS(ElemByElemConfig)* )malloc(
        sizeof( NS(ElemByElemConfig) ) ) );
}

void NS(ElemByElemConfig_delete)(
    NS(ElemByElemConfig)* SIXTRL_RESTRICT elem_by_elem_config )
{
    free( elem_by_elem_config );
}

NS(arch_status_t) NS(ElemByElemConfig_init)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const
        SIXTRL_RESTRICT particle_set,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements_buffer,
    NS(particle_index_t) const start_elem_id,
    NS(particle_index_t) const until_turn_elem_by_elem )
{
    typedef NS(particle_index_t) _index_t;

    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);
    SIXTRL_STATIC_VAR _index_t const ZERO = ( _index_t )0u;

    _index_t min_part_id, max_part_id, min_elem_id, max_elem_id,
             min_turn_id, max_turn_id;

    NS(ElemByElemConfig_preset)( config );

    if( ( config == SIXTRL_NULLPTR ) ||
        ( until_turn_elem_by_elem < ZERO ) || ( start_elem_id < ZERO ) )
    {
        return status;
    }

    status = NS(OutputBuffer_get_min_max_attributes)( particle_set,
        beam_elements_buffer, &min_part_id, &max_part_id, &min_elem_id,
            &max_elem_id, &min_turn_id, &max_turn_id,
                SIXTRL_NULLPTR, start_elem_id );

    if( status == NS(ARCH_STATUS_SUCCESS) )
    {
        SIXTRL_ASSERT( particle_set != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( beam_elements_buffer != SIXTRL_NULLPTR );

        if( ( min_part_id <= max_part_id   ) && ( min_part_id >= ZERO ) &&
            ( min_elem_id <= max_elem_id   ) &&
            ( min_elem_id >= start_elem_id ) &&
            ( min_turn_id <= max_turn_id   ) && ( min_turn_id >= ZERO ) )
        {
            if( until_turn_elem_by_elem >= min_elem_id )
            {
                if( until_turn_elem_by_elem > min_elem_id )
                {
                    SIXTRL_ASSERT( until_turn_elem_by_elem > ZERO );
                    max_turn_id = until_turn_elem_by_elem - ( _index_t )1u;
                }
            }

            status = NS(ElemByElemConfig_init_detailed)(
                config, NS(ELEM_BY_ELEM_ORDER_DEFAULT),
                    min_part_id, max_part_id, min_elem_id, max_elem_id,
                        min_turn_id, max_turn_id, true );
        }
    }

    return status;
}

NS(arch_status_t) NS(ElemByElemConfig_init_on_particle_sets)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const*
        SIXTRL_RESTRICT pset_indices_begin,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements_buffer,
    NS(particle_index_t) const start_elem_id,
    NS(particle_index_t) const until_turn_elem_by_elem )
{
    typedef NS(particle_index_t) _index_t;

    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);
    SIXTRL_STATIC_VAR _index_t const ZERO = ( _index_t )0u;

    _index_t min_part_id, max_part_id, min_elem_id, max_elem_id,
             min_turn_id, max_turn_id;

    NS(ElemByElemConfig_preset)( config );

    if( ( config == SIXTRL_NULLPTR ) ||
        ( until_turn_elem_by_elem < ZERO ) || ( start_elem_id < ZERO ) )
    {
        return status;
    }

    status = NS(OutputBuffer_get_min_max_attributes_on_particle_sets)(
        particles_buffer, num_particle_sets, pset_indices_begin,
        beam_elements_buffer, &min_part_id, &max_part_id, &min_elem_id,
            &max_elem_id, &min_turn_id, &max_turn_id,
                SIXTRL_NULLPTR, start_elem_id );

    if( status == NS(ARCH_STATUS_SUCCESS) )
    {
        SIXTRL_ASSERT( particles_buffer != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( num_particle_sets > ( ( NS(buffer_size_t) )0u ) );
        SIXTRL_ASSERT( pset_indices_begin != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( beam_elements_buffer != SIXTRL_NULLPTR );

        if( ( min_part_id <= max_part_id   ) && ( min_part_id >= ZERO ) &&
            ( min_elem_id <= max_elem_id   ) &&
            ( min_elem_id >= start_elem_id ) &&
            ( min_turn_id <= max_turn_id   ) && ( min_turn_id >= ZERO ) )
        {
            if( until_turn_elem_by_elem >= min_elem_id )
            {
                if( until_turn_elem_by_elem > min_elem_id )
                {
                    SIXTRL_ASSERT( until_turn_elem_by_elem > ZERO );
                    max_turn_id = until_turn_elem_by_elem - ( _index_t )1u;
                }
            }

            status = NS(ElemByElemConfig_init_detailed)(
                config, NS(ELEM_BY_ELEM_ORDER_DEFAULT),
                    min_part_id, max_part_id, min_elem_id, max_elem_id,
                        min_turn_id, max_turn_id, true );
        }
    }

    return status;
}

NS(arch_status_t) NS(ElemByElemConfig_assign_output_buffer)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* output_buffer,
    NS(buffer_size_t) const out_buffer_index_offset )
{
    return NS(ElemByElemConfig_assign_managed_output_buffer)(
        config, NS(Buffer_get_data_begin)( output_buffer ),
        out_buffer_index_offset, NS(Buffer_get_slot_size)( output_buffer ) );
}

NS(arch_status_t) NS(ElemByElemConfig_assign_output_buffer_debug)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* output_buffer,
    NS(buffer_size_t) const out_buffer_index_offset,
    SIXTRL_ARGPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_dbg_register )
{
    return NS(ElemByElemConfig_assign_managed_output_buffer_debug)(
        config, NS(Buffer_get_data_begin)( output_buffer ),
        out_buffer_index_offset, NS(Buffer_get_slot_size)( output_buffer ),
        ptr_dbg_register );
}

/* end: sixtracklib/common/output/elem_by_elem_config.c */
