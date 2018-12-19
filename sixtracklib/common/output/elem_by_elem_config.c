#include "sixtracklib/common/output/elem_by_elem_config.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/be_monitor/be_monitor.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"

SIXTRL_HOST_FN int NS(ElemByElemConfig_init)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT config,
    NS(elem_by_elem_order_t) const order,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer)
        *const SIXTRL_RESTRICT beam_elements_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles)
        *const SIXTRL_RESTRICT particles,
    NS(particle_index_t) const min_turn, NS(particle_index_t) const max_turn )
{
    typedef NS(particle_index_t) index_t;
    typedef NS(buffer_size_t)    buf_size_t;

    index_t min_particle_id     = ( index_t )0u;
    index_t max_particle_id     = ( index_t )0u;

    index_t min_at_element_id   = ( index_t )0u;
    index_t max_at_element_id   = ( index_t )0u;

    index_t temp_min_at_turn_id = ( index_t )0u;
    index_t temp_max_at_turn_id = ( index_t )0u;

    int success = NS(Particles_find_min_max_attributes)(
        particles, &min_particle_id, &max_particle_id, &min_at_element_id,
        &max_at_element_id, &temp_min_at_turn_id, &temp_max_at_turn_id );

    if( success == 0 )
    {
        buf_size_t num_elements_total = ( buf_size_t )0u;

        index_t temp_min_at_element_id = min_at_element_id;
        index_t temp_max_at_element_id = max_at_element_id;

        SIXTRL_ASSERT( temp_min_at_turn_id >= min_turn );
        SIXTRL_ASSERT( temp_max_at_turn_id <= max_turn );

        success = NS(ElemByElemConfig_get_min_max_element_id_from_buffer)(
            beam_elements_buffer, &temp_min_at_element_id,
            &temp_max_at_element_id, &num_elements_total, 0 );

        if( success == 0 )
        {
            if( min_at_element_id > temp_min_at_element_id )
            {
                min_at_element_id = temp_min_at_element_id;
            }

            if( max_at_element_id < temp_max_at_element_id )
            {
                max_at_element_id = temp_max_at_element_id;
            }
        }
    }

    if( success == 0 )
    {
        success = NS(ElemByElemConfig_init_detailed)( config, order,
            min_particle_id, max_particle_id,
            min_at_element_id, max_at_element_id,
            min_turn, max_turn );
    }

    return success;
}

SIXTRL_HOST_FN int NS(ElemByElemConfig_assign_output_buffer)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* output_buffer,
    NS(buffer_size_t) const out_buffer_index_offset )
{
    return NS(ElemByElemConfig_assign_managed_output_buffer)(
        config, NS(Buffer_get_data_begin)( output_buffer ),
        out_buffer_index_offset,
        NS(Buffer_get_slot_size)( output_buffer ) );
}

/* end: sixtracklib/common/output/elem_by_elem_config.c */
