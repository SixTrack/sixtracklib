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
    typedef NS(particle_index_t)        index_t;

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
        success = NS(ElemByElemConfig_init_detailed)( config, order,
            min_particle_id, max_particle_id,
            min_at_element_id, max_at_element_id,
            min_turn, max_turn );
    }

    return success;
}

/* end: sixtracklib/common/output/elem_by_elem_config.c */
