#include "sixtracklib/common/output/elem_by_elem_config.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/output/output_buffer.h"
#include "sixtracklib/common/output/elem_by_elem_output_buffer.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"

SIXTRL_HOST_FN int NS(ElemByElemConfig_init)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT conf,
    NS(elem_by_elem_order_t) const order,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(particle_index_t) min_turn, NS(particle_index_t) max_turn )
{
    NS(particle_index_t) const start_elem_id = ( NS(particle_index_t) )0u;
    NS(particle_index_t) min_part_id, max_part_id, min_elem_id, max_elem_id,
                         min_turn_id, max_turn_id;

    int success = NS(OutputBuffer_get_min_max_attributes)( p, belements,
        &min_part_id, &max_part_id, &min_elem_id, &max_elem_id, &min_turn_id,
            &max_turn_id, SIXTRL_NULLPTR, start_elem_id );

    if( success == 0 )
    {
        success = NS(ElemByElemConfig_init_detailed)( conf, order, min_part_id,
            max_part_id, min_elem_id, max_elem_id, min_turn, max_turn, true );
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
