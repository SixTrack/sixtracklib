#include "sixtracklib/common/output/elem_by_elem_config.h"

#include "sixtracklib/common/particles.h"

SIXTRL_HOST_FN int NS(ElemByElemConfig_init)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    NS(elem_by_elem_order_t) const order,
    SIXTRL_BUFFER_ARGPTR_DEC
        const NS(Buffer) *const beam_elements_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC
        const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_index_t) const min_turn, NS(particle_index_t) const max_turn )
{
    typedef NS(particle_num_elements_t) num_elem_t;
    typedef NS(particle_index_t)        index_t;

    int success = NS(Particles_get_min_max_particle_id)( particles );
    num_elem_t const

    if( success == 0 )
    {

    }

    return success;
}

SIXTRL_HOST_FN int NS(ElemByElemConfig_assign_particles_out_buffer)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const out_buffer_index_offset )
{

}
