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
    typedef NS(particle_num_elements_t) num_elem_t;
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
        success = NS(Particles_init_detailed)( config, order,
            min_particle_id, max_particle_id, min_at_element_id,
            max_at_element_id, min_turn, max_turn, max_turn );
    }

    return success;
}

SIXTRL_HOST_FN int NS(ElemByElemConfig_assign_particles_out_buffer)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT conf,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const out_buffer_index_offset )
{
    int success = -1;

    typedef NS(particle_num_elements_t)                     num_elem_t;
    typedef NS(particle_index_t)                            index_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Object) const*     ptr_object_t;
    typedef SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)*       ptr_particles_t;
    typedef NS(buffer_addr_t)                               address_t;

    ptr_object_t ptr_object = NS(Buffer_get_const_object)(
        output_buffer, out_buffer_index_offset );

    address_t const out_addr = NS(Object_get_begin_addr)( ptr_object );
    NS(object_type_id_t) const type_id = NS(Object_get_type_id)( ptr_object );

    if( ( conf != SIXTRL_NULLPTR ) && ( type_id == NS(OBJECT_TYPE_PARTICLE) ) &&
        ( ptr_object != SIXTRL_NULLPTR ) && ( out_addr != ( address_t )0u ) )
    {
        ptr_particles_t out_particles = ( ptr_particles_t )( uintptr_t
            )NS(Object_get_begin_addr)( ptr_object );

        num_elem_t const required_num_out_particles =
            NS(ElemByElemConfig_get_out_store_num_particles)( conf );

        num_elem_t const available_num_out_particles=
            NS(Particles_get_num_of_particles)( out_particles );

        if( ( required_num_out_particles >= ( num_elem_t )0u ) &&
            ( required_num_out_particles <= available_num_out_particles ) )
        {
            NS(ElemByElemConfig_set_output_store_address)( conf, out_addr );
            success = 0;
        }
    }

    return success;
}

/* end: sixtracklib/common/output/elem_by_elem_config.c */
