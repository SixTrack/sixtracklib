#ifndef SIXTRL_SIXTRACKLIB_COMMON_OUTPUT_ELEM_BY_ELEM_CONFIG_H__
#define SIXTRL_SIXTRACKLIB_COMMON_OUTPUT_ELEM_BY_ELEM_CONFIG_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
    #include "sixtracklib/common/internal/elem_by_elem_config_defines.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

typedef SIXTRL_INT64_T  NS(elem_by_elem_order_int_t);
typedef SIXTRL_UINT64_T NS(elem_by_elem_out_addr_t);

typedef enum NS(elem_by_elem_order_t)
{
    NS(ELEM_BY_ELEM_ORDER_INVALID)              = -1,
    NS(ELEM_BY_ELEM_ORDER_TURN_ELEM_PARTICLES)  =  0,
    NS(ELEM_BY_ELEM_ORDER_DEFAULT)              =  0
}
NS(elem_by_elem_order_t);

typedef struct NS(ElemByElemConfig)
{
    NS(elem_by_elem_order_int_t) order                   SIXTRL_ALIGN( 8 );
    NS(particle_num_elements_t)  num_particles_to_store  SIXTRL_ALIGN( 8 );
    NS(particle_num_elements_t)  num_elements_to_store   SIXTRL_ALIGN( 8 );
    NS(particle_num_elements_t)  num_turns_to_store      SIXTRL_ALIGN( 8 );
    NS(particle_index_t)         min_particle_id         SIXTRL_ALIGN( 8 );
    NS(particle_index_t)         min_element_id          SIXTRL_ALIGN( 8 );
    NS(particle_index_t)         min_turn                SIXTRL_ALIGN( 8 );
    NS(particle_index_t)         max_particle_id         SIXTRL_ALIGN( 8 );
    NS(particle_index_t)         max_element_id          SIXTRL_ALIGN( 8 );
    NS(particle_index_t)         max_turn                SIXTRL_ALIGN( 8 );
    NS(elem_by_elem_out_addr_t)  out_store_addr          SIXTRL_ALIGN( 8 );
}
NS(ElemByElemConfig);

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_num_elements_t)
NS(ElemByElemConfig_get_out_store_num_particles)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config );

SIXTRL_FN SIXTRL_STATIC NS(particle_num_elements_t)
NS(ElemByElemConfig_get_num_particles_to_store)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config );

SIXTRL_FN SIXTRL_STATIC NS(particle_num_elements_t)
NS(ElemByElemConfig_get_num_turns_to_store)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config );

SIXTRL_FN SIXTRL_STATIC NS(particle_num_elements_t)
NS(ElemByElemConfig_get_num_elements_to_store)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config );

SIXTRL_FN SIXTRL_STATIC NS(particle_index_t)
NS(ElemByElemConfig_get_min_particle_id)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config );

SIXTRL_FN SIXTRL_STATIC NS(particle_index_t)
NS(ElemByElemConfig_get_max_particle_id)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config );

SIXTRL_FN SIXTRL_STATIC NS(particle_index_t)
NS(ElemByElemConfig_get_min_element_id)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config );

SIXTRL_FN SIXTRL_STATIC NS(particle_index_t)
NS(ElemByElemConfig_get_max_element_id)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config );

SIXTRL_FN SIXTRL_STATIC NS(particle_index_t)
NS(ElemByElemConfig_get_min_turn)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config );

SIXTRL_FN SIXTRL_STATIC NS(particle_index_t)
NS(ElemByElemConfig_get_max_turn)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config );

SIXTRL_FN SIXTRL_STATIC NS(elem_by_elem_order_t)
NS(ElemByElemConfig_get_order)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config );

SIXTRL_FN SIXTRL_STATIC NS(elem_by_elem_out_addr_t)
NS(ElemByElemConfig_get_output_store_address)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config );

/* -------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_num_elements_t)
NS(ElemByElemConfig_get_particles_store_index_details)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    NS(particle_index_t) const particle_id,
    NS(particle_index_t) const at_element_id,
    NS(particle_index_t) const at_turn );

SIXTRL_FN SIXTRL_STATIC NS(particle_num_elements_t)
NS(ElemByElemConfig_get_particles_store_index)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles)
        *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index );

SIXTRL_FN SIXTRL_STATIC NS(particle_index_t)
NS(ElemByElemConfig_get_particle_id_from_store_index)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    NS(particle_num_elements_t) const out_store_index );

SIXTRL_FN SIXTRL_STATIC NS(particle_index_t)
NS(ElemByElemConfig_get_at_element_id_from_store_index)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    NS(particle_num_elements_t) const out_store_index );

SIXTRL_FN SIXTRL_STATIC NS(particle_index_t)
NS(ElemByElemConfig_get_at_turn_from_store_index)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    NS(particle_num_elements_t) const out_store_index );

/* ------------------------------------------------------------------------ */

SIXTRL_FN SIXTRL_STATIC int NS(ElemByElemConfig_init_detailed)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    NS(elem_by_elem_order_t) const order,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn );

SIXTRL_FN SIXTRL_STATIC int
NS(ElemByElemConfig_assign_managed_output_buffer)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const out_buffer_index_offset,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
NS(ElemByElemConfig)* NS(ElemByElemConfig_preset)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config );

SIXTRL_FN SIXTRL_STATIC void NS(ElemByElemConfig_clear)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config );

SIXTRL_FN SIXTRL_STATIC void NS(ElemByElemConfig_set_order)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    NS(elem_by_elem_order_t) const order );

SIXTRL_FN SIXTRL_STATIC void NS(ElemByElemConfig_set_output_store_address)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    NS(elem_by_elem_out_addr_t) const out_address );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(ElemByElemConfig_init)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    NS(elem_by_elem_order_t) const order,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const beam_elements_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(particle_index_t) const min_turn, NS(particle_index_t) const max_turn );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(ElemByElemConfig_get_num_elem_by_elem_objects)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT belements );

SIXTRL_FN SIXTRL_STATIC int
NS(ElemByElemConfig_get_min_max_element_id_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_element_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_element_id,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT ptr_num_elem_by_elem_objects,
    NS(particle_index_t) const element_id_offset );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(ElemByElemConfig_get_required_num_slots)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        const NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(ElemByElemConfig_get_required_num_dataptrs)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        const NS(ElemByElemConfig) *const SIXTRL_RESTRICT config );

SIXTRL_FN SIXTRL_STATIC NS(object_type_id_t)
NS(ElemByElemConfig_get_type_id)( SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        const NS(ElemByElemConfig) *const SIXTRL_RESTRICT config );

SIXTRL_FN SIXTRL_STATIC bool NS(ElemByElemConfig_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
NS(ElemByElemConfig)* NS(ElemByElemConfig_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
NS(ElemByElemConfig)* NS(ElemByElemConfig_add)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(elem_by_elem_order_t) const order,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
NS(ElemByElemConfig)* NS(ElemByElemConfig_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
     SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(ElemByElemConfig_assign_output_buffer)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* output_buffer,
    NS(buffer_size_t) const out_buffer_index_offset );

#endif /* !defined( _GPUCODE ) */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(ElemByElemConfig_get_num_elem_by_elem_objects_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT belements_buffer,
    NS(buffer_size_t) const slot_size );

 /* ------------------------------------------------------------------------ */
 /*  Implementation of inline functions: */
 /* ------------------------------------------------------------------------ */

SIXTRL_INLINE NS(particle_num_elements_t)
NS(ElemByElemConfig_get_out_store_num_particles)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    SIXTRL_ASSERT( config != SIXTRL_NULLPTR );
    return ( config->num_particles_to_store *
             config->num_elements_to_store *
             config->num_turns_to_store );
}

SIXTRL_INLINE NS(particle_num_elements_t)
NS(ElemByElemConfig_get_num_particles_to_store)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    SIXTRL_ASSERT( config != SIXTRL_NULLPTR );
    return config->num_particles_to_store;
}

SIXTRL_INLINE NS(particle_num_elements_t)
NS(ElemByElemConfig_get_num_turns_to_store)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    SIXTRL_ASSERT( config != SIXTRL_NULLPTR );
    return config->num_turns_to_store;
}

SIXTRL_INLINE NS(particle_num_elements_t)
NS(ElemByElemConfig_get_num_elements_to_store)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    SIXTRL_ASSERT( config != SIXTRL_NULLPTR );
    return config->num_elements_to_store;
}

SIXTRL_INLINE NS(particle_index_t)
NS(ElemByElemConfig_get_min_particle_id)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    SIXTRL_ASSERT( config != SIXTRL_NULLPTR );
    return config->min_particle_id;
}

SIXTRL_INLINE NS(particle_index_t)
NS(ElemByElemConfig_get_max_particle_id)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )

{
    SIXTRL_ASSERT( config != SIXTRL_NULLPTR );
    return config->max_particle_id;
}

SIXTRL_INLINE NS(particle_index_t)
NS(ElemByElemConfig_get_min_element_id)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    SIXTRL_ASSERT( config != SIXTRL_NULLPTR );
    return config->min_element_id;
}

SIXTRL_INLINE NS(particle_index_t)
NS(ElemByElemConfig_get_max_element_id)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    SIXTRL_ASSERT( config != SIXTRL_NULLPTR );
    return config->max_element_id;
}

SIXTRL_INLINE NS(particle_index_t)
NS(ElemByElemConfig_get_min_turn)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    SIXTRL_ASSERT( config != SIXTRL_NULLPTR );
    return config->min_turn;
}

SIXTRL_INLINE NS(particle_index_t)
NS(ElemByElemConfig_get_max_turn)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    SIXTRL_ASSERT( config != SIXTRL_NULLPTR );
    return config->max_turn;
}

SIXTRL_INLINE NS(elem_by_elem_order_t) NS(ElemByElemConfig_get_order)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    SIXTRL_ASSERT( config != SIXTRL_NULLPTR );
    return ( NS(elem_by_elem_order_t) )config->order;
}

SIXTRL_INLINE NS(elem_by_elem_out_addr_t)
NS(ElemByElemConfig_get_output_store_address)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    SIXTRL_ASSERT( config != SIXTRL_NULLPTR );
    return config->out_store_addr;
}

/* -------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_num_elements_t)
NS(ElemByElemConfig_get_particles_store_index_details)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    NS(particle_index_t) const particle_id,
    NS(particle_index_t) const at_element, NS(particle_index_t) const at_turn )
{
    typedef NS(particle_num_elements_t) num_elem_t;
    num_elem_t out_store_index = ( num_elem_t )-1;

    SIXTRL_ASSERT( config != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( particle_id >= ( NS(particle_index_t) )0u );
    SIXTRL_ASSERT( at_element  >= ( NS(particle_index_t) )0u );
    SIXTRL_ASSERT( at_turn     >= ( NS(particle_index_t) )0u );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_min_particle_id)( config ) <=
                   NS(ElemByElemConfig_get_max_particle_id)( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_min_element_id)( config ) <=
                   NS(ElemByElemConfig_get_max_element_id)( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_min_turn)( config ) <=
                   NS(ElemByElemConfig_get_max_turn)( config ) );

    if( ( NS(ElemByElemConfig_get_min_particle_id)( config ) <= particle_id ) &&
        ( NS(ElemByElemConfig_get_max_particle_id)( config ) >= particle_id ) &&
        ( NS(ElemByElemConfig_get_min_element_id)( config )  <= at_element  ) &&
        ( NS(ElemByElemConfig_get_max_element_id)( config )  >= at_element  ) &&
        ( NS(ElemByElemConfig_get_min_turn)( config )        <= at_turn ) &&
        ( NS(ElemByElemConfig_get_max_turn)( config )        >= at_turn ) )
    {
        num_elem_t const particle_id_offset = ( num_elem_t )( particle_id -
            NS(ElemByElemConfig_get_min_particle_id)( config ) );

        num_elem_t const element_id_offset = ( num_elem_t )( at_element -
            NS(ElemByElemConfig_get_min_element_id)( config ) );

        num_elem_t const turn_offset = ( num_elem_t )( at_turn -
            NS(ElemByElemConfig_get_min_turn)( config ) );

        num_elem_t const num_particles_to_store =
            NS(ElemByElemConfig_get_num_particles_to_store)( config );

        num_elem_t const num_elements_to_store =
            NS(ElemByElemConfig_get_num_elements_to_store)( config );

        num_elem_t const num_turns_to_store =
            NS(ElemByElemConfig_get_num_turns_to_store)( config );

        SIXTRL_ASSERT( num_particles_to_store > ( num_elem_t )0u );
        SIXTRL_ASSERT( num_elements_to_store  > ( num_elem_t )0u );
        SIXTRL_ASSERT( num_turns_to_store     > ( num_elem_t )0u );

        switch( NS(ElemByElemConfig_get_order)( config ) )
        {
            case NS(ELEM_BY_ELEM_ORDER_TURN_ELEM_PARTICLES):
            {
                ( void )num_turns_to_store;

                out_store_index  = turn_offset *
                    num_particles_to_store * num_elements_to_store;

                out_store_index += element_id_offset * num_particles_to_store;
                out_store_index += particle_id_offset;

                break;
            }

            default:
            {
                out_store_index = ( num_elem_t )-1;
            }
        };

        SIXTRL_ASSERT(
            ( ( out_store_index >= ( num_elem_t )0u ) && ( out_store_index <
                NS(ElemByElemConfig_get_out_store_num_particles)( config ) )
            ) || ( out_store_index < ( num_elem_t )0u ) );
    }

    return out_store_index;
}


SIXTRL_INLINE NS(particle_num_elements_t)
NS(ElemByElemConfig_get_particles_store_index)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles)
        *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const index )
{
    typedef NS(particle_index_t) index_t;

    index_t const temp_particle_id =
        NS(Particles_get_particle_id_value)( particles, index );

    index_t const particle_id = ( temp_particle_id >= ( index_t )0u )
        ? temp_particle_id : -temp_particle_id;

    return NS(ElemByElemConfig_get_particles_store_index_details)( config,
        particle_id, NS(Particles_get_at_element_id_value)( particles, index ),
        NS(Particles_get_at_turn_value)( particles, index ) );
}


SIXTRL_INLINE NS(particle_index_t)
NS(ElemByElemConfig_get_particle_id_from_store_index)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    NS(particle_num_elements_t) const out_store_index )
{
    typedef NS(particle_num_elements_t) num_elem_t;
    typedef NS(particle_index_t)        index_t;

    index_t particle_id = ( index_t )-1;

    SIXTRL_ASSERT( config != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_min_particle_id)( config ) <=
                   NS(ElemByElemConfig_get_max_particle_id)( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_min_element_id)( config ) <=
                   NS(ElemByElemConfig_get_max_element_id)( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_min_turn)( config ) <=
                   NS(ElemByElemConfig_get_max_turn)( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_particles_to_store)( config ) >=
                   ( num_elem_t )0u );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_elements_to_store)( config ) >=
                   ( num_elem_t )0u );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_turns_to_store)( config ) >=
                   ( num_elem_t )0u );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_particles_to_store)( config ) <
                   NS(ElemByElemConfig_get_out_store_num_particles)( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_elements_to_store)( config ) <
                   NS(ElemByElemConfig_get_out_store_num_particles)( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_turns_to_store)( config ) <
                   NS(ElemByElemConfig_get_out_store_num_particles)( config ) );

    if( ( out_store_index >= ( num_elem_t )0u ) && ( out_store_index <
        ( NS(ElemByElemConfig_get_out_store_num_particles)( config ) ) ) )
    {
        num_elem_t const num_particles_to_store =
            NS(ElemByElemConfig_get_num_particles_to_store)( config );

        num_elem_t const num_elements_to_store =
            NS(ElemByElemConfig_get_num_elements_to_store)( config );

        num_elem_t const num_turns_to_store =
            NS(ElemByElemConfig_get_num_turns_to_store)( config );

        ( void )num_turns_to_store;

        SIXTRL_ASSERT( num_particles_to_store > ( num_elem_t )0u );
        SIXTRL_ASSERT( num_elements_to_store  > ( num_elem_t )0u );
        SIXTRL_ASSERT( num_turns_to_store     > ( num_elem_t )0u );

        switch( NS(ElemByElemConfig_get_order)( config ) )
        {
            case NS(ELEM_BY_ELEM_ORDER_TURN_ELEM_PARTICLES):
            {
                num_elem_t const stored_particles_per_turn =
                    num_particles_to_store * num_elements_to_store;

                num_elem_t temp = out_store_index % stored_particles_per_turn;
                temp = temp % num_particles_to_store;

                particle_id = ( index_t )temp;
                break;
            }

            default:
            {
                particle_id = ( num_elem_t )-1;
            }
        };

        if( particle_id >= ( index_t )0u )
        {
            particle_id += NS(ElemByElemConfig_get_min_particle_id)( config );
        }
    }

    return particle_id;
}

SIXTRL_INLINE NS(particle_index_t)
NS(ElemByElemConfig_get_at_element_id_from_store_index)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    NS(particle_num_elements_t) const out_store_index )
{
    typedef NS(particle_num_elements_t) num_elem_t;
    typedef NS(particle_index_t)        index_t;

    index_t at_element_id = ( index_t )-1;

    SIXTRL_ASSERT( config != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_min_particle_id)( config ) <=
                   NS(ElemByElemConfig_get_max_particle_id)( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_min_element_id)( config ) <=
                   NS(ElemByElemConfig_get_max_element_id)( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_min_turn)( config ) <=
                   NS(ElemByElemConfig_get_max_turn)( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_particles_to_store)( config ) >=
                   ( num_elem_t )0u );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_elements_to_store)( config ) >=
                   ( num_elem_t )0u );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_turns_to_store)( config ) >=
                   ( num_elem_t )0u );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_particles_to_store)( config ) <
                   NS(ElemByElemConfig_get_out_store_num_particles)( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_elements_to_store)( config ) <
                   NS(ElemByElemConfig_get_out_store_num_particles)( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_turns_to_store)( config ) <
                   NS(ElemByElemConfig_get_out_store_num_particles)( config ) );

    if( ( out_store_index >= ( num_elem_t )0u ) && ( out_store_index <
        ( NS(ElemByElemConfig_get_out_store_num_particles)( config ) ) ) )
    {
        num_elem_t const num_particles_to_store =
            NS(ElemByElemConfig_get_num_particles_to_store)( config );

        num_elem_t const num_elements_to_store =
            NS(ElemByElemConfig_get_num_elements_to_store)( config );

        num_elem_t const num_turns_to_store =
            NS(ElemByElemConfig_get_num_turns_to_store)( config );

        SIXTRL_ASSERT( num_particles_to_store > ( num_elem_t )0u );
        SIXTRL_ASSERT( num_elements_to_store  > ( num_elem_t )0u );
        SIXTRL_ASSERT( num_turns_to_store     > ( num_elem_t )0u );

        ( void )num_turns_to_store;

        switch( NS(ElemByElemConfig_get_order)( config ) )
        {
            case NS(ELEM_BY_ELEM_ORDER_TURN_ELEM_PARTICLES):
            {
                num_elem_t const stored_particles_per_turn =
                    num_particles_to_store * num_elements_to_store;

                num_elem_t temp = out_store_index % stored_particles_per_turn;
                temp = temp / num_particles_to_store;

                at_element_id = ( index_t )temp;
                break;
            }

            default:
            {
                at_element_id = ( num_elem_t )-1;
            }
        };

        if( at_element_id >= ( index_t )0u )
        {
            at_element_id += NS(ElemByElemConfig_get_min_element_id)( config );
        }
    }

    return at_element_id;
}

SIXTRL_INLINE NS(particle_index_t)
NS(ElemByElemConfig_get_at_turn_from_store_index)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    NS(particle_num_elements_t) const out_store_index )
{
    typedef NS(particle_num_elements_t) num_elem_t;
    typedef NS(particle_index_t)        index_t;

    index_t at_turn = ( index_t )-1;

    SIXTRL_ASSERT( config != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_min_particle_id)( config ) <=
                   NS(ElemByElemConfig_get_max_particle_id)( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_min_element_id)( config ) <=
                   NS(ElemByElemConfig_get_max_element_id)( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_min_turn)( config ) <=
                   NS(ElemByElemConfig_get_max_turn)( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_particles_to_store)( config ) >=
                   ( num_elem_t )0u );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_elements_to_store)( config ) >=
                   ( num_elem_t )0u );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_turns_to_store)( config ) >=
                   ( num_elem_t )0u );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_particles_to_store)( config ) <
                   NS(ElemByElemConfig_get_out_store_num_particles)( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_elements_to_store)( config ) <
                   NS(ElemByElemConfig_get_out_store_num_particles)( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_turns_to_store)( config ) <
                   NS(ElemByElemConfig_get_out_store_num_particles)( config ) );

    if( ( out_store_index >= ( num_elem_t )0u ) && ( out_store_index <
        ( NS(ElemByElemConfig_get_out_store_num_particles)( config ) ) ) )
    {
        num_elem_t const num_particles_to_store =
            NS(ElemByElemConfig_get_num_particles_to_store)( config );

        num_elem_t const num_elements_to_store =
            NS(ElemByElemConfig_get_num_elements_to_store)( config );

        num_elem_t const num_turns_to_store =
            NS(ElemByElemConfig_get_num_turns_to_store)( config );

        ( void )num_turns_to_store;

        SIXTRL_ASSERT( num_particles_to_store > ( num_elem_t )0u );
        SIXTRL_ASSERT( num_elements_to_store  > ( num_elem_t )0u );
        SIXTRL_ASSERT( num_turns_to_store     > ( num_elem_t )0u );

        switch( NS(ElemByElemConfig_get_order)( config ) )
        {
            case NS(ELEM_BY_ELEM_ORDER_TURN_ELEM_PARTICLES):
            {
                num_elem_t const stored_particles_per_turn =
                    num_particles_to_store * num_elements_to_store;

                at_turn = ( index_t )(
                    out_store_index / stored_particles_per_turn );

                break;
            }

            default:
            {
                at_turn = ( num_elem_t )-1;
            }
        };

        if( at_turn >= ( index_t )0u )
        {
            at_turn += NS(ElemByElemConfig_get_min_turn)( config );
        }
    }

    return at_turn;
}

/* ------------------------------------------------------------------------ */

SIXTRL_INLINE int NS(ElemByElemConfig_init_detailed)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT config,
    NS(elem_by_elem_order_t) const order,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn )
{
    int success = -1;

    typedef NS(particle_num_elements_t)     num_elem_t;
    typedef NS(particle_index_t)            index_t;

    SIXTRL_STATIC_VAR index_t const ZERO = ( index_t )0u;

    if( ( config != SIXTRL_NULLPTR ) &&
        ( order  != NS(ELEM_BY_ELEM_ORDER_INVALID) ) &&
        ( min_particle_id >= ZERO ) && ( max_particle_id >= min_particle_id ) &&
        ( min_element_id  >= ZERO ) && ( max_element_id  >= min_element_id ) &&
        ( min_turn        >= ZERO ) && ( max_turn        >= min_turn ) )
    {
        SIXTRL_STATIC_VAR num_elem_t const ONE = ( num_elem_t )1u;

        num_elem_t const num_particles_to_store =
            ( num_elem_t )( max_particle_id - min_particle_id + ONE );

        num_elem_t const num_elements_to_store  =
            ( num_elem_t )( max_element_id  - min_element_id  + ONE );

        num_elem_t const num_turns_to_store =
            ( num_elem_t )( max_turn - min_turn + ONE );

        NS(ElemByElemConfig_set_order)( config, order );
        NS(ElemByElemConfig_clear)( config );

        config->num_particles_to_store  = num_particles_to_store;
        config->num_elements_to_store   = num_elements_to_store;
        config->num_turns_to_store      = num_turns_to_store;

        config->min_particle_id         = min_particle_id;
        config->max_particle_id         = max_particle_id;

        config->min_element_id          = min_element_id;
        config->max_element_id          = max_element_id;

        config->min_turn                = min_turn;
        config->max_turn                = max_turn;

        success = 0;
    }

    return success;
}

SIXTRL_INLINE int NS(ElemByElemConfig_assign_managed_output_buffer)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const out_buffer_index_offset,
    NS(buffer_size_t) const slot_size )
{
    int success = -1;

    typedef NS(buffer_size_t) buf_size_t;

    if( ( config != SIXTRL_NULLPTR ) && ( output_buffer != SIXTRL_NULLPTR ) &&
        ( slot_size > ( buf_size_t )0u ) )
    {
        typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Particles)*  ptr_particles_t;
        typedef NS(elem_by_elem_out_addr_t)               address_t;

        ptr_particles_t particles = SIXTRL_NULLPTR;

        SIXTRL_ASSERT( NS(ElemByElemConfig_get_order)( config ) !=
            NS(ELEM_BY_ELEM_ORDER_INVALID) );

        SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
            output_buffer, slot_size ) );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)( output_buffer,
            slot_size ) > out_buffer_index_offset );

        particles = NS(Particles_managed_buffer_get_particles)(
            output_buffer, out_buffer_index_offset, slot_size );

        if( particles != SIXTRL_NULLPTR )
        {
            SIXTRL_ASSERT( ( NS(Particles_get_num_of_particles)( particles ) >=
                NS(ElemByElemConfig_get_out_store_num_particles)( config ) ) ||
                ( NS(ElemByElemConfig_get_out_store_num_particles)( config ) ==
                    ( NS(buffer_size_t) )0u ) );

            NS(ElemByElemConfig_set_output_store_address)( config,
                ( address_t )( uintptr_t )particles );

            success = 0;
        }
    }

    return success;
}

SIXTRL_INLINE SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
NS(ElemByElemConfig)* NS(ElemByElemConfig_preset)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT conf )
{
    if( conf != SIXTRL_NULLPTR )
    {
        NS(ElemByElemConfig_clear)( conf );
        NS(ElemByElemConfig_set_order)( conf, NS(ELEM_BY_ELEM_ORDER_INVALID) );
    }

    return conf;
}

SIXTRL_INLINE void NS(ElemByElemConfig_clear)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config )
{
    typedef NS(particle_num_elements_t) num_elem_t;
    typedef NS(particle_index_t)        index_t;
    typedef NS(elem_by_elem_out_addr_t) out_addr_t;

    if( config != SIXTRL_NULLPTR )
    {
        config->num_particles_to_store = ( num_elem_t )0u;
        config->num_elements_to_store  = ( num_elem_t )0u;
        config->num_turns_to_store     = ( num_elem_t )0u;

        config->min_particle_id        = ( index_t )-1;
        config->max_particle_id        = ( index_t )-1;
        config->min_element_id         = ( index_t )-1;
        config->max_element_id         = ( index_t )-1;
        config->min_turn               = ( index_t )-1;
        config->max_turn               = ( index_t )-1;

        NS(ElemByElemConfig_set_output_store_address)(
            config, ( out_addr_t )0u );
    }

    return;
}

SIXTRL_INLINE void NS(ElemByElemConfig_set_order)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    NS(elem_by_elem_order_t) const order )
{
    if( config != SIXTRL_NULLPTR )
    {
        config->order = ( NS(elem_by_elem_order_int_t ) )order;
    }

    return;
}

SIXTRL_INLINE void NS(ElemByElemConfig_set_output_store_address)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    NS(elem_by_elem_out_addr_t) const out_address )
{
    if( config != SIXTRL_NULLPTR ) config->out_store_addr = out_address;
    return;
}

#if !defined( _GPUCODE )

SIXTRL_INLINE NS(buffer_size_t)
NS(ElemByElemConfig_get_num_elem_by_elem_objects)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT belements )
{
    return NS(ElemByElemConfig_get_num_elem_by_elem_objects_from_managed_buffer)(
        NS(Buffer_get_const_data_begin)( belements ),
        NS(Buffer_get_slot_size)( belements ) );
}

SIXTRL_INLINE int NS(ElemByElemConfig_get_min_max_element_id_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_element_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_element_id,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT ptr_num_elem_by_elem_objects,
    NS(particle_index_t) const element_id_offset )
{
    int success = -1;

    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(particle_index_t) index_t;
    typedef NS(object_type_id_t) type_id_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* obj_iter_t;

    buf_size_t const num_objects = NS(Buffer_get_num_of_objects)( belements );

    SIXTRL_ASSERT( belements != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( belements ) );

    if( ( num_objects > ( buf_size_t )0u ) &&
        ( element_id_offset >= ( index_t )0u ) )
    {
        index_t min_element_id = element_id_offset;
        index_t max_element_id = element_id_offset;
        buf_size_t num_elem_by_elem_objects = ( buf_size_t )0u;

        obj_iter_t obj_it  = NS(Buffer_get_const_objects_begin)( belements );
        obj_iter_t obj_end = NS(Buffer_get_const_objects_end)( belements );

        for( ; obj_it != obj_end ; ++obj_it )
        {
            type_id_t const type_id = NS(Object_get_type_id)( obj_it );

            if( ( type_id != NS(OBJECT_TYPE_NONE) ) &&
                ( type_id != NS(OBJECT_TYPE_PARTICLE) ) &&
                ( type_id != NS(OBJECT_TYPE_INVALID) ) &&
                ( type_id != NS(OBJECT_TYPE_LINE) ) &&
                ( type_id != NS(OBJECT_TYPE_ELEM_BY_ELEM_CONF) ) )
            {
                ++num_elem_by_elem_objects;
                ++obj_it;
//
                break;
            }

            ++min_element_id;
        }

        max_element_id = min_element_id;

        for( ; obj_it != obj_end ; ++obj_it )
        {
            type_id_t const type_id = NS(Object_get_type_id)( obj_it );

            if( ( type_id != NS(OBJECT_TYPE_NONE) ) &&
                ( type_id != NS(OBJECT_TYPE_PARTICLE) ) &&
                ( type_id != NS(OBJECT_TYPE_INVALID) ) &&
                ( type_id != NS(OBJECT_TYPE_LINE) ) &&
                ( type_id != NS(OBJECT_TYPE_ELEM_BY_ELEM_CONF) ) )
            {
                ++num_elem_by_elem_objects;
                ++max_element_id;
            }
        }

        SIXTRL_ASSERT( num_elem_by_elem_objects <= num_objects );

        if( num_elem_by_elem_objects == ( buf_size_t )0u )
        {
            min_element_id = max_element_id = element_id_offset;
        }

        SIXTRL_ASSERT( min_element_id <= max_element_id );

        if( ( min_element_id >= element_id_offset ) &&
            ( max_element_id >= min_element_id ) &&
            ( max_element_id <  ( index_t )(
                min_element_id + num_elem_by_elem_objects ) ) )
        {
            if(  ptr_min_element_id != SIXTRL_NULLPTR )
            {
                *ptr_min_element_id = min_element_id;
            }

            if(  ptr_max_element_id != SIXTRL_NULLPTR )
            {
                *ptr_max_element_id = max_element_id;
            }

            if(  ptr_num_elem_by_elem_objects != SIXTRL_NULLPTR )
            {
                *ptr_num_elem_by_elem_objects = num_elem_by_elem_objects;
            }

            success = 0;
        }
    }

    return success;
}

SIXTRL_INLINE NS(buffer_size_t) NS(ElemByElemConfig_get_required_num_slots)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        const NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t num_required_slots = ( buf_size_t )0u;

    if( ( config != SIXTRL_NULLPTR ) && ( slot_size != ( buf_size_t )0u ) )
    {
        num_required_slots = NS(ManagedBuffer_get_slot_based_length)(
            sizeof( NS(ElemByElemConfig) ), slot_size );

        num_required_slots /= slot_size;
    }

    return num_required_slots;
}

SIXTRL_INLINE NS(buffer_size_t) NS(ElemByElemConfig_get_required_num_dataptrs)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        const NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    ( void )config;
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(ElemByElemConfig_get_type_id)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        const NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    ( void )config;
    return NS(OBJECT_TYPE_ELEM_BY_ELEM_CONF);
}

SIXTRL_INLINE bool NS(ElemByElemConfig_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs )
{
    typedef NS(ElemByElemConfig) config_t;

    config_t config;
    NS(ElemByElemConfig_preset)( &config );

    return NS(Buffer_can_add_object)( buffer, sizeof( NS(ElemByElemConfig) ),
        NS(ElemByElemConfig_get_required_num_dataptrs)( &config ),
        SIXTRL_NULLPTR, SIXTRL_NULLPTR,
        requ_objects, requ_slots, requ_dataptrs );
}

SIXTRL_INLINE SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
NS(ElemByElemConfig)* NS(ElemByElemConfig_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef NS(ElemByElemConfig)                    config_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC config_t*     ptr_to_config_t;

    SIXTRL_ASSERT( buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(Buffer_has_datastore)( buffer ) );
    SIXTRL_ASSERT( NS(Buffer_allow_append_objects)( buffer ) );

    NS(ElemByElemConfig) config;
    NS(ElemByElemConfig_preset)( &config );

    return ( ptr_to_config_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &config, sizeof( config_t ),
            NS(ElemByElemConfig_get_type_id)( &config ),
            NS(ElemByElemConfig_get_required_num_dataptrs)( &config ),
            SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
}

SIXTRL_INLINE SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
NS(ElemByElemConfig)* NS(ElemByElemConfig_add)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(elem_by_elem_order_t) const order,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn )
{
    typedef NS(ElemByElemConfig) config_t;
    typedef SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
            ptr_new_config_t;

    ptr_new_config_t ptr_new_config = SIXTRL_NULLPTR;

    SIXTRL_ASSERT( buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(Buffer_has_datastore)( buffer ) );
    SIXTRL_ASSERT( NS(Buffer_allow_append_objects)( buffer ) );

    config_t config;
    NS(ElemByElemConfig_preset)( &config );

    if( 0 == NS(ElemByElemConfig_init_detailed)( &config, order,
            min_particle_id, max_particle_id, min_element_id, max_element_id,
            min_turn, max_turn ) )
    {
        ptr_new_config = ( ptr_new_config_t )( uintptr_t
            )NS(Object_get_begin_addr)( NS(Buffer_add_object)(
                buffer, &config, sizeof( config_t ),
                NS(ElemByElemConfig_get_type_id)( &config ),
                NS(ElemByElemConfig_get_required_num_dataptrs)( &config ),
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
    }

    return ptr_new_config;
}

SIXTRL_INLINE SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
NS(ElemByElemConfig)* NS(ElemByElemConfig_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
     SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    typedef NS(ElemByElemConfig)                    config_t;
    typedef SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
            ptr_new_config_t;

    ptr_new_config_t ptr_new_config = SIXTRL_NULLPTR;

    SIXTRL_ASSERT( buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(Buffer_has_datastore)( buffer ) );
    SIXTRL_ASSERT( NS(Buffer_allow_append_objects)( buffer ) );

    if( config != SIXTRL_NULLPTR )
    {
        ptr_new_config = ( ptr_new_config_t )( uintptr_t
            )NS(Object_get_begin_addr)( NS(Buffer_add_object)(
                buffer, config, sizeof( config_t ),
                NS(ElemByElemConfig_get_type_id)( config ),
                NS(ElemByElemConfig_get_required_num_dataptrs)( config ),
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
    }

    return ptr_new_config;
}

#endif /* !defined( _GPUCODE ) */

SIXTRL_INLINE NS(buffer_size_t)
NS(ElemByElemConfig_get_num_elem_by_elem_objects_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT belements_buffer, NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(object_type_id_t) type_id_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* obj_iter_t;

    buf_size_t num_elem_by_elem_objects = ( buf_size_t )0u;

    obj_iter_t it = NS(ManagedBuffer_get_const_objects_index_begin)(
        belements_buffer, slot_size );

    obj_iter_t end = NS(ManagedBuffer_get_const_objects_index_end)(
        belements_buffer, slot_size );

    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
        belements_buffer, slot_size ) );

    SIXTRL_ASSERT( ( ( uintptr_t )it ) <= ( uintptr_t )end );

    if( ( belements_buffer != SIXTRL_NULLPTR ) &&
        ( slot_size > ( buf_size_t )0u ) &&
        ( it != SIXTRL_NULLPTR ) && ( end != SIXTRL_NULLPTR ) )
    {
        for( ; it != end ; ++it )
        {
            type_id_t const type_id = NS(Object_get_type_id)( it );

            if( ( type_id != NS(OBJECT_TYPE_NONE) ) &&
                ( type_id != NS(OBJECT_TYPE_PARTICLE) ) &&
                ( type_id != NS(OBJECT_TYPE_INVALID) ) &&
                ( type_id != NS(OBJECT_TYPE_LINE) ) &&
                ( type_id != NS(OBJECT_TYPE_ELEM_BY_ELEM_CONF) ) )
            {
                ++num_elem_by_elem_objects;
            }
        }
    }

    return num_elem_by_elem_objects;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRL_SIXTRACKLIB_COMMON_OUTPUT_ELEM_BY_ELEM_CONFIG_H__ */

/* end: sixtracklib/common/output/elem_by_elem_config.h */
