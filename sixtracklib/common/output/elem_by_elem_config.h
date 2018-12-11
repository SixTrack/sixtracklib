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
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/internal/buffer_defines.h"
    #include "sixtracklib/common/internal/elem_by_elem_config_defines.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

typedef SIXTRL_INT64_T  NS(elem_by_elem_order_int_t);
typedef SIXTRL_UINT64_T NS(elem_by_elem_out_addr_t);

typedef enum NS(ElemByElemStoreOrder)
{
    NS(ELEM_BY_ELEM_ORDER_TURN_ELEM_PARTICLES)  =  0,
    NS(ELEM_BY_ELEM_ORDER_TURN_INVALID)         = -1
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
NS(ElemByElemConfig_get_particles_store_index)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    NS(particle_index_t) const particle_id,
    NS(particle_index_t) const at_element_id,
    NS(particle_index_t) const at_turn );

SIXTRL_FN SIXTRL_STATIC NS(particle_index_t) const
NS(ElemByElemConfig_get_particle_id_from_store_index)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    NS(particle_num_elements_t) const out_store_index );

SIXTRL_FN SIXTRL_STATIC NS(particle_index_t) const
NS(ElemByElemConfig_get_at_element_id_from_store_index)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    NS(particle_num_elements_t) const out_store_index );

SIXTRL_FN SIXTRL_STATIC NS(particle_index_t) const
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
NS(ElemByElemConfig_assign_managed_particles_out_buffer)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const out_buffer_index_offset );


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
    elem_by_elem_order_t const order );

SIXTRL_FN SIXTRL_STATIC void NS(ElemByElemConfig_set_output_store_address)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    NS(particle_num_elements_t) const out_buffer_index_offset );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(ElemByElemConfig_init)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    NS(elem_by_elem_order_t) const order,
    SIXTRL_BUFFER_ARGPTR_DEC
        const NS(Buffer) *const beam_elements_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC
        const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(ElemByElemConfig_assign_particles_out_buffer)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const out_buffer_index_offset );


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
    NS(particle_index_t) const max_turn,
    NS(particle_num_elements_t) const out_buffer_index_offset );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
NS(ElemByElemConfig)* NS(ElemByElemConfig_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
     SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config );

#endif /* !defined( _GPUCODE ) */

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

SIXTRL_INLINE NS(elem_by_elem_order_t)
NS(ElemByElemConfig_get_order)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config )
{
    SIXTRL_ASSERT( config != SIXTRL_NULLPTR );
    return config->order;
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
NS(ElemByElemConfig_get_particles_store_index)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    NS(particle_index_t) const particle_id,
    NS(particle_index_t) const at_element, NS(particle_index_t) const at_turn )
{
    typedef NS(particle_num_elements_t) num_elem_t;
    typedef NS(particle_index_t)        index_t;

    num_elem_t out_store_index = ( num_elem_t )-1;

    SIXTRL_ASSERT( config != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( particle_id >= ( index_t )0u );
    SIXTRL_ASSERT( at_element  >= ( index_t )0u );
    SIXTRL_ASSERT( at_turn     >= ( index_t )0u );

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
            NS(ElemByElemConfig_get_num_elements_to_store)( config );

        SIXTRL_ASSERT( num_particles_to_store > ( num_elem_t )0u );
        SIXTRL_ASERTT( num_elements_to_store  > ( num_elem_t )0u );
        SIXTRL_ASSERT( num_turns_to_store     > ( num_elem_t )0u );

        switch( NS(ElemByElemConfig_get_order)( config ) )
        {
            case NS(ELEM_BY_ELEM_ORDER_TURN_ELEM_PARTICLES):
            {
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

        ASSERT_TRUE(
            ( ( out_store_index >= ( num_elem_t )0u ) && ( out_store_index <
                NS(ElemByElemConfig_get_out_store_num_particles)( config ) )
            ) || ( out_store_index < ( num_elem_t )0u ) );
    }

    return out_store_index;
}

SIXTRL_INLINE NS(particle_index_t) const
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
                   NS(ElemByElemConfig_get_out_store_num_particles( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_elements_to_store)( config ) <
                   NS(ElemByElemConfig_get_out_store_num_particles( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_turns_to_store)( config ) <
                   NS(ElemByElemConfig_get_out_store_num_particles( config ) );

    if( ( out_store_index >= ( num_elem_t )0u ) && ( out_store_index <
        ( NS(ElemByElemConfig_get_out_store_num_particles)( config ) ) )
    {
        num_elem_t const num_particles_to_store =
            NS(ElemByElemConfig_get_num_particles_to_store)( config );

        num_elem_t const num_elements_to_store =
            NS(ElemByElemConfig_get_num_elements_to_store)( config );

        num_elem_t const num_turns_to_store =
            NS(ElemByElemConfig_get_num_turns_to_store)( config );

        SIXTRL_ASSERT( num_particles_to_store > ( num_elem_t )0u );
        SIXTRL_ASERTT( num_elements_to_store  > ( num_elem_t )0u );
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
                out_store_index = ( num_elem_t )-1;
            }
        };

        if( particle_id >= ( index_t )0u )
        {
            particle_id += NS(ElemByElemConfig_get_min_particle_id)( config );
        }
    }

    return particle_id;
}

SIXTRL_INLINE NS(particle_index_t) const
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
                   NS(ElemByElemConfig_get_out_store_num_particles( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_elements_to_store)( config ) <
                   NS(ElemByElemConfig_get_out_store_num_particles( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_turns_to_store)( config ) <
                   NS(ElemByElemConfig_get_out_store_num_particles( config ) );

    if( ( out_store_index >= ( num_elem_t )0u ) && ( out_store_index <
        ( NS(ElemByElemConfig_get_out_store_num_particles)( config ) ) )
    {
        num_elem_t const num_particles_to_store =
            NS(ElemByElemConfig_get_num_particles_to_store)( config );

        num_elem_t const num_elements_to_store =
            NS(ElemByElemConfig_get_num_elements_to_store)( config );

        num_elem_t const num_turns_to_store =
            NS(ElemByElemConfig_get_num_turns_to_store)( config );

        SIXTRL_ASSERT( num_particles_to_store > ( num_elem_t )0u );
        SIXTRL_ASERTT( num_elements_to_store  > ( num_elem_t )0u );
        SIXTRL_ASSERT( num_turns_to_store     > ( num_elem_t )0u );

        switch( NS(ElemByElemConfig_get_order)( config ) )
        {
            case NS(SIXTRL_ELEM_BY_ELEM_ORDER_TURN_ELEM_PARTICLES):
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
                out_store_index = ( num_elem_t )-1;
            }
        };

        if( at_element_id >= ( index_t )0u )
        {
            at_element_id += NS(ElemByElemConfig_get_min_element_id)( config );
        }
    }

    return at_element_id;
}

SIXTRL_INLINE NS(particle_index_t) const
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
                   NS(ElemByElemConfig_get_out_store_num_particles( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_elements_to_store)( config ) <
                   NS(ElemByElemConfig_get_out_store_num_particles( config ) );

    SIXTRL_ASSERT( NS(ElemByElemConfig_get_num_turns_to_store)( config ) <
                   NS(ElemByElemConfig_get_out_store_num_particles( config ) );

    if( ( out_store_index >= ( num_elem_t )0u ) && ( out_store_index <
        ( NS(ElemByElemConfig_get_out_store_num_particles)( config ) ) )
    {
        num_elem_t const num_particles_to_store =
            NS(ElemByElemConfig_get_num_particles_to_store)( config );

        num_elem_t const num_elements_to_store =
            NS(ElemByElemConfig_get_num_elements_to_store)( config );

        num_elem_t const num_turns_to_store =
            NS(ElemByElemConfig_get_num_turns_to_store)( config );

        SIXTRL_ASSERT( num_particles_to_store > ( num_elem_t )0u );
        SIXTRL_ASERTT( num_elements_to_store  > ( num_elem_t )0u );
        SIXTRL_ASSERT( num_turns_to_store     > ( num_elem_t )0u );

        switch( NS(ElemByElemConfig_get_order)( config ) )
        {
            case NS(SIXTRL_ELEM_BY_ELEM_ORDER_TURN_ELEM_PARTICLES):
            {
                num_elem_t const stored_particles_per_turn =
                    num_particles_to_store * num_elements_to_store;

                at_turn = ( index_t )(
                    out_store_index / stored_particles_per_turn );

                break;
            }

            default:
            {
                out_store_index = ( num_elem_t )-1;
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
    typedef NS(elem_by_elem_order_int_t)    order_int_t;
    typedef NS(particle_index_t)            index_t;

    if( ( config != SIXTRL_NULLPTR ) &&
        ( ( NS(order_int_t ) )order >= ( NS(order_int_t ) )0u ) &&
        ( ( NS(order_int_t ) )order <= ( NS(order_int_t ) )1u ) &&
        ( min_particle_id >= ( index_t )0u ) &&
        ( max_particle_id >= min_particle_id ) &&
        ( min_element_id  >= ( index_t )0u ) &&
        ( max_element_id  >= min_element_id ) &&
        ( min_turn        >= ( index_t )0u ) &&
        ( max_turn        >= min_turn ) &&
        ( out_buffer_index_offset >= ( num_elem_t )0u ) )
    {
        num_elem_t const num_particles_to_store = ( num_elem_t )(
            max_particle_id - min_particle_id + ( num_elem_t )1u );

        num_elem_t const num_elements_to_store  = ( num_elem_t )(
            max_element_id  - min_element_id  + ( num_elem_t )1u );

        num_elem_t const num_turns_to_store = ( num_elem_t )(
            max_turn - min_turn + ( num_elem_t )1u );

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

SIXTRL_INLINE SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
NS(ElemByElemConfig)* NS(ElemByElemConfig_preset)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config )
{
    if( config != SIXTRL_NULLPTR )
    {
        NS(ElemByELemConfig_clear)( config );

        NS(ElemByElemConfig_set_order)(
            config, NS(ELEM_BY_ELEM_ORDER_TURN_INVALID) );
    }

    return config;
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
        config->out_buffer_index_offset =
            ( NS(elem_by_elem_order_int_t ) )order;
    }

    return;
}

SIXTRL_INLINE void NS(ElemByElemConfig_set_output_store_address)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    NS(elem_by_elem_out_addr_t) const out_address )
{
    if( config != SIXTRL_NULPTR ) config->out_store_addr = out_address;
   return;
}

#if !defined( _GPUCODE )

SIXTRL_INLINE NS(buffer_size_t) NS(ElemByElemConfig_get_required_num_slots)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        const NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t num_required_slots = ( buf_size_t )0u;

    if( ( buffer != SIXTRL_NULLPTR ) && ( slot_size != ( buf_size_t )0u ) )
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
    typedef NS(buffer_size_t)    buf_size_t;
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
    NS(Particles_preset)( &config );

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
    typedef NS(ElemByElemConfig)                    config_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC config_t*     ptr_to_config_t;
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
    typedef SIXTRL_BUFFER_DATAPTR_DEC config_t*     ptr_to_config_t;
    typedef SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
            ptr_new_config_t;

    ptr_new_config_t ptr_new_config = SIXTRL_NULLPTR;

    SIXTRL_ASSERT( buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(Buffer_has_datastore)( buffer ) );
    SIXTRL_ASSERT( NS(Buffer_allow_append_objects)( buffer ) );

    config_t config;
    NS(ElemByElemConfig_preset)( &config );

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

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRL_SIXTRACKLIB_COMMON_OUTPUT_ELEM_BY_ELEM_CONFIG_H__ */

/* end: sixtracklib/common/output/elem_by_elem_config.h */
