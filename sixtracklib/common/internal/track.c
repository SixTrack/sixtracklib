#include "sixtracklib/common/track.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/config.h"
#include "sixtracklib/common/internal/beam_elements_defines.h"
#include "sixtracklib/common/internal/particles_defines.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"
#include "sixtracklib/common/output/output_buffer.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index )
{
    return NS(Track_particle_beam_element_obj)(
        particles, particle_idx, NS(Buffer_get_const_object)(
            beam_elements, be_index ) );
}

SIXTRL_HOST_FN int NS(Track_subset_of_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index )
{
    return NS(Track_subset_of_particles_beam_element_obj)(
        particles, particle_idx_begin, particle_idx_end, particle_idx_stride,
        NS(Buffer_get_const_object)( beam_elements, be_index ) );
}


SIXTRL_HOST_FN int NS(Track_all_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index )
{
    return NS(Track_subset_of_particles_beam_element_obj)(
        particles, 0, NS(Particles_get_num_of_particles)( particles ), 1,
        NS(Buffer_get_const_object)( beam_elements, be_index ) );
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index_begin,
    NS(buffer_size_t) const be_index_end )
{
    return NS(Track_particle_beam_elements_obj)(
        particles, particle_idx,
        NS(Buffer_get_const_object)( beam_elements, be_index_begin ),
        NS(Buffer_get_const_object)( beam_elements, be_index_end ) );
}

SIXTRL_HOST_FN int NS(Track_subset_of_particles_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index_begin,
    NS(buffer_size_t) const be_index_end )
{
    return NS(Track_subset_of_particles_beam_elements_obj)(
        particles, particle_idx_begin, particle_idx_end, particle_idx_stride,
        NS(Buffer_get_const_object)( beam_elements, be_index_begin ),
        NS(Buffer_get_const_object)( beam_elements, be_index_end ) );
}

SIXTRL_HOST_FN int NS(Track_all_particles_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_begin_idx, NS(buffer_size_t) const be_end_idx )
{
    return NS(Track_subset_of_particles_beam_elements_obj)(
        particles, 0, NS(Particles_get_num_of_particles)( particles), 1,
        NS(Buffer_get_const_object)( beam_elements, be_begin_idx ),
        NS(Buffer_get_const_object)( beam_elements, be_end_idx ) );
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements )
{
    return NS(Track_particle_beam_elements_obj)( particles, particle_idx,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ) );
}


SIXTRL_HOST_FN int NS(Track_subset_of_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements )
{
    return NS(Track_subset_of_particles_beam_elements_obj)(
        particles, particle_idx_begin, particle_idx_end, particle_idx_stride,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ) );
}


SIXTRL_HOST_FN int NS(Track_all_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements )
{
    return NS(Track_subset_of_particles_beam_elements_obj)( particles, 0,
        NS(Particles_get_num_of_particles)( particles ), 1,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ) );
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn )
{
    return NS(Track_particle_until_turn_obj)( particles, particle_index,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ), until_turn );
}

SIXTRL_HOST_FN int NS(Track_subset_of_particles_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn )
{
    return NS(Track_subset_of_particles_until_turn_obj)(
        particles, particle_idx_begin, particle_idx_end, particle_idx_stride,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ), until_turn );
}

SIXTRL_HOST_FN int NS(Track_all_particles_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn )
{
    return NS(Track_all_particles_until_turn_obj)( particles,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ),
        until_turn );
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_element_by_element_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const idx,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig)
        *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const be_index )
{
    return NS(Track_particle_element_by_element_obj)( particles, idx, config,
        NS(Buffer_get_const_object)( belements, be_index ) );
}

SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_element_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) idx,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig)
        *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const be_index )
{
    return NS(Track_subset_of_particles_element_by_element_obj)(
        particles, idx, particle_idx_end, particle_idx_stride, config,
        NS(Buffer_get_const_object)( belements, be_index ) );
}

SIXTRL_HOST_FN int NS(Track_all_particles_element_by_element_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig)
        *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const be_index )
{
    return NS(Track_all_particles_element_by_element_obj)( particles, config,
        NS(Buffer_get_const_object)( belements, be_index ) );
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const index,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles )
{
    typedef NS(particle_index_t)        index_t;
    typedef NS(particle_num_elements_t) num_elem_t;

    index_t min_particle_id   = ( index_t )0;
    index_t max_particle_id   = ( index_t )0;

    index_t min_at_element_id = ( index_t )0;
    index_t max_at_element_id = ( index_t )0;

    index_t min_at_turn       = ( index_t )0;
    index_t max_at_turn       = ( index_t )0;

    int success = NS(Particles_find_min_max_attributes)( particles,
        &min_particle_id, &max_particle_id,
        &min_at_element_id, &max_at_element_id,
        &min_at_turn, &max_at_turn );

    NS(ElemByElemConfig) config;
    NS(ElemByElemConfig_preset)( &config );

    if( success == 0 )
    {
        num_elem_t const num_beam_elements =
            NS(Buffer_get_num_of_objects)( belements );

        if( ( num_beam_elements >= ( num_elem_t )0u ) &&
            ( num_beam_elements >  ( num_elem_t )(
                max_at_element_id + ( index_t )1u ) ) )
        {
            max_at_element_id = ( index_t )(
                num_beam_elements - ( num_elem_t )1u );
        }

        SIXTRL_ASSERT( min_at_turn >= ( index_t )0u );
        SIXTRL_ASSERT( max_at_turn >= min_at_turn );

        if( max_at_turn < max_turn )
        {
            max_at_turn = max_turn;
        }

        success = NS(ElemByElemConfig_init_detailed)(
            &config, NS(ELEM_BY_ELEM_ORDER_TURN_ELEM_PARTICLES),
            min_particle_id, max_particle_id, min_at_element_id,
            max_at_element_id, min_at_turn, max_at_turn, true );

        if( success == 0 )
        {
            NS(ElemByElemConfig_set_output_store_address)(
                &config, ( uintptr_t )out_particles );

            success = NS(Track_particle_element_by_element_until_turn_objs)(
                particles, index, &config,
                NS(Buffer_get_const_objects_begin)( belements ),
                NS(Buffer_get_const_objects_end)( belements ), max_turn );
        }
    }

    return success;
}

SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const idx,
    NS(particle_num_elements_t) const idx_end,
    NS(particle_num_elements_t) const stride,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles )
{
    typedef NS(particle_index_t)        index_t;
    typedef NS(particle_num_elements_t) num_elem_t;

    index_t min_particle_id   = ( index_t )0;
    index_t max_particle_id   = ( index_t )0;

    index_t min_at_element_id = ( index_t )0;
    index_t max_at_element_id = ( index_t )0;

    index_t min_at_turn       = ( index_t )0;
    index_t max_at_turn       = ( index_t )0;

    int success = NS(Particles_find_min_max_attributes)( particles,
        &min_particle_id, &max_particle_id,
        &min_at_element_id, &max_at_element_id,
        &min_at_turn, &max_at_turn );

    NS(ElemByElemConfig) config;
    NS(ElemByElemConfig_preset)( &config );

    if( success == 0 )
    {
        num_elem_t const num_beam_elements =
            NS(Buffer_get_num_of_objects)( belements );

        if( ( num_beam_elements >= ( num_elem_t )0u ) &&
            ( num_beam_elements >  ( num_elem_t )(
                max_at_element_id + ( index_t )1u ) ) )
        {
            max_at_element_id = ( index_t )(
                num_beam_elements - ( num_elem_t )1u );
        }

        SIXTRL_ASSERT( min_at_turn >= ( index_t )0u );
        SIXTRL_ASSERT( max_at_turn >= min_at_turn );

        if( max_at_turn < max_turn )
        {
            max_at_turn = max_turn;
        }

        success = NS(ElemByElemConfig_init_detailed)(
            &config, NS(ELEM_BY_ELEM_ORDER_TURN_ELEM_PARTICLES),
            min_particle_id, max_particle_id, min_at_element_id,
            max_at_element_id, min_at_turn, max_at_turn, true );

        if( success == 0 )
        {
            NS(ElemByElemConfig_set_output_store_address)(
                &config, ( uintptr_t )out_particles );

            success =
            NS(Track_subset_of_particles_element_by_element_until_turn_objs)(
                particles, idx, idx_end, stride, &config,
                NS(Buffer_get_const_objects_begin)( belements ),
                NS(Buffer_get_const_objects_end)( belements ), max_turn );
        }
    }

    return success;
}

SIXTRL_HOST_FN int NS(Track_all_particles_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    return NS(Track_subset_of_particles_element_by_element)( particles,
        ( num_elem_t )0u, NS(Particles_get_num_of_particles)( particles ),
        ( num_elem_t )1u, max_turn, belements, out_particles );
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_element_by_elements_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const idx,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig)
        *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const be_begin_idx, NS(buffer_size_t) const be_end_idx )
{
    SIXTRL_ASSERT( be_begin_idx <= be_end_idx );
    SIXTRL_ASSERT( NS(Buffer_get_num_of_objects)( belements ) >= be_end_idx );

    return NS(Track_particle_element_by_element_objs)(
        particles, idx, config,
        NS(Buffer_get_const_object)( belements, be_begin_idx ),
        NS(Buffer_get_const_object)( belements, be_end_idx ) );
}

SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_elements_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) idx,
    NS(particle_num_elements_t) const idx_end,
    NS(particle_num_elements_t) const stride,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig)
        *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const be_begin_idx, NS(buffer_size_t) const be_end_idx )
{
    SIXTRL_ASSERT( be_begin_idx <= be_end_idx );
    SIXTRL_ASSERT( be_end_idx <= NS(Buffer_get_num_of_objects)( belements ) );

    return NS(Track_subset_of_particles_element_by_element_objs)(
        particles, idx, idx_end, stride, config,
        NS(Buffer_get_const_object)( belements, be_begin_idx ),
        NS(Buffer_get_const_object)( belements, be_end_idx ) );
}

SIXTRL_HOST_FN int NS(Track_all_particles_element_by_elements_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig)
        *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const be_begin_idx, NS(buffer_size_t) const be_end_idx )
{
    SIXTRL_ASSERT( be_begin_idx <= be_end_idx );
    SIXTRL_ASSERT( be_end_idx <= NS(Buffer_get_num_of_objects)( belements ) );

    return NS(Track_all_particles_element_by_element_objs)( particles, config,
        NS(Buffer_get_const_object)( belements, be_begin_idx ),
        NS(Buffer_get_const_object)( belements, be_end_idx ) );
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_element_by_element_until_turn_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const index,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig)
        *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(particle_index_t) const until_turn )
{
    return NS(Track_particle_element_by_element_until_turn_objs)(
        particles, index, config,
        NS(Buffer_get_const_objects_begin)( belements ),
        NS(Buffer_get_const_objects_end)( belements ), until_turn );
}

SIXTRL_HOST_FN int
    NS(Track_subset_of_particles_element_by_element_until_turn_details)(
        SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
        NS(particle_num_elements_t) idx,
        NS(particle_num_elements_t) const idx_end,
        NS(particle_num_elements_t) const stride,
        SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig)
            *const SIXTRL_RESTRICT config,
        SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer)
            *const SIXTRL_RESTRICT belements,
        NS(particle_index_t) const until_turn )
{
    return NS(Track_subset_of_particles_element_by_element_until_turn_objs)(
        particles, idx, idx_end, stride, config,
        NS(Buffer_get_const_objects_begin)( belements ),
        NS(Buffer_get_const_objects_end)( belements ), until_turn );
}

SIXTRL_HOST_FN int
    NS(Track_all_particles_element_by_element_until_turn_details)(
        SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig)
            *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(particle_index_t) const until_turn )
{
    return NS(Track_all_particles_element_by_element_until_turn_objs)(
        particles, config, NS(Buffer_get_const_objects_begin)( belements ),
        NS(Buffer_get_const_objects_end)( belements ), until_turn );
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_element_by_element_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const index,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles )
{
    typedef NS(particle_index_t)        index_t;
    typedef NS(particle_num_elements_t) num_elem_t;

    index_t min_particle_id   = ( index_t )0;
    index_t max_particle_id   = ( index_t )0;

    index_t min_at_element_id = ( index_t )0;
    index_t max_at_element_id = ( index_t )0;

    index_t min_at_turn       = ( index_t )0;
    index_t max_at_turn       = ( index_t )0;

    int success = NS(Particles_find_min_max_attributes)( particles,
        &min_particle_id, &max_particle_id,
        &min_at_element_id, &max_at_element_id,
        &min_at_turn, &max_at_turn );

    NS(ElemByElemConfig) config;
    NS(ElemByElemConfig_preset)( &config );

    if( success == 0 )
    {
        num_elem_t const num_beam_elements =
            NS(Buffer_get_num_of_objects)( belements );

        if( ( num_beam_elements >= ( num_elem_t )0u ) &&
            ( num_beam_elements >  ( num_elem_t )(
                max_at_element_id + ( index_t )1u ) ) )
        {
            max_at_element_id = ( index_t )(
                num_beam_elements - ( num_elem_t )1u );
        }

        SIXTRL_ASSERT( min_at_turn >= ( index_t )0u );
        SIXTRL_ASSERT( max_at_turn >= min_at_turn );

        if( max_at_turn < until_turn )
        {
            max_at_turn = until_turn;
        }

        success = NS(ElemByElemConfig_init_detailed)(
            &config, NS(ELEM_BY_ELEM_ORDER_TURN_ELEM_PARTICLES),
            min_particle_id, max_particle_id, min_at_element_id,
            max_at_element_id, min_at_turn, max_at_turn, true );

        if( success == 0 )
        {
            NS(ElemByElemConfig_set_output_store_address)(
                &config, ( uintptr_t )out_particles );

            success = NS(Track_particle_element_by_element_until_turn_objs)(
                particles, index, &config,
                NS(Buffer_get_const_objects_begin)( belements ),
                NS(Buffer_get_const_objects_end)( belements ), until_turn );
        }
    }

    return success;
}

SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_element_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) idx,
    NS(particle_num_elements_t) const idx_end,
    NS(particle_num_elements_t) const stride,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles )
{
    typedef NS(particle_index_t)        index_t;
    typedef NS(particle_num_elements_t) num_elem_t;
    typedef NS(buffer_size_t)           buf_size_t;

    SIXTRL_STATIC_VAR buf_size_t ZERO = ( buf_size_t )0u;

    num_elem_t const available_num_out_particles =
        NS(Particles_get_num_of_particles)( out_particles );

    buf_size_t num_elem_by_elem_objs = ZERO;

    index_t const start_elem = ( index_t )0u;
    index_t min_part_id, max_part_id, min_elem_id, max_elem_id,
            min_turn_id, max_turn_id;

    int success = NS(OutputBuffer_get_min_max_attributes)( particles,
        belements, &min_part_id, &max_part_id, &min_elem_id, &max_elem_id,
            &min_turn_id, &max_turn_id, &num_elem_by_elem_objs, start_elem );

    num_elem_t const requ_num_output_particles = ( num_elem_t
        )NS(ElemByElemConfig_get_stored_num_particles_detailed)( min_part_id,
            max_part_id, min_elem_id, max_elem_id, min_turn_id, until_turn );

    if( ( 0 == success ) && ( min_turn_id < until_turn ) &&
        ( num_elem_by_elem_objs > ZERO ) &&
        ( requ_num_output_particles <= available_num_out_particles ) )
    {
        NS(ElemByElemConfig) config;
        NS(ElemByElemConfig_preset)( &config );

        success = NS(ElemByElemConfig_init_detailed)( &config,
            NS(ELEM_BY_ELEM_ORDER_TURN_ELEM_PARTICLES), min_part_id,
                max_part_id, min_elem_id, max_elem_id,
                    min_turn_id, until_turn, true );

        if( success == 0 )
        {
            NS(ElemByElemConfig_set_output_store_address)(
                &config, ( uintptr_t )out_particles );

            success =
            NS(Track_subset_of_particles_element_by_element_until_turn_objs)(
                particles, idx, idx_end, stride, &config,
                    NS(Buffer_get_const_objects_begin)( belements ),
                    NS(Buffer_get_const_objects_end)( belements ), until_turn );
        }
    }
    else if( ( success == 0 ) && ( min_turn_id < until_turn ) )
    {
        success = -1;
    }

    return success;
}

SIXTRL_HOST_FN int NS(Track_all_particles_element_by_element_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles )
{
    return NS(Track_subset_of_particles_element_by_element_until_turn)(
        particles, 0, NS(Particles_get_num_of_particles)( particles ), 1,
            belements, until_turn, out_particles );
}

/* end: /common/internal/track.c */
