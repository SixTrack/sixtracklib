#include "sixtracklib/common/track.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/config.h"
#include "sixtracklib/common/internal/beam_elements_defines.h"
#include "sixtracklib/common/internal/particles_defines.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/buffer/buffer_type.h"


extern SIXTRL_HOST_FN int NS(Track_particle_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index );


extern SIXTRL_HOST_FN int NS(Track_subset_of_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index );


extern SIXTRL_HOST_FN int NS(Track_all_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index );

/* ------------------------------------------------------------------------- */

extern SIXTRL_HOST_FN int NS(Track_particle_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index_begin, NS(buffer_size_t) const be_index_end );

extern SIXTRL_HOST_FN int NS(Track_subset_of_particles_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_idx_begin, NS(buffer_size_t) const be_idx_end );

extern SIXTRL_HOST_FN int NS(Track_all_particles_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_idx_begin, NS(buffer_size_t) const be_idx_end );

/* ------------------------------------------------------------------------- */

extern SIXTRL_HOST_FN int NS(Track_particle_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements );

extern SIXTRL_HOST_FN int NS(Track_subset_of_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements );

extern SIXTRL_HOST_FN int NS(Track_all_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements );

/* ------------------------------------------------------------------------- */

extern SIXTRL_HOST_FN int NS(Track_particle_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const end_turn );

extern SIXTRL_HOST_FN int NS(Track_subset_of_particles_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const end_turn );

extern SIXTRL_HOST_FN int NS(Track_all_particles_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const end_turn );

/* ------------------------------------------------------------------------- */

extern SIXTRL_HOST_FN int NS(Track_particle_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

extern SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

extern SIXTRL_HOST_FN int NS(Track_all_particles_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

/* ------------------------------------------------------------------------- */

extern SIXTRL_HOST_FN int NS(Track_particle_element_by_element_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const end_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

extern SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_element_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const end_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

extern SIXTRL_HOST_FN int NS(Track_all_particles_element_by_element_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const end_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

/* ------------------------------------------------------------------------- */
/* ------ Implementation */
/* ------------------------------------------------------------------------- */

int NS(Track_particle_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    ptr_obj_t be_info = NS(Buffer_get_const_object)( beam_elements, be_index );
    SIXTRL_ASSERT( be_info != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_id >= ( NS(particle_index_t) )0u );

    SIXTRL_ASSERT( particle_idx >= ( NS(particle_num_elements_t) )0u );
    SIXTRL_ASSERT( particle_idx < NS(Particles_get_num_of_particles)( particles ) );

    return NS(Track_particle_beam_element_obj)(
        particles, particle_idx, be_info, be_id );
}

int NS(Track_subset_of_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index, NS(particle_index_t) const be_id )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    ptr_obj_t be_info = NS(Buffer_get_const_object)( beam_elements, be_index );
    SIXTRL_ASSERT( be_info != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_id >= ( NS(particle_index_t) )0u );

    SIXTRL_ASSERT( particle_idx_begin  >= ( NS(particle_num_elements_t) )0u );
    SIXTRL_ASSERT( particle_idx_stride >  ( NS(particle_num_elements_t) )0u );
    SIXTRL_ASSERT( particle_idx_end >= particle_idx_begin );
    SIXTRL_ASSERT( particle_idx_end <=
        NS(Particles_get_num_of_particles)( particles ) );

    return NS(Track_subset_of_particles_beam_element_obj)(
        particles, particle_idx_begin, particle_idx_end, particle_idx_stride,
        NS(Buffer_get_const_object)( beam_elements, be_index ), be_id );
}


int NS(Track_all_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index, NS(particle_index_t) const be_id )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    ptr_obj_t be_info = NS(Buffer_get_const_object)( beam_elements, be_index );
    SIXTRL_ASSERT( be_info != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_id >= ( NS(particle_index_t) )0u );

    return NS(Track_subset_of_particles_beam_element_obj)( particles, 0,
        NS(Particles_get_num_of_particles)( particles ), 1, be_info, be_id );
}

/* ------------------------------------------------------------------------- */

int NS(Track_particle_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index_begin, NS(buffer_size_t) const be_index_end,
    NS(particle_index_t) const be_id_begin )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    ptr_obj_t it  = NS(Buffer_get_const_object)( beam_elements, be_index_begin );
    ptr_obj_t end = NS(Buffer_get_const_object)( beam_elements, be_index_end );

    SIXTRL_ASSERT( be_index_begin <= be_index_end );
    SIXTRL_ASSERT( be_id_begin  >= ( NS(particle_index_t) )0u );
    SIXTRL_ASSERT( it  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( end != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( ( uintptr_t )end ) >= ( uintptr_t )it );

    return NS(Track_particle_subset_of_beam_element_objs)(
        particles, particle_idx, it, end, be_id_begin );
}

int NS(Track_subset_of_particles_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index_begin, NS(buffer_size_t) const be_index_end,
    NS(particle_index_t) const be_id_begin )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    ptr_obj_t it  = NS(Buffer_get_const_object)( beam_elements, be_index_begin );
    ptr_obj_t end = NS(Buffer_get_const_object)( beam_elements, be_index_end );

    SIXTRL_ASSERT( be_index_begin <= be_index_end );
    SIXTRL_ASSERT( be_id_begin  >= ( NS(particle_index_t) )0u );
    SIXTRL_ASSERT( it  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( end != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( ( uintptr_t )end ) >= ( uintptr_t )it );

    SIXTRL_ASSERT( particle_idx_begin >= ( NS(particle_num_elements_t) )0u );
    SIXTRL_ASSERT( particle_idx_end   >= particle_idx_begin );
    SIXTRL_ASSERT( particle_idx_end   <
        NS(Particles_get_num_of_particles)( particles ) );

    return NS(Track_particle_subset_of_beam_element_objs)(
        particles, particle_idx_begin, particle_idx_end, particle_idx_stride,
        it, end, be_id_begin );
}

int NS(Track_all_particles_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_begin_idx, NS(buffer_size_t) const be_end_idx )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    ptr_obj_t it  = NS(Buffer_get_const_object)( beam_elements, be_index_begin );
    ptr_obj_t end = NS(Buffer_get_const_object)( beam_elements, be_index_end );

    SIXTRL_ASSERT( be_index_begin <= be_index_end );
    SIXTRL_ASSERT( be_id_begin  >= ( NS(particle_index_t) )0u );
    SIXTRL_ASSERT( it  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( end != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( ( uintptr_t )end ) >= ( uintptr_t )it );

    return NS(Track_particle_subset_of_beam_element_objs)(
        particles, 0, NS(Particles_get_num_of_particles)( particles), 1,
        it, end, be_id_begin );
}

/* ------------------------------------------------------------------------- */

int NS(Track_particle_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements )
{
    return NS(Track_particle_beam_elements_obj)(
        particles, particle_index,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ) );
}


SIXTRL_INLINE int NS(Track_subset_of_particles_beam_elements)(
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


SIXTRL_INLINE int NS(Track_all_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements )
{
    return NS(Track_all_particles_beam_elements_obj)( particles,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ) );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Track_particle_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const end_turn )
{
    return NS(Track_particle_until_turn_obj)(
        particles, particle_index,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ), end_turn );
}

SIXTRL_INLINE int NS(Track_subset_of_particles_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const end_turn )
{
    return NS(Track_subset_of_particles_until_turn_obj)(
        particles, particle_idx_begin, particle_idx_end, particle_idx_stride,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ), end_turn );
}

SIXTRL_INLINE int NS(Track_all_particles_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const end_turn )
{
    return NS(Track_all_particles_until_turn_obj)( particles,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ), end_turn );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Track_particle_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_particle_buffer,
    NS(buffer_size_t) const out_particle_blocks_offset )
{
    int ret = -1;

    if( ( out_particle_blocks_offset +
          NS(Buffer_get_num_of_objects)( out_particle_buffer ) ) >=
        NS(Particles_buffer_get_num_of_particle_blocks)( out_particle_buffer ) )
    {
        ret = NS(Track_particle_element_by_element_obj)(
            particles, particle_idx, start_beam_element_id,
            NS(Buffer_get_const_objects_begin)( beam_elements ),
            NS(Buffer_get_const_objects_end)( beam_elements ),
            NS(Buffer_get_object)( out_particle_buffer, out_particle_blocks_offset ) );
    }

    return ret;
}

SIXTRL_INLINE int NS(Track_subset_of_particles_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_particle_buffer,
    NS(buffer_size_t) const out_particle_blocks_offset )
{
    return NS(Track_subset_of_particles_element_by_element_obj)(
        particles, particle_idx_begin, particle_idx_end, particle_idx_stride,
        start_beam_element_id,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ),
        NS(Buffer_get_object)( out_particle_buffer, out_particle_blocks_offset ) );
}

SIXTRL_INLINE int NS(Track_all_particles_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const start_beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_particle_buffer,
    NS(buffer_size_t) const out_particle_blocks_offset )
{
    return NS(Track_subset_of_particles_element_by_element)(
        particles, 0u, NS(Particles_get_num_of_particles)( particles ), 1u,
            start_beam_element_id, beam_elements,
                out_particle_buffer, out_particle_blocks_offset );
}

SIXTRL_INLINE int NS(Track_all_particles_append_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) beam_element_id,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_particle_buffer )
{
    typedef NS(particle_num_elements_t) num_elements_t;
    typedef NS(particle_index_t)        index_t;

    int ret = 0;

    NS(Object) const* obj_it  = NS(Buffer_get_const_objects_begin)( beam_elements );
    NS(Object) const* obj_end = NS(Buffer_get_const_objects_end)( beam_elements );

    for( ; obj_it != obj_end ; ++obj_it, ++beam_element_id )
    {
        num_elements_t const NUM_PARTICLES =
            NS(Particles_get_num_of_particles)( particles );

        num_elements_t ii = ( num_elements_t )0u;

        for( ; ii < NUM_PARTICLES ; ++ii )
        {
            if( NS(Particles_get_state_value)( particles, ii ) == ( index_t )1 )
            {
                NS(Particles_set_at_element_id_value)(
                    particles, ii, beam_element_id );
            }
        }

        NS(Particles)* elem_by_elem_dump = NS(Particles_add_copy)(
                out_particle_buffer, particles );

        if( ( elem_by_elem_dump == SIXTRL_NULLPTR ) ||
            ( 0 != NS(Track_all_particles_beam_element_obj)(
                particles, beam_element_id, obj_it ) ) )
        {
            ret = -1;
            break;
        }
    }

    return ret;
}

