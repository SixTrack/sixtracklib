#include "sixtracklib/common/track.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/config.h"
#include "sixtracklib/common/internal/beam_elements_defines.h"
#include "sixtracklib/common/internal/particles_defines.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/be_monitor/output_buffer.h"
#include "sixtracklib/common/particles.h"


extern SIXTRL_HOST_FN int NS(Track_particle_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
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
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index_begin,
    NS(buffer_size_t) const be_index_end );

extern SIXTRL_HOST_FN int NS(Track_subset_of_particles_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index_begin,
    NS(buffer_size_t) const be_index_end );

extern SIXTRL_HOST_FN int NS(Track_all_particles_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_begin_idx, NS(buffer_size_t) const be_end_idx );

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
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements );

extern SIXTRL_HOST_FN int NS(Track_all_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements );

/* ------------------------------------------------------------------------- */

extern SIXTRL_HOST_FN int NS(Track_particle_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn );

extern SIXTRL_HOST_FN int NS(Track_subset_of_particles_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn );

extern SIXTRL_HOST_FN int NS(Track_all_particles_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn );

/* ------------------------------------------------------------------------- */

extern SIXTRL_HOST_FN int NS(Track_particle_element_by_element_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

extern SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_element_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

extern SIXTRL_HOST_FN int NS(Track_all_particles_element_by_element_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

/* ------------------------------------------------------------------------- */

extern SIXTRL_HOST_FN int NS(Track_particle_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

extern SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

extern SIXTRL_HOST_FN int NS(Track_all_particles_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

/* ------------------------------------------------------------------------- */

extern SIXTRL_HOST_FN int NS(Track_particle_element_by_elements_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

extern SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_elements_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

extern SIXTRL_HOST_FN int NS(Track_all_particles_element_by_elements_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

/* ------------------------------------------------------------------------- */

extern SIXTRL_HOST_FN int NS(Track_particle_element_by_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

extern SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

extern SIXTRL_HOST_FN int NS(Track_all_particles_element_by_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

/* ------------------------------------------------------------------------- */

extern SIXTRL_HOST_FN int NS(Track_particle_element_by_elements_until_turn_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

extern SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_elements_until_turn_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

extern SIXTRL_HOST_FN int NS(Track_all_particles_element_by_elements_until_turn_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

/* ------------------------------------------------------------------------- */

extern SIXTRL_HOST_FN int NS(Track_particle_element_by_elements_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

extern SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_elements_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

extern SIXTRL_HOST_FN int NS(Track_all_particles_element_by_elements_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

/* ------------------------------------------------------------------------- */

extern SIXTRL_HOST_FN SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* NS(TrackCpu)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output,
    int const until_turn, int const elem_by_elem_turns );

/* ------------------------------------------------------------------------- */
/* ------ Implementation */
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
    NS(particle_num_elements_t) const particle_idx,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order )
{
    return NS(Track_particle_element_by_element_obj)( particles, particle_idx,
        min_particle_id, max_particle_id, min_element_id, max_element_id,
        min_turn, max_turn, NS(Buffer_get_const_object)(
            beam_elements, be_index ), out_particles, order );
}

SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_element_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order )
{
    return NS(Track_subset_of_particles_element_by_element_obj)(
        particles, particle_idx, particle_idx_end, particle_idx_stride,
        min_particle_id, max_particle_id, min_element_id, max_element_id,
        min_turn, max_turn, NS(Buffer_get_const_object)(
            beam_elements, be_index ), out_particles, order );
}

SIXTRL_HOST_FN int NS(Track_all_particles_element_by_element_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order )
{
    return NS(Track_all_particles_element_by_element_obj)( particles,
        min_particle_id, max_particle_id, min_element_id, max_element_id,
        min_turn, max_turn, NS(Buffer_get_const_object)(
            beam_elements, be_index ), out_particles, order );
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const index,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles )
{
    typedef NS(particle_index_t) index_t;
    index_t min_particle_id = ( index_t )0;
    index_t max_particle_id = ( index_t )-1;

    int success = NS(Particles_get_min_max_particle_id)(
        particles, &min_particle_id, &max_particle_id );

    if( success == 0 )
    {
        success = NS(Track_particle_element_by_element_details)(
            particles, index, min_particle_id, max_particle_id,
            0, NS(Buffer_get_num_of_objects)( beam_elements ),
            0, max_turn, beam_elements, be_index, out_particles, 0 );
    }

    return success;
}

SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const idx,
    NS(particle_num_elements_t) const idx_end,
    NS(particle_num_elements_t) const stride,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles )
{
    typedef NS(particle_index_t) index_t;
    index_t min_particle_id = ( index_t )0;
    index_t max_particle_id = ( index_t )-1;

    int success = NS(Particles_get_min_max_particle_id)(
        particles, &min_particle_id, &max_particle_id );

    if( success == 0 )
    {
        success = NS(Track_subset_of_particles_element_by_element_details)(
            particles, idx, idx_end, stride, min_particle_id, max_particle_id,
            0, NS(Buffer_get_num_of_objects)( beam_elements ),
            0, max_turn, beam_elements, be_index, out_particles, 0 );
    }

    return success;
}

SIXTRL_HOST_FN int NS(Track_all_particles_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles )
{
    typedef NS(particle_index_t) index_t;
    index_t min_particle_id = ( index_t )0;
    index_t max_particle_id = ( index_t )-1;

    int success = NS(Particles_get_min_max_particle_id)(
        particles, &min_particle_id, &max_particle_id );

    if( success == 0 )
    {
        success = NS(Track_all_particles_element_by_element_details)(
            particles, min_particle_id, max_particle_id,
            ( NS(particle_index_t) )0, NS(Buffer_get_num_of_objects)( beam_elements ),
            ( NS(particle_index_t) )0, max_turn,
            beam_elements, be_index, out_particles, 0 );
    }

    return success;
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_element_by_elements_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const index,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order )
{
    return NS(Track_particle_element_by_element_objs)( particles, index,
        min_particle_id, max_particle_id, min_element_id, max_element_id,
        min_turn, max_turn,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ),
        out_particles, order );
}

SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_elements_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) idx,
    NS(particle_num_elements_t) const idx_end,
    NS(particle_num_elements_t) const stride,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order )
{
    return NS(Track_subset_of_particles_element_by_element_objs)(
        particles,idx, idx_end, stride,
        min_particle_id, max_particle_id,
        min_element_id, max_element_id,
        min_turn, max_turn,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ),
        out_particles, order );
}

SIXTRL_HOST_FN int NS(Track_all_particles_element_by_elements_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order )
{
    return NS(Track_all_particles_element_by_element_objs)( particles,
        min_particle_id, max_particle_id,
        min_element_id, max_element_id,
        min_turn, max_turn,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ),
        out_particles, order );
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_element_by_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const index,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles )
{
    typedef NS(particle_index_t) index_t;
    index_t min_particle_id = ( index_t )0;
    index_t max_particle_id = ( index_t )-1;

    int success = NS(Particles_get_min_max_particle_id)(
        particles, &min_particle_id, &max_particle_id );

    if( success == 0 )
    {
        success = NS(Track_particle_element_by_elements_details)(
            particles, index, min_particle_id, max_particle_id,
            ( index_t )0, NS(Buffer_get_num_of_objects)( beam_elements ),
            ( index_t )0, max_turn,
            beam_elements, out_particles, 0 );
    }

    return success;
}

SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) idx, NS(particle_num_elements_t) const idx_end,
    NS(particle_num_elements_t) const stride, NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles )
{
    typedef NS(particle_index_t) index_t;
    index_t min_particle_id = ( index_t )0;
    index_t max_particle_id = ( index_t )-1;

    int success = NS(Particles_get_min_max_particle_id)(
        particles, &min_particle_id, &max_particle_id );

    if( success == 0 )
    {
        success = NS(Track_subset_of_particles_element_by_elements_details)(
            particles, idx, idx_end, stride, min_particle_id, max_particle_id,
            0, NS(Buffer_get_num_of_objects)( beam_elements ),
            0, max_turn, beam_elements, out_particles, 0 );
    }

    return success;
}

SIXTRL_HOST_FN int NS(Track_all_particles_element_by_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles )
{
    typedef NS(particle_index_t) index_t;
    index_t min_particle_id = ( index_t )0;
    index_t max_particle_id = ( index_t )-1;

    int success = NS(Particles_get_min_max_particle_id)(
        particles, &min_particle_id, &max_particle_id );

    if( success == 0 )
    {
        success = NS(Track_all_particles_element_by_elements_details)(
            particles, min_particle_id, max_particle_id,
            0, NS(Buffer_get_num_of_objects)( beam_elements ),
            0, max_turn, beam_elements, out_particles, 0 );
    }

    return success;
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_element_by_elements_until_turn_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order )
{
    return NS(Track_particle_element_by_element_until_turn_objs)(
        particles, particle_idx, min_particle_id, max_particle_id, min_element_id,
        max_element_id, min_turn, max_turn,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ),
        until_turn, out_particles, order );
}

SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_elements_until_turn_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) idx,
    NS(particle_num_elements_t) const idx_end,
    NS(particle_num_elements_t) const stride,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order )
{
    return NS(Track_subset_of_particles_element_by_element_until_turn_objs)(
        particles, idx, idx_end, stride, min_particle_id,
        max_particle_id, min_element_id,
        max_element_id, min_turn, max_turn,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ),
        until_turn, out_particles, order );
}

SIXTRL_HOST_FN int NS(Track_all_particles_element_by_elements_until_turn_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order )
{
    return NS(Track_all_particles_element_by_element_until_turn_objs)(
        particles, min_particle_id, max_particle_id, min_element_id,
        max_element_id, min_turn, max_turn,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ),
        until_turn, out_particles, order );
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_element_by_elements_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles )
{
    typedef NS(particle_index_t) index_t;
    index_t min_particle_id = ( index_t )0;
    index_t max_particle_id = ( index_t )-1;

    int success = NS(Particles_get_min_max_particle_id)(
        particles, &min_particle_id, &max_particle_id );

    if( success == 0 )
    {
        success = NS(Track_particle_element_by_elements_until_turn_details)(
            particles, particle_idx, min_particle_id, max_particle_id,
            0, NS(Buffer_get_num_of_objects)( beam_elements ),
            0, until_turn, beam_elements, until_turn, out_particles, 0 );
    }

    return success;
}

SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_elements_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) idx,
    NS(particle_num_elements_t) const idx_end,
    NS(particle_num_elements_t) const stride,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles )
{
    typedef NS(particle_index_t) index_t;
    index_t min_particle_id = ( index_t )0;
    index_t max_particle_id = ( index_t )-1;

    int success = NS(Particles_get_min_max_particle_id)(
        particles, &min_particle_id, &max_particle_id );

    if( success == 0 )
    {
        success = NS(Track_subset_of_particles_element_by_elements_until_turn_details)(
            particles, idx, idx_end, stride, min_particle_id, max_particle_id,
            0, NS(Buffer_get_num_of_objects)( beam_elements ),
            0, until_turn, beam_elements, until_turn, out_particles, 0 );
    }

    return success;
}

SIXTRL_HOST_FN int NS(Track_all_particles_element_by_elements_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles )
{
    typedef NS(particle_index_t) index_t;
    index_t min_particle_id = ( index_t )0;
    index_t max_particle_id = ( index_t )-1;

    int success = NS(Particles_get_min_max_particle_id)(
        particles, &min_particle_id, &max_particle_id );

    if( success == 0 )
    {
        success = NS(Track_all_particles_element_by_elements_until_turn_details)(
            particles, min_particle_id, max_particle_id,
            0, NS(Buffer_get_num_of_objects)( beam_elements ),
            0, until_turn, beam_elements, until_turn, out_particles, 0 );
    }

    return success;
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* NS(TrackCpu)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output,
    int const until_turn, int const elem_by_elem_turns )
{
    NS(Buffer)* ptr_output = SIXTRL_NULLPTR;

    if( ( particles != SIXTRL_NULLPTR ) &&
        ( beam_elements != SIXTRL_NULLPTR ) &&
        ( until_turn >= 0 ) && ( elem_by_elem_turns >= 0 ) &&
        ( elem_by_elem_turns <= until_turn ) )
    {
        int success = -1;

        if( output != SIXTRL_NULLPTR )
        {
            ptr_output = output;
        }
        else
        {
            ptr_output = NS(Buffer_new)( 0u );
        }

        if( ptr_output != SIXTRL_NULLPTR )
        {
            success  = NS(BeamMonitor_prepare_particles_out_buffer)(
                    beam_elements, ptr_output, particles, elem_by_elem_turns );

            success |= NS(BeamMonitor_assign_particles_out_buffer)(
                    beam_elements, ptr_output, elem_by_elem_turns );

            if( ( success == 0 ) && ( elem_by_elem_turns > 0 ) )
            {
                SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* elem_by_elem_particles =
                    NS(Particles_buffer_get_particles)( ptr_output, 0u );

                SIXTRL_ASSERT( elem_by_elem_particles != SIXTRL_NULLPTR );
                SIXTRL_ASSERT( NS(Buffer_get_num_of_objects)( ptr_output ) >
                           ( NS(buffer_size_t) )0u );

                success = NS(Track_all_particles_element_by_elements_until_turn)(
                    particles, beam_elements, elem_by_elem_turns,
                        elem_by_elem_particles );
            }

            if( ( success == 0 ) && ( elem_by_elem_turns < until_turn ) )
            {
                success = NS(Track_all_particles_until_turn)(
                    particles, beam_elements, until_turn );
            }

            if( success != 0 )
            {
                NS(Buffer_delete)( ptr_output );
                ptr_output = SIXTRL_NULLPTR;
            }
        }
    }

    return ptr_output;
}

/* end: /common/internal/track.c */
