#ifndef SIXTRACKLIB_COMMON_TRACK_H__
#define SIXTRACKLIB_COMMON_TRACK_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_FN SIXTRL_STATIC void NS(Track_particle_increment_at_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index );

SIXTRL_FN SIXTRL_STATIC void NS(Track_subset_particles_increment_at_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_begin_index,
    NS(particle_num_elements_t) const particle_end_index,
    NS(particle_num_elements_t) const particle_index_stride );

SIXTRL_FN SIXTRL_STATIC void NS(Track_all_particles_increment_at_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC void NS(Track_particle_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    NS(particle_index_t) const start_beam_element_id );

SIXTRL_FN SIXTRL_STATIC void NS(Track_subset_particles_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_begin_index,
    NS(particle_num_elements_t) const particle_end_index,
    NS(particle_num_elements_t) const particle_index_stride,
    NS(particle_index_t) const start_beam_element_id );

SIXTRL_FN SIXTRL_STATIC void NS(Track_all_particles_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const start_beam_element_id );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_beam_element_obj_dispatcher)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info );


SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info );


SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_beam_elements_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end );


SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_beam_elements_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end );


SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_beam_elements_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_until_turn_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn );

SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_until_turn_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn );

SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_until_turn_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn );

/* ------------------------------------------------------------------------- */

typedef SIXTRL_UINT64_T NS(elem_by_elem_order_t);

SIXTRL_FN SIXTRL_STATIC NS(particle_num_elements_t)
NS(Track_element_by_element_get_out_particle_index)(
    NS(paritcle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const at_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    NS(particle_index_t) const at_turn,
    NS(elem_by_elem_order_t) const order );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_element_by_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    NS(paritcle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_element_by_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    NS(paritcle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_element_by_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(paritcle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_element_by_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(paritcle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_element_by_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(paritcle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_element_by_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(paritcle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int
NS(Track_particle_element_by_element_until_turn_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    NS(paritcle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

SIXTRL_FN SIXTRL_STATIC int
NS(Track_subset_of_particles_element_by_element_until_turn_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(paritcle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

SIXTRL_FN SIXTRL_STATIC int
NS(Track_all_particles_element_by_element_until_turn_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(paritcle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order );

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

struct NS(Buffer);

SIXTRL_HOST_FN int NS(Track_particle_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index );

SIXTRL_HOST_FN int NS(Track_subset_of_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index );

SIXTRL_HOST_FN int NS(Track_all_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index );

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index_begin,
    NS(buffer_size_t) const be_index_end );

SIXTRL_HOST_FN int NS(Track_subset_of_particles_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index_begin,
    NS(buffer_size_t) const be_index_end );

SIXTRL_HOST_FN int NS(Track_all_particles_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_begin_idx, NS(buffer_size_t) const be_end_idx );

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements );

SIXTRL_HOST_FN int NS(Track_subset_of_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements );

SIXTRL_HOST_FN int NS(Track_all_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements );

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn );

SIXTRL_HOST_FN int NS(Track_subset_of_particles_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn );

SIXTRL_HOST_FN int NS(Track_all_particles_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn );

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
    NS(elem_by_elem_order_t) const order );

SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_element_details)(
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
    NS(elem_by_elem_order_t) const order );

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

SIXTRL_HOST_FN int NS(Track_all_particles_element_by_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const be_index,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_element_by_elements_details)(
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

SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_elements_details)(
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
    NS(elem_by_elem_order_t) const order );

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_element_by_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

SIXTRL_HOST_FN int NS(Track_all_particles_element_by_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

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
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_elements_until_turn_details)(
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
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

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
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(Track_particle_element_by_elements_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

SIXTRL_HOST_FN int NS(Track_subset_of_particles_element_by_elements_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

SIXTRL_HOST_FN int NS(Track_all_particles_element_by_elements_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* NS(TrackCpu)(
    char const* SIXTRL_RESTRICT device_id_str,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output,
    int const until_turn, int const elem_by_elem_turns );

#endif /* defined( _GPUCODE ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====        Implementation of Inline functions and methods         ===== */
/* ========================================================================= */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
#endif /* #if !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/generated/config.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/beam_elements.h"
    #include "sixtracklib/common/buffer/buffer_type.h"

    #include "sixtracklib/common/be_drift/track.h"
    #include "sixtracklib/common/be_cavity/track.h"
    #include "sixtracklib/common/be_multipole/track.h"
    #include "sixtracklib/common/be_monitor/track.h"
    #include "sixtracklib/common/be_srotation/track.h"
    #include "sixtracklib/common/be_xyshift/track.h"

    #if !defined( SIXTRL_DISABLE_BEAM_BEAM )
        #include "sixtracklib/common/be_beambeam/track.h"
    #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE void NS(Track_particle_increment_at_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index )
{
    if( NS(Particles_is_not_lost_value)( particles, particle_index ) )
    {
        NS(Particles_increment_at_element_id_value)( particles, particle_index );
    }

    return;
}

SIXTRL_INLINE void NS(Track_subset_particles_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride )
{
    SIXTRL_ASSERT( particle_idx_stride >  ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( particle_idx  >= ( NS(particle_num_elements_t) )0 );

    for( ; particle_idx < particle_idx_end ; particle_idx += particle_idx_stride )
    {
        if( NS(Particles_is_not_lost_value)( particles, particle_idx ) )
        {
            NS(Particles_increment_at_element_id_value)( particles, particle_idx );
        }
    }

    return;
}

SIXTRL_INLINE void NS(Track_all_particles_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t ii = ( num_elem_t )0u;
    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    for( ; ii < num_particles ; ++ii )
    {
        if( NS(Particles_is_not_lost_value)( particles, ii ) )
        {
            NS(Particles_increment_at_element_id_value)( particles, ii );
        }
    }

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS(Track_particle_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    NS(particle_index_t) const start_beam_element_id )
{
    if( NS(Particles_get_state_value)( particles, particle_index ) )
    {
        NS(Particles_increment_at_turn_value)(
            particles, particle_index );

        NS(Particles_set_at_element_id_value)(
            particles, particle_index, start_beam_element_id );
    }

    return;
}

SIXTRL_INLINE void NS(Track_subset_particles_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_index,
    NS(particle_num_elements_t) const particle_end_index,
    NS(particle_num_elements_t) const particle_index_stride,
    NS(particle_index_t) const start_beam_element_id )
{
    SIXTRL_ASSERT( particle_index_stride >  ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( particle_index  >= ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( particle_index  <= particle_end_index );

    for( ; particle_index < particle_end_index ;
           particle_index += particle_index_stride )
    {
        if( NS(Particles_get_state_value)( particles, particle_index ) )
        {
            NS(Particles_increment_at_turn_value)(
                particles, particle_index );

            NS(Particles_set_at_element_id_value)(
                particles, particle_index, start_beam_element_id );
        }
    }

    return;
}

SIXTRL_INLINE void NS(Track_all_particles_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const start_beam_element_id )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t ii = ( num_elem_t )0u;
    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    for( ; ii < num_particles ; ++ii )
    {
        if( NS(Particles_get_state_value)( particles, ii ) )
        {
            NS(Particles_increment_at_turn_value)( particles, ii );

            NS(Particles_set_at_element_id_value)(
                particles, ii, start_beam_element_id );
        }
    }

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Track_particle_beam_element_obj_dispatcher)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info )
{
    typedef NS(object_type_id_t) type_id_t;
    typedef NS(buffer_addr_t)    address_t;

    int ret = 0;

    type_id_t const    type_id = NS(Object_get_type_id)( be_info );
    address_t const begin_addr = NS(Object_get_begin_addr)( be_info );

    SIXTRL_ASSERT( begin_addr != ( address_t )0u );
    SIXTRL_ASSERT( index >= ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( index < NS(Particles_get_num_of_particles)( particles ) );
    SIXTRL_ASSERT( NS(Particles_is_not_lost_value)( particles, index ) );

    switch( type_id )
    {
        case NS(OBJECT_TYPE_DRIFT):
        {
            typedef NS(Drift)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_drift)( particles, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_DRIFT_EXACT):
        {
            typedef NS(DriftExact)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_drift_exact)( particles, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_MULTIPOLE):
        {
            typedef NS(MultiPole)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_multipole)( particles, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_CAVITY):
        {
            typedef NS(Cavity)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_cavity)( particles, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_XYSHIFT):
        {
            typedef NS(XYShift)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_xy_shift)( particles, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_SROTATION):
        {
            typedef NS(SRotation)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_srotation)( particles, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_BEAM_MONITOR):
        {
            typedef NS(BeamMonitor)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_beam_monitor)( particles, index, belem );
            break;
        }

        #if !defined( SIXTRL_DISABLE_BEAM_BEAM )

        case NS(OBJECT_TYPE_BEAM_BEAM_4D):
        {
            typedef NS(BeamBeam4D)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_beam_beam_4d)(
                particles, particle_index, belem );
            break;
        }

        case NS(OBJECT_TYPE_BEAM_BEAM_6D):
        {
            typedef NS(BeamBeam6D)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_beam_beam_6d)( particles, index, belem );
            break;
        }

        #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */

        default:
        {
            NS(Particles_set_state_value)( particles, particle_index, 0 );
            ret = -8;
        }
    };

    return ret;
}


SIXTRL_INLINE int NS(Track_particle_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info )
{
    int success = -1;

    if( ( NS(Particles_is_not_lost_value)( particles, index ) ) &&
        ( 0 == NS(Track_particle_beam_element_obj_dispatcher)(
            particles, index, be_info ) ) )
    {
        NS(Particles_increment_at_element_value)( particles, index );
    }

    return success;
}


SIXTRL_INLINE int NS(Track_subset_of_particles_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(particle_index_t) const beam_element_id,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info )
{
    SIXTRL_ASSERT( particle_idx_stride >  ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( particle_idx        >= ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( particle_idx_end    >= particle_idx );

    for( ; particle_idx < particle_idx_end ; particle_idx += particle_idx_stride )
    {
        if( NS(Particles_is_not_lost_value)( particles, particle_idx ) )
        {
            NS(Track_particle_beam_element_obj)( particles, particle_idx, be_info );
        }
    }

    return 0;
}

SIXTRL_INLINE int NS(Track_all_particles_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const nn = NS(Particles_get_num_of_particles)( particles );
    num_elem_t ii = ( num_elem_t )0u;

    for( ; ii < nn ; ++ii )
    {
        NS(Track_particle_beam_element_obj)( particles, ii, be_info );
    }

    return 0;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Track_particle_beam_elements_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end )
{
    int success = -1;

    typedef NS(particle_index_t)  index_t;

    SIXTRL_ASSERT( be_it  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_end != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( ( uintptr_t )be_it ) <= ( ( uintptr_t )be_end ) );

    success = ( ( NS(Particles_is_not_lost_value)( particles, index ) ) &&
                ( be_it != be_end ) ) ? 0 : -1;

    while( ( success == 0 ) && ( be_it != be_end ) )
    {
        success = NS(Track_particle_beam_element_obj_dispatcher)(
            particles, index, be_it++ );

        SIXTRL_ASSERT(
            ( ( success == 0 ) &&
              ( NS(Particles_is_not_lost_value)( particles, index ) ) ) ||
            ( success != 0 ) );

        NS(Track_particle_increment_at_element)( particles, index );
    }

    return success;
}


SIXTRL_INLINE int NS(Track_subset_of_particles_beam_elements_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) index,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end )
{
    SIXTRL_ASSERT( begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( end   != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( ( uintptr_t )begin ) <= ( ( uintptr_t )end ) );

    SIXTRL_ASSERT( particle_idx_stride > ( NS(particle_num_elements_t) )0u );
    SIXTRL_ASSERT( index >= ( NS(particle_num_elements_t) )0u );

    for( ; index < particle_idx_end ; index += particle_idx_stride )
    {
        NS(Track_particle_beam_elements_obj)( particles, index, begin, end );
    }

    return 0;
}


SIXTRL_INLINE int NS(Track_all_particles_beam_elements_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end )
{
    NS(particle_num_elements_t) ii = ( NS(particle_num_elements_t) )0u;

    NS(particle_num_elements_t) const nn =
        NS(Particles_get_num_of_particles)( particles );

    for( ; ii < nn ; ++ii )
    {
        NS(Track_particle_beam_elements_obj)( particles, ii, begin, end );
    }

    return 0;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Track_particle_until_turn_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn )
{
    int success = 0;

    typedef NS(particle_index_t) index_t;

    index_t const start_beam_element_id =
        NS(Particles_get_at_element_id_value)( particles, index );

    SIXTRL_ASSERT( be_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_end   != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( ( uintptr_t )be_begin ) <= ( ( uintptr_t )be_end ) );

    while( ( success == 0 ) &&
           ( NS(Particles_get_at_turn_value)( particles, index ) < until_turn ) )
    {
        success =  NS(Track_particle_beam_elements_obj)(
            particles, index, be_begin, be_end );

        if( success == 0 )
        {
            NS(Track_particle_increment_at_turn)(
                particle, index, start_beam_element_id );
        }
    }

    return success;
}

SIXTRL_INLINE int NS(Track_subset_of_particles_until_turn_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) index,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn )
{
    SIXTRL_ASSERT( index >= ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( particle_idx_stride > ( NS(particle_num_elements_t) )0 );

    for( ; index < particle_idx_end ; index += particle_idx_stride )
    {
        NS(Track_particle_until_turn_obj)(
            particles, index, be_begin, be_end, until_turn );
    }

    return 0;
}

SIXTRL_INLINE int NS(Track_all_particles_until_turn_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    num_elem_t ii = ( num_elem_t )0u;

    for( ; ii < num_particles ; ++ii )
    {
        NS(Track_particle_until_turn_obj)(
            particles, ii, be_begin, be_end, until_turn );
    }

    return 0;
}

/* ------------------------------------------------------------------------- */


typedef SIXTRL_UINT64_T NS(elem_by_elem_order_t);

SIXTRL_FN SIXTRL_STATIC NS(particle_num_elements_t)
NS(Track_element_by_element_get_out_particle_index)(
    NS(paritcle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const at_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    NS(particle_index_t) const at_turn,
    NS(elem_by_elem_order_t) const order )
{
    typedef NS(particle_num_elements_t) num_elem_t;
    typedef NS(particle_index_t)        index_t;

    SIXTRL_STATIC_VAR index_t const ZERO_INDEX = ( index_t )0u;
    SIXTRL_STATIC_VAR num_elem_t const ONE = ( num_elem_t )1u;

    num_elem_t out_index = ( num_elem_t )-1;

    if( ( min_particle_id >= ZERO_INDEX    ) &&
        ( min_particle_id <= particle_id   ) &&
        ( max_particle_id >= particle_id   ) &&
        ( min_element_id  >= ZERO_INDEX    ) &&
        ( min_element_id  <= at_element_id ) &&
        ( max_element_id  >= at_element_id ) &&
        ( min_turn >= ZERO_INDEX ) &&
        ( min_turn <= at_turn    ) &&
        ( max_turn >= at_turn    ) )
    {
        num_elem_t const particles_to_store =
            ( num_elem_t )( max_particle_id - min_particle_id ) + ONE;

        num_elem_t const particle_id_offset =
            ( num_elem_t )( particle_id - min_particle_id );

        num_elem_t const elements_to_store =
            ( num_elem_t )( max_element_id - min_element_id ) + ONE;

        num_elem_t const element_id_offset =
            ( num_elem_t )( at_element_id - min_element_id );

        num_elem_t const turns_to_store =
            ( num_elem_t )( max_turn - min_turn ) + ONE;

        num_elem_t const turn_id_offset = ( num_elem_t )( at_turn - min_turn );

        switch( order )
        {
            case 0:
            {
                /* 1st: turn
                 * 2nd: element
                 * 3rd: particle_id */

                out_index  = turn_id_offset * elements_to_store * particles_to_store;
                out_index += element_id_offset * particles_to_store;
                out_index += particle_id_offset;

                break;
            }

            default:
            {
                out_index = ( num_elem_t )-1;
            }
        };
    }

    return out_index;
}

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_element_by_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const index,
    NS(paritcle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order )
{
    int success = -1;

    typedef NS(particle_index_t) index_t;
    typedef NS(particle_num_elements_t) num_elem_t;

    index_t const temp_particle_id = NS(Particles_get_particle_id_value)(
            particles, index );

    index_t const particle_id = ( temp_particle_id >= ( index_t )0u )
        ? temp_particle_id : -temp_particle_id;

    num_elem_t const out_index =
    NS(Track_element_by_element_get_out_particle_index)(
        min_particle_id, max_particle_id, particle_id,
        min_element_id,  max_element_id, NS(Particles_get_at_element_id_value)(
            patricles, index ),
        min_turn, max_turn, NS(Particles_get_at_turn_value)(
            particles, index ),
        order );

    if( ( NS(Particles_is_not_lost_value)( particles, index ) ) &&
        ( 0 == NS(Particles_copy_single)(
            out_particles, out_index, particles, index ) ) )
    {
        success = NS(Track_particle_beam_element_obj)(
            particles, index, be_info );
    }

    return success;
}

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_element_by_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const index,
    NS(paritcle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order )
{
    int success = -1;

    typedef NS(particle_index_t) index_t;
    typedef NS(particle_num_elements_t) num_elem_t;

    index_t const at_turn =
        NS(Particles_get_at_turn_value)( particles, index );

    index_t const temp_particle_id =
        NS(Particles_get_particle_id_value)( particles, index );

    index_t const particle_id = ( temp_particle_id >= ( index_t )0u )
        ? temp_particle_id : -temp_particle_id;

    SIXTRL_ASSERT( at_turn >= ( index_t )0u );
    SIXTRL_ASSERT( at_turn <= max_turn );
    SIXTRL_ASSERT( at_turn >= min_turn );

    SIXTRL_ASSERT( min_particle_id >= ( index_t )0u );
    SIXTRL_ASSERT( min_particle_id <= particle_id );
    SIXTRL_ASSERT( max_particle_id >= particle_id );

    SIXTRL_ASSERT( be_it  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_end != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( ( uintptr_t )be_end ) >= ( ( uintptr_t )be_it ) );

    SIXTRL_ASSERT( min_element_id >= ( index_t )0u );
    SIXTRL_ASSERT( min_element_id <= max_element_id );
    SIXTRL_ASSERT( ( ( uintptr_t )( max_element_id - min_element_id + 1 ) ) >=
        ( uintptr_t )( be_end - be_it ) );

    success = ( ( NS(Particles_is_not_lost_value)( particles, index ) ) &&
                ( be_it != be_end ) ) ? 0 : -1;

    while( ( success == 0 ) && ( be_it != be_end ) )
    {
        index_t const at_element_id =
            NS(Particles_get_at_element_id_value)( patricles, index );

        num_elem_t const out_index =
            NS(Track_element_by_element_get_out_particle_index)(
                min_particle_id, max_particle_id, particle_id,
                min_element_id,  max_element_id, at_element_id,
                min_turn, max_turn, at_turn, order );

        success = (
            ( NS(Particles_copy_single)( out_particles, out_index, particles, index ) ) &&
            ( 0 == NS(Track_particle_beam_element_obj)( particles, index, be_it++ ) ) )
            ? 0 : -1;
    }

    return success;
}


SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_element_by_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(paritcle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order )
{
    SIXTRL_ASSERT( particle_idx >= ( NS(particle_num_elements_t) )0u );
    SIXTRL_ASSERT( particle_idx_stride > ( NS(particle_num_elements_t) )0u );

    for( ; particle_idx < particle_idx_end ; particle_idx += particle_idx_stride )
    {
        NS(Track_particle_element_by_element_obj)(
            particles, particle_idx, min_particle_id, max_particle_id,
            min_element_id, max_element_id, min_turn, max_turn, be_info,
            out_particles, order );
    }

    return 0;
}

SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_element_by_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(paritcle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order )
{
    SIXTRL_ASSERT( particle_idx >= ( NS(particle_num_elements_t) )0u );
    SIXTRL_ASSERT( particle_idx_stride > ( NS(particle_num_elements_t) )0u );

    for( ; particle_idx < particle_idx_end ; particle_idx += particle_idx_stride )
    {
        NS(Track_particle_element_by_element_objs)(
            particles, particle_idx, min_particle_id, max_particle_id,
            min_element_id, max_element_id, min_turn, max_turn, be_begin, be_end,
            out_particles, order );
    }

    return 0;
}


SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_element_by_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(paritcle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const nn = NS(Particles_get_num_of_particles)( particles );
    num_elem_t ii = ( num_elem_t )0u;

    for( ; ii < nn ; ++ii )
    {
        NS(Track_particle_element_by_element_obj)( particles, ii,
            min_particle_id, max_particle_id, min_element_id, max_element_id,
            min_turn, max_turn, be_info, out_particles, order );
    }

    return 0;
}

SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_element_by_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(paritcle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const nn = NS(Particles_get_num_of_particles)( particles );
    num_elem_t ii = ( num_elem_t )0u;

    for( ; ii < nn ; ++ii )
    {
        NS(Track_particle_element_by_element_objs)( particles, ii,
            min_particle_id, max_particle_id, min_element_id, max_element_id,
            min_turn, max_turn, be_begin, be_end, out_particles, order );
    }

    return 0;
}

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int
NS(Track_particle_element_by_element_until_turn_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const index,
    NS(paritcle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order )
{
    typedef NS(particle_num_elements_t) num_elem_t;
    typedef NS(particle_index_t)        index_t;

    index_t const start_beam_element_id =
        NS(Particles_get_at_element_id_value)( particles, index );

    int success = 0;

    while( ( success == 0 ) &&
           ( NS(Particles_get_at_turn_value)( particles, index ) < until_turn ) )
    {
        success = NS(Track_particle_element_by_element_objs)(
            particles, index, min_particle_id, max_particle_id, min_element_id,
            max_element_id, min_turn, max_turn, be_begin, be_end, out_particles,
            order );

        if( success == 0 )
        {
            NS(Track_particle_increment_at_turn)(
                particles, index, start_beam_element_id );
        }
    }

    return success;
}

SIXTRL_FN SIXTRL_STATIC int
NS(Track_subset_of_particles_element_by_element_until_turn_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    NS(paritcle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order )
{
    SIXTRL_ASSERT( particle_idx_stride > ( NS(particle_num_elements_t) )0u );
    SIXTRL_ASSERT( particle_idx >= ( NS(particle_num_elements_t) )0u );

    for( ; particle_idx < particle_idx_end ; particle_idx += particle_idx_stride )
    {
        NS(Track_particle_element_by_element_until_turn_objs)( particles, particle_idx,
            min_particle_id, max_particle_id, min_element_id, max_element_id,
            min_turn, max_turn, be_begin, be_end, until_turn, out_particles, order );
    }

    return 0;
}

SIXTRL_FN SIXTRL_STATIC int
NS(Track_all_particles_element_by_element_until_turn_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(paritcle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles,
    NS(elem_by_elem_order_t) const order )
{
    NS(particle_num_elements_t) const nn =
        NS(Particles_get_num_of_particles)( particles );

    NS(particle_num_elements_t) ii = ( NS(particle_num_elements_t) )0u;

    for( ; ii < nn ; ++ii )
    {
        NS(Track_particle_element_by_element_until_turn_objs)( particles, ii,
            min_particle_id, max_particle_id, min_element_id, max_element_id,
            min_turn, max_turn, be_begin, be_end, until_turn, out_particles, order );
    }

    return 0;
}

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_TRACK_H__ */

/* end: sixtracklib/common/track.h */
