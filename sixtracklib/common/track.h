#ifndef SIXTRACKLIB_COMMON_TRACK_H__
#define SIXTRACKLIB_COMMON_TRACK_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/track/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

struct NS(Particles);

SIXTRL_FN SIXTRL_STATIC void NS(Track_particle_increment_at_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index );

SIXTRL_FN SIXTRL_STATIC void NS(Track_subset_particles_increment_at_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_begin_index,
    NS(particle_num_elements_t) const particle_end_index,
    NS(particle_num_elements_t) const particle_index_stride );

SIXTRL_FN SIXTRL_STATIC void NS(Track_all_particles_increment_at_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN
void NS(Track_particle_increment_at_element_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index );

SIXTRL_EXTERN SIXTRL_HOST_FN
void NS(Track_subset_particles_increment_at_element_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_begin_index,
    NS(particle_num_elements_t) const particle_end_index,
    NS(particle_num_elements_t) const particle_index_stride );

SIXTRL_EXTERN SIXTRL_HOST_FN
void NS(Track_all_particles_increment_at_element_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)*
        SIXTRL_RESTRICT particles );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC void NS(Track_particle_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    NS(particle_index_t) const start_beam_element_id );

SIXTRL_FN SIXTRL_STATIC void NS(Track_subset_particles_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_begin_index,
    NS(particle_num_elements_t) const particle_end_index,
    NS(particle_num_elements_t) const particle_index_stride,
    NS(particle_index_t) const start_beam_element_id );

SIXTRL_FN SIXTRL_STATIC void NS(Track_all_particles_increment_at_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const start_beam_element_id );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN
void NS(Track_particle_increment_at_turn_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    NS(particle_index_t) const start_beam_element_id );

SIXTRL_EXTERN SIXTRL_HOST_FN
void NS(Track_subset_particles_increment_at_turn_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_begin_index,
    NS(particle_num_elements_t) const particle_end_index,
    NS(particle_num_elements_t) const particle_index_stride,
    NS(particle_index_t) const start_beam_element_id );

SIXTRL_EXTERN SIXTRL_HOST_FN void
NS(Track_all_particles_increment_at_turn_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const start_beam_element_id );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_beam_element_obj_dispatcher)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info );


SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) p_idx_begin,
    NS(particle_num_elements_t) const p_idx_end,
    NS(particle_num_elements_t) const p_idx_stride,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info );


SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_beam_elements_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end );


SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_beam_elements_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) p_idx_begin,
    NS(particle_num_elements_t) const p_idx_end,
    NS(particle_num_elements_t) const p_idx_stride,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end );


SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_beam_elements_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_until_turn_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn );

SIXTRL_FN SIXTRL_STATIC int NS(Track_subset_of_particles_until_turn_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) p_idx_begin,
    NS(particle_num_elements_t) const p_idx_end,
    NS(particle_num_elements_t) const p_idx_stride,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn );

SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_until_turn_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_element_by_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info );

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_element_by_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int
    NS(Track_subset_of_particles_element_by_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info );

SIXTRL_FN SIXTRL_STATIC int
    NS(Track_subset_of_particles_element_by_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_element_by_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info );

SIXTRL_FN SIXTRL_STATIC int NS(Track_all_particles_element_by_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int
NS(Track_particle_element_by_element_until_turn_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const index,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn );

SIXTRL_FN SIXTRL_STATIC int
NS(Track_subset_of_particles_element_by_element_until_turn_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn );

SIXTRL_FN SIXTRL_STATIC int
NS(Track_all_particles_element_by_element_until_turn_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN int NS(Track_particle_line)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT line_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT line_end,
    bool const finish_turn );

SIXTRL_STATIC SIXTRL_FN int NS(Track_subset_of_particles_line)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT line_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT line_end,
    bool const finish_turn );

SIXTRL_STATIC SIXTRL_FN int NS(Track_all_particles_line)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT line_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT line_end,
    bool const finish_turn );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */


#if !defined( _GPUCODE )

struct NS(Buffer);

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Track_particle_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const be_index );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Track_subset_of_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const p_idx_end,
    NS(particle_num_elements_t) const p_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const be_index );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Track_all_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const be_index );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Track_particle_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const be_index_begin,
    NS(buffer_size_t) const be_index_end );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(Track_subset_of_particles_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const be_index_begin,
    NS(buffer_size_t) const be_index_end );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(Track_all_particles_subset_of_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const be_begin_idx, NS(buffer_size_t) const be_end_idx );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Track_particle_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const*
        SIXTRL_RESTRICT belements );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Track_subset_of_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const*
        SIXTRL_RESTRICT belements );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Track_all_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const*
        SIXTRL_RESTRICT belements );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Track_particle_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT belements,
    NS(particle_index_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Track_subset_of_particles_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const p_idx_begin,
    NS(particle_num_elements_t) const p_idx_end,
    NS(particle_num_elements_t) const p_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT belements,
    NS(particle_index_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Track_all_particles_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC struct NS(Buffer) const* SIXTRL_RESTRICT belements,
    NS(particle_index_t) const until_turn );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Track_particle_element_by_element_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const be_index );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(Track_subset_of_particles_element_by_element_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const p_idx_end,
    NS(particle_num_elements_t) const p_idx_stride,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const be_index );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(Track_all_particles_element_by_element_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const be_index );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Track_particle_element_by_elements_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const be_begin_idx, NS(buffer_size_t) const be_end_idx );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(Track_subset_of_particles_element_by_elements_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const be_begin_idx, NS(buffer_size_t) const be_end_idx );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(Track_all_particles_element_by_elements_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const be_begin_idx, NS(buffer_size_t) const be_end_idx );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Track_particle_element_by_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer)
        *const SIXTRL_RESTRICT belements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(Track_subset_of_particles_element_by_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const p_idx_end,
    NS(particle_num_elements_t) const p_idx_stride,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT belements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Track_all_particles_element_by_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_t) const max_turn,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT belements,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(Track_particle_element_by_element_until_turn_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(particle_index_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(Track_subset_of_particles_element_by_element_until_turn_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const p_idx_end,
    NS(particle_num_elements_t) const p_idx_stride,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(particle_index_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(Track_all_particles_element_by_element_until_turn_details)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(particle_index_t) const until_turn );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(Track_particle_element_by_element_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(Track_subset_of_particles_element_by_element_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(Track_all_particles_element_by_element_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(particle_index_t) const until_turn,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT out_particles );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Track_particle_line_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const line_begin_idx,
    NS(buffer_size_t) const line_end_idx,
    bool const finish_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Track_subset_of_particles_line_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const line_begin_idx,
    NS(buffer_size_t) const line_end_idx,
    bool const finish_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Track_all_particles_line_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const line_begin_idx,
    NS(buffer_size_t) const line_end_idx,
    bool const finish_turn );

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
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/buffer_object_defines.h"
    #include "sixtracklib/common/beam_elements.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"

    #include "sixtracklib/common/be_drift/track.h"
    #include "sixtracklib/common/be_cavity/track.h"
    #include "sixtracklib/common/be_multipole/track.h"
    #include "sixtracklib/common/be_monitor/track.h"
    #include "sixtracklib/common/be_srotation/track.h"
    #include "sixtracklib/common/be_xyshift/track.h"
    #include "sixtracklib/common/be_limit/track.h"
    #include "sixtracklib/common/be_dipedge/track.h"

    #if !defined( SIXTRL_DISABLE_BEAM_BEAM )
        #include "sixtracklib/common/be_beamfields/track.h"
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

SIXTRL_INLINE void NS(Track_subset_particles_increment_at_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) p_idx,
    NS(particle_num_elements_t) const p_idx_end,
    NS(particle_num_elements_t) const p_idx_stride )
{
    SIXTRL_ASSERT( p_idx_stride >  ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( p_idx  >= ( NS(particle_num_elements_t) )0 );

    for( ; p_idx < p_idx_end ; p_idx += p_idx_stride )
    {
        if( NS(Particles_is_not_lost_value)( particles, p_idx ) )
        {
            NS(Particles_increment_at_element_id_value)( particles, p_idx );
        }
    }

    return;
}

SIXTRL_INLINE void NS(Track_all_particles_increment_at_element)(
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

        case NS(OBJECT_TYPE_LIMIT_RECT):
        {
            typedef NS(LimitRect) belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_limit_rect)( particles, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_LIMIT_ELLIPSE):
        {
            typedef NS(LimitEllipse) belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_limit_ellipse)( particles, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_DIPEDGE):
        {
            typedef NS(DipoleEdge) belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_dipedge)( particles, index, belem );
            break;
        }

        #if !defined( SIXTRL_DISABLE_BEAM_BEAM )

        case NS(OBJECT_TYPE_BEAM_BEAM_4D):
        {
            typedef NS(BeamBeam4D)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_beam_beam_4d)( particles, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_SPACE_CHARGE_COASTING):
        {
            typedef NS(SpaceChargeCoasting)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_space_charge_coasting)(
                particles, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_SPACE_CHARGE_BUNCHED):
        {
            typedef NS(SpaceChargeBunched)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_space_charge_bunched)(
                particles, index, belem );
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
            NS(Particles_set_state_value)( particles, index, 0 );
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
    int success = SIXTRL_TRACK_STATUS_GENERAL_FAILURE;

    if( ( NS(Particles_is_not_lost_value)( particles, index ) ) &&
        ( 0 == NS(Track_particle_beam_element_obj_dispatcher)(
            particles, index, be_info ) ) )
    {
        success = 0;
        NS(Particles_increment_at_element_id_value)( particles, index );
    }

    return success;
}


SIXTRL_INLINE int NS(Track_subset_of_particles_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) p_idx,
    NS(particle_num_elements_t) const p_idx_end,
    NS(particle_num_elements_t) const p_idx_stride,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info )
{
    SIXTRL_ASSERT( p_idx_stride >  ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( p_idx        >= ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( p_idx_end    >= p_idx );

    for( ; p_idx < p_idx_end ; p_idx += p_idx_stride )
    {
        if( NS(Particles_is_not_lost_value)( particles, p_idx ) )
        {
            NS(Track_particle_beam_element_obj)( particles, p_idx, be_info );
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
    int success = SIXTRL_TRACK_STATUS_GENERAL_FAILURE;

    SIXTRL_ASSERT( be_it  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_end != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( ( uintptr_t )be_it ) <= ( ( uintptr_t )be_end ) );

    success = ( ( NS(Particles_is_not_lost_value)( particles, index ) ) &&
                ( be_it != be_end ) )
            ? SIXTRL_TRACK_SUCCESS : SIXTRL_TRACK_STATUS_GENERAL_FAILURE;

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
    NS(particle_num_elements_t) const p_idx_end,
    NS(particle_num_elements_t) const p_idx_stride,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end )
{
    SIXTRL_ASSERT( begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( end   != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( ( uintptr_t )begin ) <= ( ( uintptr_t )end ) );

    SIXTRL_ASSERT( p_idx_stride > ( NS(particle_num_elements_t) )0u );
    SIXTRL_ASSERT( index >= ( NS(particle_num_elements_t) )0u );

    for( ; index < p_idx_end ; index += p_idx_stride )
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
    NS(particle_num_elements_t) const idx,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn )
{
    int success = 0;

    typedef NS(particle_index_t) index_t;

    index_t const start_beam_element_id =
        NS(Particles_get_at_element_id_value)( particles, idx );

    SIXTRL_ASSERT( be_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_end   != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( ( uintptr_t )be_begin ) <= ( ( uintptr_t )be_end ) );

    while( ( success == 0 ) &&
           ( NS(Particles_get_at_turn_value)( particles, idx ) < until_turn ) )
    {
        success =  NS(Track_particle_beam_elements_obj)(
            particles, idx, be_begin, be_end );

        if( success == 0 )
        {
            NS(Track_particle_increment_at_turn)(
                particles, idx, start_beam_element_id );
        }
    }

    return success;
}

SIXTRL_INLINE int NS(Track_subset_of_particles_until_turn_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) index,
    NS(particle_num_elements_t) const p_idx_end,
    NS(particle_num_elements_t) const p_idx_stride,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn )
{
    SIXTRL_ASSERT( index >= ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( p_idx_stride > ( NS(particle_num_elements_t) )0 );

    for( ; index < p_idx_end ; index += p_idx_stride )
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

    return SIXTRL_TRACK_SUCCESS;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Track_particle_element_by_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p,
    NS(particle_num_elements_t) const idx,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT conf,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info )
{
    int success = SIXTRL_TRACK_STATUS_GENERAL_FAILURE;

    typedef NS(particle_num_elements_t) num_elem_t;
    typedef SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* ptr_out_particles_t;

    if( NS(Particles_is_not_lost_value)( p, idx ) )
    {
        ptr_out_particles_t out_particles = ( ptr_out_particles_t )(
            uintptr_t )NS(ElemByElemConfig_get_output_store_address)( conf );

        if( out_particles == SIXTRL_NULLPTR )
        {
            success = SIXTRL_TRACK_SUCCESS;
        }
        else
        {
            num_elem_t const out_index =
                NS(ElemByElemConfig_get_particles_store_index)( conf, p, idx );

            success = ( ( NS(Particles_copy_single)(
                out_particles, out_index, p, idx ) ) ==
                    SIXTRL_ARCH_STATUS_SUCCESS )
                ? SIXTRL_TRACK_SUCCESS
                : SIXTRL_TRACK_STATUS_GENERAL_FAILURE;
        }

        success |= NS(Track_particle_beam_element_obj)( p, idx, be_info );
    }

    return success;
}

SIXTRL_INLINE  int NS(Track_particle_element_by_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const idx,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig)
        *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end )
{
    int success = SIXTRL_TRACK_STATUS_GENERAL_FAILURE;

    typedef NS(particle_index_t) index_t;
    typedef NS(particle_num_elements_t) num_elem_t;
    typedef SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* ptr_out_particles_t;

    ptr_out_particles_t out_particles = ( ptr_out_particles_t )(
            uintptr_t )NS(ElemByElemConfig_get_output_store_address)( config );

    SIXTRL_ASSERT( be_it  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_end != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( ( uintptr_t )be_end ) >= ( ( uintptr_t )be_it ) );

    if( out_particles != SIXTRL_NULLPTR )
    {
        index_t const at_turn =
            NS(Particles_get_at_turn_value)( particles, idx );

        index_t const temp =
            NS(Particles_get_particle_id_value)( particles, idx );

        index_t const particle_id = ( temp >= ( index_t )0u ) ? temp : -temp;

        success = ( NS(Particles_is_not_lost_value)( particles, idx ) )
            ? SIXTRL_TRACK_SUCCESS : SIXTRL_TRACK_STATUS_GENERAL_FAILURE;

        while( ( success == SIXTRL_TRACK_SUCCESS ) && ( be_it != be_end ) )
        {
            index_t const at_element_id =
                NS(Particles_get_at_element_id_value)( particles, idx );

            num_elem_t const out_index =
                NS(ElemByElemConfig_get_particles_store_index_details)(
                    config, particle_id, at_element_id, at_turn );

            if( NS(Particles_copy_single)(
                    out_particles, out_index, particles, idx ) ==
                  SIXTRL_ARCH_STATUS_SUCCESS )
            {
                success = NS(Track_particle_beam_element_obj)(
                    particles, idx, be_it++ );
            }
            else
            {
                success = SIXTRL_TRACK_STATUS_GENERAL_FAILURE;
            }
        }
    }
    else
    {
        success = ( NS(Particles_is_not_lost_value)( particles, idx ) );

        while( ( success == SIXTRL_TRACK_SUCCESS ) && ( be_it != be_end ) )
        {
            success = NS(Track_particle_beam_element_obj)(
                particles, idx, be_it++ );
        }
    }

    return success;
}


SIXTRL_INLINE  int NS(Track_subset_of_particles_element_by_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) p_idx,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig)
        *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info )
{
    SIXTRL_ASSERT( p_idx >= ( NS(particle_num_elements_t) )0u );
    SIXTRL_ASSERT( particle_idx_stride > ( NS(particle_num_elements_t) )0u );

    for( ; p_idx < particle_idx_end ; p_idx += particle_idx_stride )
    {
        NS(Track_particle_element_by_element_obj)(
            particles, p_idx, config, be_info );
    }

    return SIXTRL_TRACK_SUCCESS;
}

SIXTRL_INLINE  int NS(Track_subset_of_particles_element_by_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) p_idx,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig)
        *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end )
{
    SIXTRL_ASSERT( p_idx >= ( NS(particle_num_elements_t) )0u );
    SIXTRL_ASSERT( particle_idx_stride > ( NS(particle_num_elements_t) )0u );

    for( ; p_idx < particle_idx_end ; p_idx += particle_idx_stride )
    {
        NS(Track_particle_element_by_element_objs)(
            particles, p_idx, config, be_begin, be_end );
    }

    return SIXTRL_TRACK_SUCCESS;
}


SIXTRL_INLINE  int NS(Track_all_particles_element_by_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig)
        *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const nn = NS(Particles_get_num_of_particles)( particles );
    num_elem_t ii = ( num_elem_t )0u;

    for( ; ii < nn ; ++ii )
    {
        NS(Track_particle_element_by_element_obj)(
            particles, ii, config, be_info );
    }

    return SIXTRL_TRACK_SUCCESS;
}

SIXTRL_INLINE  int NS(Track_all_particles_element_by_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig)
        *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const nn = NS(Particles_get_num_of_particles)( particles );
    num_elem_t ii = ( num_elem_t )0u;

    for( ; ii < nn ; ++ii )
    {
        NS(Track_particle_element_by_element_objs)(
            particles, ii, config, be_begin, be_end );
    }

    return SIXTRL_TRACK_SUCCESS;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE  int NS(Track_particle_element_by_element_until_turn_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const idx,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig)
        *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn )
{
    typedef NS(particle_index_t) index_t;

    index_t const start_belement_id =
        NS(Particles_get_at_element_id_value)( particles, idx );

    int success = NS(Particles_is_not_lost_value)( particles, idx )
        ? SIXTRL_TRACK_SUCCESS : SIXTRL_TRACK_STATUS_GENERAL_FAILURE;

    while( ( success == SIXTRL_TRACK_SUCCESS ) &&
           ( NS(Particles_get_at_turn_value)( particles, idx ) < until_turn ) )
    {
        success = NS(Track_particle_element_by_element_objs)(
            particles, idx, config, be_begin, be_end );

        if( success == SIXTRL_TRACK_SUCCESS )
        {
            NS(Track_particle_increment_at_turn)(
                particles, idx, start_belement_id );
        }
    }

    return success;
}

SIXTRL_INLINE int
NS(Track_subset_of_particles_element_by_element_until_turn_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) p_idx,
    NS(particle_num_elements_t) const p_idx_end,
    NS(particle_num_elements_t) const p_idx_stride,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig)
        *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn )
{
    SIXTRL_ASSERT( p_idx_stride > ( NS(particle_num_elements_t) )0u );
    SIXTRL_ASSERT( p_idx >= ( NS(particle_num_elements_t) )0u );

    for( ; p_idx < p_idx_end ; p_idx += p_idx_stride )
    {
        NS(Track_particle_element_by_element_until_turn_objs)(
            particles, p_idx, config, be_begin, be_end, until_turn );
    }

    return SIXTRL_TRACK_SUCCESS;
}

SIXTRL_INLINE int NS(Track_all_particles_element_by_element_until_turn_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig)
        *const SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end,
    NS(particle_index_t) const until_turn )
{
    NS(particle_num_elements_t) const nn =
        NS(Particles_get_num_of_particles)( particles );

    NS(particle_num_elements_t) ii = ( NS(particle_num_elements_t) )0u;

    for( ; ii < nn ; ++ii )
    {
        NS(Track_particle_element_by_element_until_turn_objs)(
            particles, ii, config, be_begin, be_end, until_turn );
    }

    return SIXTRL_TRACK_SUCCESS;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Track_particle_line)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT line_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT line_end,
    bool const finish_turn )
{
    SIXTRL_ASSERT( line_it  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( line_end != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( uintptr_t )line_end >= ( ( uintptr_t )line_it ) );

    int status = SIXTRL_TRACK_SUCCESS;

    for( ; line_it != line_end ; ++line_it )
    {
        status |= NS(Track_particle_beam_element_obj)(
            particles, particle_idx, line_it );
    }

    if( finish_turn )
    {
        NS(Particles_set_at_element_id_value)( particles, particle_idx, 0 );
        NS(Particles_increment_at_turn_value)( particles, particle_idx );
    }

    return status;
}

SIXTRL_INLINE int NS(Track_subset_of_particles_line)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_idx_begin,
    NS(particle_num_elements_t) const particle_idx_end,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT line_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT line_end,
    bool const finish_turn )
{
    SIXTRL_ASSERT( line_it  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( line_end != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( uintptr_t )line_end >= ( ( uintptr_t )line_it ) );

    int status = SIXTRL_TRACK_SUCCESS;

    for( ; line_it != line_end ; ++line_it )
    {
        status |= NS(Track_subset_of_particles_beam_element_obj)(
            particles, particle_idx_begin, particle_idx_end,
                particle_idx_stride, line_it );
    }

    if( finish_turn )
    {
        NS(particle_num_elements_t) idx = particle_idx_begin;

        for( ; idx < particle_idx_end ; idx += particle_idx_stride )
        {
            NS(Particles_set_at_element_id_value)( particles, idx, 0 );
            NS(Particles_increment_at_turn_value)( particles, idx );
        }
    }

    return status;
}

SIXTRL_INLINE int NS(Track_all_particles_line)(
    SIXTRL_PARTICLE_ARGPTR_DEC struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT line_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT line_end,
    bool const finish_turn )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const begin_idx = ( num_elem_t )0u;
    num_elem_t const stride = ( num_elem_t )1u;

    return NS(Track_subset_of_particles_line)( particles, begin_idx,
        NS(Particles_get_num_of_particles)( particles ), stride, line_begin,
            line_end, finish_turn );
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_TRACK_H__ */

/* end: sixtracklib/common/track.h */
