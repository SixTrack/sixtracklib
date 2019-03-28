#ifndef SIXTRACKLIB_SIXTRACKLIB_COMMON_TRACK_JOB_H__
#define SIXTRACKLIB_SIXTRACKLIB_COMMON_TRACK_JOB_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/internal/track_job_base.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

struct NS(ElemByElemConfig);

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJob_clear)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJob_collect)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJob_reset)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJob_reset_with_output)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const dump_elem_by_elem_turns );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJob_reset_detailed)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const dump_elem_by_elem_turns );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJob_assign_output_buffer)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT ptr_output_buffer );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_job_type_t) NS(TrackJob_get_type_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN char const* NS(TrackJob_get_type_str)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJob_has_device_id_str)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN char const* NS(TrackJob_get_device_id_str)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJob_has_config_str)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN char const* NS(TrackJob_get_config_str)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJob_get_num_particle_sets)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t) const*
NS(TrackJob_get_particle_set_indices_begin)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t) const*
NS(TrackJob_get_particle_set_indices_end)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJob_get_particle_set_index)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job,
    NS(buffer_size_t) const n );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(particle_index_t)
NS(TrackJob_get_min_particle_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(particle_index_t)
NS(TrackJob_get_max_particle_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(particle_index_t)
NS(TrackJob_get_min_element_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(particle_index_t)
NS(TrackJob_get_max_element_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(particle_index_t)
NS(TrackJob_get_min_initial_turn_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(particle_index_t)
NS(TrackJob_get_max_initial_turn_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)* NS(TrackJob_get_particles_buffer)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer) const*
NS(TrackJob_get_const_particles_buffer)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)*
NS(TrackJob_get_beam_elements_buffer)( NS(TrackJobBase)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer) const*
NS(TrackJob_get_const_beam_elements_buffer)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJob_has_output_buffer)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJob_owns_output_buffer)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJob_has_elem_by_elem_output)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJob_has_beam_monitor_output)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJob_get_beam_monitor_output_buffer_offset)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJob_get_elem_by_elem_output_buffer_offset)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJob_get_num_elem_by_elem_turns)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)* NS(TrackJob_get_output_buffer)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer) const*
NS(TrackJob_get_const_output_buffer)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJob_has_beam_monitors)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJob_get_num_beam_monitors)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t) const*
NS(TrackJob_get_beam_monitor_indices_begin)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t) const*
NS(TrackJob_get_beam_monitor_indices_end)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJob_get_beam_monitor_index)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job,
    NS(buffer_size_t) const n );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJob_has_elem_by_elem_config)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ElemByElemConfig) const*
NS(TrackJob_get_elem_by_elem_config)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJob_is_elem_by_elem_config_rolling)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(TrackJob_get_default_elem_by_elem_config_rolling_flag)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN void
NS(TrackJob_set_default_elem_by_elem_config_rolling_flag)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job,
    bool const is_rolling_flag );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(elem_by_elem_order_t)
NS(TrackJob_get_elem_by_elem_config_order)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(elem_by_elem_order_t)
NS(TrackJob_get_default_elem_by_elem_config_order)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN void
NS(TrackJob_set_default_elem_by_elem_config_order)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job,
    NS(elem_by_elem_order_t) const order );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_SIXTRACKLIB_COMMON_TRACK_JOB_H__ */

/* end: sixtracklib/common/track_job.h */
