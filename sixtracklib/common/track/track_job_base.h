#ifndef SIXTRL_SIXTRACKLIB_COMMON_TRACK_TRACK_JOB_BASE_H__
#define SIXTRL_SIXTRACKLIB_COMMON_TRACK_TRACK_JOB_BASE_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/track/track_job_base.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

struct NS(ElemByElemConfig);
struct NS(ParticlesAddr);

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
extern "C" {
#endif /* C++, Host */

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobBaseNew)* NS(TrackJobNew_create)(
    const char *const SIXTRL_RESTRICT arch_str,
    const char *const SIXTRL_RESTRICT config_str );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobBaseNew)* NS(TrackJobNew_new)(
    const char *const SIXTRL_RESTRICT arch,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    const char *const SIXTRL_RESTRICT config_str );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobBaseNew)* NS(TrackJobNew_new_with_output)(
    const char *const SIXTRL_RESTRICT arch,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    const char *const SIXTRL_RESTRICT config_str );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobBaseNew)* NS(TrackJobNew_new_detailed)(
    const char *const SIXTRL_RESTRICT arch,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    const char *const SIXTRL_RESTRICT config_str );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobNew_delete)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t) NS(TrackJobNew_track_until)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    NS(buffer_size_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(TrackJobNew_track_elem_by_elem)( NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    NS(buffer_size_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(TrackJobNew_track_line)( NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    NS(buffer_size_t) const beam_elem_begin_index,
    NS(buffer_size_t) const beam_elem_end_index,
    bool const finish_turn );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_job_collect_flag_t) NS(TrackJobNew_collect)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_job_collect_flag_t)
NS(TrackJobNew_collect_detailed)( NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    NS(track_job_collect_flag_t) const flags );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TrackJobNew_collect_particles)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TrackJobNew_collect_beam_elements)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(TrackJobNew_collect_output)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TrackJobNew_collect_debug_flag)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TrackJobNew_collect_particles_addresses)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobNew_enable_collect_particles)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobNew_disable_collect_particles)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobNew_is_collecting_particles)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobNew_enable_collect_beam_elements)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobNew_disable_collect_beam_elements)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobNew_is_collecting_beam_elements)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobNew_enable_collect_output)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobNew_disable_collect_output)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobNew_is_collecting_output)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_job_collect_flag_t)
NS(TrackJobNew_get_collect_flags)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobNew_set_collect_flags)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    NS(track_job_collect_flag_t) const flag );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobNew_requires_collecting)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_job_push_flag_t) NS(TrackJobNew_push)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    NS(track_job_push_flag_t) const flag );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_job_push_flag_t)
NS(TrackJobNew_push_particles)( NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_job_push_flag_t)
NS(TrackJobNew_push_beam_elements)( NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_job_push_flag_t)
NS(TrackJobNew_push_output)( NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(TrackJobNew_can_fetch_particle_addresses)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(TrackJobNew_has_particle_addresses)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TrackJobNew_fetch_particle_addresses)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TrackJobNew_clear_particle_addresses)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    NS(buffer_size_t) const particle_set_index );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TrackJobNew_clear_all_particle_addresses)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN struct NS(ParticlesAddr) const*
NS(TrackJobNew_get_particle_addresses)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    NS(buffer_size_t) const particle_set_index );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer) const*
NS(TrackJobNew_get_ptr_particle_addresses_buffer)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobNew_is_in_debug_mode)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TrackJobNew_enable_debug_mode)( NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TrackJobNew_disable_debug_mode)( NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobNew_clear)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(TrackJobNew_reset)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TrackJobNew_reset_particle_set)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const particle_set_index,
    NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TrackJobNew_reset_with_output)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(TrackJobNew_reset_detailed)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TrackJobNew_select_particle_set)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    NS(buffer_size_t) const particle_set_index );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TrackJobNew_assign_output_buffer)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT ptr_output_buffer );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_id_t) NS(TrackJobNew_get_arch_id)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobNew_has_arch_string)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN char const* NS(TrackJobNew_get_arch_string)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobNew_has_config_str)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN char const* NS(TrackJobNew_get_config_str)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJobNew_get_num_particle_sets)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t) const*
NS(TrackJobNew_get_particle_set_indices_begin)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t) const*
NS(TrackJobNew_get_particle_set_indices_end)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJobNew_get_particle_set_index)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    NS(buffer_size_t) const n );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJobNew_get_total_num_of_particles)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(particle_index_t)
NS(TrackJobNew_get_min_particle_id)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(particle_index_t)
NS(TrackJobNew_get_max_particle_id)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(particle_index_t)
NS(TrackJobNew_get_min_element_id)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(particle_index_t)
NS(TrackJobNew_get_max_element_id)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(particle_index_t)
NS(TrackJobNew_get_min_initial_turn_id)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(particle_index_t)
NS(TrackJobNew_get_max_initial_turn_id)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)* NS(TrackJobNew_get_particles_buffer)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer) const*
NS(TrackJobNew_get_const_particles_buffer)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)*
NS(TrackJobNew_get_beam_elements_buffer)( NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer) const*
NS(TrackJobNew_get_const_beam_elements_buffer)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobNew_has_output_buffer)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobNew_owns_output_buffer)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobNew_has_elem_by_elem_output)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobNew_has_beam_monitor_output)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJobNew_get_beam_monitor_output_buffer_offset)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJobNew_get_elem_by_elem_output_buffer_offset)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJobNew_get_num_elem_by_elem_turns)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)* NS(TrackJobNew_get_output_buffer)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer) const*
NS(TrackJobNew_get_const_output_buffer)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobNew_has_beam_monitors)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJobNew_get_num_beam_monitors)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t) const*
NS(TrackJobNew_get_beam_monitor_indices_begin)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t) const*
NS(TrackJobNew_get_beam_monitor_indices_end)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJobNew_get_beam_monitor_index)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    NS(buffer_size_t) const n );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobNew_has_elem_by_elem_config)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ElemByElemConfig) const*
NS(TrackJobNew_get_elem_by_elem_config)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobNew_is_elem_by_elem_config_rolling)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(TrackJobNew_get_default_elem_by_elem_config_rolling_flag)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN void
NS(TrackJobNew_set_default_elem_by_elem_config_rolling_flag)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job, bool const is_rolling_flag );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(elem_by_elem_order_t)
NS(TrackJobNew_get_elem_by_elem_config_order)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(elem_by_elem_order_t)
NS(TrackJobNew_get_default_elem_by_elem_config_order)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN void
NS(TrackJobNew_set_default_elem_by_elem_config_order)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    NS(elem_by_elem_order_t) const order );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobNew_uses_controller)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobNew_uses_arguments)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job );

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
} /* extern "C" { */
#endif /* C++, Host */

#endif /* SIXTRL_SIXTRACKLIB_COMMON_TRACK_TRACK_JOB_BASE_H__ */

/* end: sixtracklib/common/track/track_job_base.h */
