#ifndef SIXTRACKL_COMMON_BE_MONITOR_BE_MONITOR_C99_HEADER_H__
#define SIXTRACKL_COMMON_BE_MONITOR_BE_MONITOR_C99_HEADER_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/particles/definitions.h"
    #include "sixtracklib/common/control/debug_register.h"
    #include "sixtracklib/common/internal/compiler_attributes.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/buffer_object_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */

typedef SIXTRL_INT64_T      NS(be_monitor_turn_t);
typedef SIXTRL_INT64_T      NS(be_monitor_flag_t);
typedef SIXTRL_INT64_T      NS(be_monitor_index_t);
typedef NS(buffer_addr_t)   NS(be_monitor_addr_t);

typedef struct NS(BeamMonitor)
{
    NS(be_monitor_turn_t)   num_stores        SIXTRL_ALIGN( 8 );
    NS(be_monitor_turn_t)   start             SIXTRL_ALIGN( 8 );
    NS(be_monitor_turn_t)   skip              SIXTRL_ALIGN( 8 );
    NS(be_monitor_addr_t)   out_address       SIXTRL_ALIGN( 8 );
    NS(be_monitor_index_t)  max_particle_id   SIXTRL_ALIGN( 8 );
    NS(be_monitor_index_t)  min_particle_id   SIXTRL_ALIGN( 8 );
    NS(be_monitor_flag_t)   is_rolling        SIXTRL_ALIGN( 8 );
    NS(be_monitor_flag_t)   is_turn_ordered   SIXTRL_ALIGN( 8 );
}
NS(BeamMonitor);

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*
NS(BeamMonitor_preset)( SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*
    SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(BeamMonitor_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT
        monitor ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(BeamMonitor_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(BeamMonitor_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t) NS(BeamMonitor_type_id)(
    void ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(be_monitor_turn_t) NS(BeamMonitor_num_stores)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_monitor_turn_t) NS(BeamMonitor_start)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_monitor_turn_t) NS(BeamMonitor_skip)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(BeamMonitor_is_rolling)( SIXTRL_BE_ARGPTR_DEC
    const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(BeamMonitor_is_turn_ordered)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(BeamMonitor_is_particle_ordered)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_monitor_addr_t) NS(BeamMonitor_out_address)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_monitor_index_t) NS(BeamMonitor_min_particle_id)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_monitor_index_t) NS(BeamMonitor_max_particle_id)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(BeamMonitor_stored_num_particles)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(BeamMonitor_has_turn_stored)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor,
    NS(be_monitor_turn_t) const turn_id,
    NS(be_monitor_turn_t) const max_num_turns ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(BeamMonitor_set_num_stores)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_turn_t) const num_stores ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(BeamMonitor_set_start)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_turn_t) const start )SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(BeamMonitor_set_skip)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_turn_t) const skip ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(BeamMonitor_set_is_rolling)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    bool const is_rolling ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(BeamMonitor_set_is_turn_ordered)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    bool const is_turn_ordered ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(BeamMonitor_set_is_particle_ordered)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    bool const is_particle_ordered ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(BeamMonitor_set_out_address)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_addr_t) const out_address ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(BeamMonitor_set_min_particle_id)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_index_t) const min_particle_id ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(BeamMonitor_set_max_particle_id)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_index_t) const min_particle_id ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(BeamMonitor_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT source ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(BeamMonitor_get_beam_monitor_indices_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_be_monitors,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(BeamMonitor_get_beam_monitor_indices_from_index_range)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* index_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* index_end,
    NS(buffer_size_t) const start_index,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT ptr_num_be_monitors )SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor) const*
NS(BeamMonitor_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object)
    *const SIXTRL_RESTRICT index_obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*
NS(BeamMonitor_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
    SIXTRL_RESTRICT index_obj ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor) const*
NS(BeamMonitor_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*
NS(BeamMonitor_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN bool NS(BeamMonitor_are_present_in_obj_index_range)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_end
) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(BeamMonitor_are_present_in_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BeamMonitor_num_monitors_in_obj_index_range)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_end
) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BeamMonitor_num_monitors_in_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BeamMonitor_monitor_indices_from_obj_index_range)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_end,
    NS(buffer_size_t) const start_index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BeamMonitor_monitor_indices_from_managed_buffer)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(BeamMonitor_reset)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor
) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(BeamMonitor_reset_all_in_obj_index_range)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT obj_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT obj_end
) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(BeamMonitor_reset_all_in_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor) const*
NS(BeamMonitor_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*
NS(BeamMonitor_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN bool NS(BeamMonitor_are_present_in_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT buffer ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BeamMonitor_num_monitors_in_buffer)( SIXTRL_BUFFER_ARGPTR_DEC const
    NS(Buffer) *const SIXTRL_RESTRICT buffer ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BeamMonitor_monitor_indices_from_buffer)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer
) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(BeamMonitor_reset_all_in_buffer)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
    SIXTRL_RESTRICT buffer ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(BeamMonitor_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(BeamMonitor_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(BeamMonitor_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(object_type_id_t)
NS(BeamMonitor_type_id_ext)( void ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(BeamMonitor_are_present_in_obj_index_range_ext)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_end
) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(BeamMonitor_are_present_in_managed_buffer_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(BeamMonitor_are_present_in_buffer_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT buffer ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BeamMonitor_num_monitors_in_obj_index_range_ext)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_end
) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BeamMonitor_num_monitors_in_managed_buffer_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BeamMonitor_num_monitors_in_buffer_ext)( SIXTRL_BUFFER_ARGPTR_DEC const
    NS(Buffer) *const SIXTRL_RESTRICT buffer ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BeamMonitor_monitor_indices_from_obj_index_range_ext)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_end,
    NS(buffer_size_t) const start_index ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BeamMonitor_monitor_indices_from_managed_buffer_ext)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BeamMonitor_monitor_indices_from_buffer_ext)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer
) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(BeamMonitor_reset_ext)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor
) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(BeamMonitor_reset_all_in_obj_index_range_ext)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT obj_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT obj_end
) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(BeamMonitor_reset_all_in_managed_buffer_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(BeamMonitor_reset_all_in_buffer_ext)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
    SIXTRL_RESTRICT buffer ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(BeamMonitor_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        requ_dataptrs ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*
NS(BeamMonitor_new)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
    SIXTRL_RESTRICT buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*
NS(BeamMonitor_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(be_monitor_turn_t) const num_stores, NS(be_monitor_turn_t) const start,
    NS(be_monitor_turn_t) const skip,  NS(be_monitor_addr_t) const out_address,
    NS(be_monitor_index_t) const min_particle_id,
    NS(be_monitor_index_t) const max_particle_id,
    bool const is_rolling, bool const is_turn_ordered );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*
NS(BeamMonitor_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(BeamMonitor_insert_end_of_turn_monitors)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(be_monitor_turn_t) const turn_by_turn_start,
    NS(be_monitor_turn_t) const num_turn_by_turn_turns,
    NS(be_monitor_turn_t) const target_num_turns,
    NS(be_monitor_turn_t) const skip_turns,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT prev_node );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(BeamMonitor_insert_end_of_turn_monitors_at_pos)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(be_monitor_turn_t) const turn_by_turn_start,
    NS(be_monitor_turn_t) const num_turn_by_turn_turns,
    NS(be_monitor_turn_t) const target_num_turns,
    NS(be_monitor_turn_t) const skip_turns,
    NS(buffer_size_t) const insert_at_index );

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* *****          Implementation of C inline functions                   *** */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #if !defined( _GPUCODE )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* NS(BeamMonitor_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT
        monitor ) SIXTRL_NOEXCEPT
{
    if( monitor != SIXTRL_NULLPTR ) NS(BeamMonitor_clear)( monitor );
    return monitor;
}

SIXTRL_INLINE NS(arch_status_t) NS(BeamMonitor_clear)( SIXTRL_BE_ARGPTR_DEC
    NS(BeamMonitor)* SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = NS(BeamMonitor_set_num_stores)( monitor, 0 );
    status |= NS(BeamMonitor_set_start)( monitor, 0 );
    status |= NS(BeamMonitor_set_skip)( monitor, 1 );
    status |= NS(BeamMonitor_set_is_rolling)( monitor, false );
    status |= NS(BeamMonitor_set_is_turn_ordered)( monitor, true );

    status |= NS(BeamMonitor_set_out_address)(
        monitor, ( NS(be_monitor_addr_t) )0 );

    status |= NS(BeamMonitor_set_min_particle_id)(
        monitor, ( NS(be_monitor_index_t) )-1 );

    status |= NS(BeamMonitor_set_max_particle_id)(
        monitor, ( NS(be_monitor_index_t) )-1 );

    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(BeamMonitor_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( monitor ) ) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(BeamMonitor_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT
        SIXTRL_UNUSED( monitor ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(buffer_size_t) num_slots = ( NS(buffer_size_t) )0u;
    NS(buffer_size_t) const num_bytes = NS(ManagedBuffer_get_slot_based_length)(
                sizeof( NS(BeamMonitor) ), slot_size );

    SIXTRL_ASSERT( slot_size > ( NS(buffer_size_t) )0 );
    num_slots = num_bytes / slot_size;
    if( num_slots * slot_size < num_bytes ) ++num_slots;
    return num_slots;
}

SIXTRL_INLINE NS(object_type_id_t)
    NS(BeamMonitor_type_id)( void ) SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(OBJECT_TYPE_BEAM_MONITOR);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(be_monitor_turn_t) NS(BeamMonitor_num_stores)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    return monitor->num_stores;
}

SIXTRL_INLINE NS(be_monitor_turn_t) NS(BeamMonitor_start)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    return monitor->start;
}

SIXTRL_INLINE NS(be_monitor_turn_t) NS(BeamMonitor_skip)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    return monitor->skip;
}

SIXTRL_INLINE bool NS(BeamMonitor_is_rolling)( SIXTRL_BE_ARGPTR_DEC const
    NS(BeamMonitor) *const SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    return ( monitor->is_rolling == ( NS(be_monitor_flag_t) )1 );
}

SIXTRL_INLINE bool NS(BeamMonitor_is_turn_ordered)( SIXTRL_BE_ARGPTR_DEC const
    NS(BeamMonitor) *const SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    return ( monitor->is_turn_ordered == ( NS(be_monitor_flag_t) )1 );
}

SIXTRL_INLINE bool NS(BeamMonitor_is_particle_ordered)( SIXTRL_BE_ARGPTR_DEC
    const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT
{
    return !NS(BeamMonitor_is_turn_ordered)( monitor );
}

SIXTRL_INLINE NS(be_monitor_addr_t) NS(BeamMonitor_out_address)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    return monitor->out_address;
}

SIXTRL_INLINE NS(be_monitor_index_t) NS(BeamMonitor_min_particle_id)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    return monitor->min_particle_id;
}

SIXTRL_INLINE NS(be_monitor_index_t) NS(BeamMonitor_max_particle_id)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    return monitor->max_particle_id;
}

SIXTRL_INLINE NS(buffer_size_t) NS(BeamMonitor_stored_num_particles)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) size_t;
    typedef NS(be_monitor_index_t) index_t;

    size_t required_num_particles = ( size_t )0u;
    index_t const min_particle_id = NS(BeamMonitor_min_particle_id)( monitor );
    index_t const max_particle_id = NS(BeamMonitor_max_particle_id)( monitor );

    if( ( min_particle_id >= ( index_t )0u ) &&
        ( max_particle_id >= min_particle_id ) )
    {
        required_num_particles = ( size_t )(
            max_particle_id - min_particle_id + ( size_t )1u );
        required_num_particles *= NS(BeamMonitor_num_stores)( monitor );
    }

    return required_num_particles;
}

SIXTRL_INLINE bool NS(BeamMonitor_has_turn_stored)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor,
    NS(be_monitor_turn_t) const turn_id,
    NS(be_monitor_turn_t) const max_num_turns ) SIXTRL_NOEXCEPT
{
    bool is_stored = false;
    typedef NS(be_monitor_turn_t) nturn_t;

    nturn_t const num_stores = NS(BeamMonitor_num_stores)( monitor );
    nturn_t const start_turn = NS(BeamMonitor_start)( monitor );
    nturn_t const skip_turns = NS(BeamMonitor_skip)( monitor );

    if( ( monitor != SIXTRL_NULLPTR ) && ( start_turn >= ( nturn_t )0u ) &&
        ( start_turn <= turn_id ) && ( num_stores > ( nturn_t )0u ) &&
        ( skip_turns > ( nturn_t )0u ) &&
        ( ( max_num_turns >= turn_id ) || ( max_num_turns < ( nturn_t )0u ) ) )
    {
        if( turn_id >= start_turn )
        {
            nturn_t turns_since_start = turn_id - start_turn;

            if( ( turns_since_start % skip_turns ) == ( nturn_t )0u )
            {
                nturn_t store_idx = turns_since_start / skip_turns;

                if( !NS(BeamMonitor_is_rolling)( monitor ) )
                {
                    if( store_idx < num_stores ) is_stored = true;
                }
                else if( max_num_turns > start_turn )
                {
                    nturn_t const max_turn = max_num_turns - start_turn;
                    nturn_t max_num_stores = max_turn / skip_turns;

                    if( ( max_turn % skip_turns ) != ( nturn_t )0u )
                    {
                        ++max_num_stores;
                    }

                    if( max_num_stores <= ( store_idx + num_stores ) )
                    {
                        is_stored = true;
                    }
                }
            }
        }
    }

    return is_stored;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t) NS(BeamMonitor_set_num_stores)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_turn_t) const num_stores ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    monitor->num_stores = num_stores;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(BeamMonitor_set_start)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_turn_t) const start )SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    monitor->start = start;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(BeamMonitor_set_skip)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_turn_t) const skip ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    monitor->skip = skip;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(BeamMonitor_set_is_rolling)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    bool const is_rolling ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    monitor->is_rolling = ( is_rolling )
        ? ( NS(be_monitor_flag_t) )1 : ( NS(be_monitor_flag_t) )0;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(BeamMonitor_set_is_turn_ordered)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    bool const is_turn_ordered ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    monitor->is_turn_ordered = ( is_turn_ordered )
        ? ( NS(be_monitor_flag_t) )1 : ( NS(be_monitor_flag_t) )0;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(BeamMonitor_set_is_particle_ordered)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    bool const is_particle_ordered ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    monitor->is_turn_ordered = ( is_particle_ordered )
        ? ( NS(be_monitor_flag_t) )0 : ( NS(be_monitor_flag_t) )1;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(BeamMonitor_set_out_address)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_addr_t) const out_address ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    monitor->out_address = out_address;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(BeamMonitor_set_min_particle_id)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_index_t) const min_particle_id ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    monitor->min_particle_id = min_particle_id;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(BeamMonitor_set_max_particle_id)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_index_t) const max_particle_id ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    monitor->max_particle_id = max_particle_id;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t) NS(BeamMonitor_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT dst,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const
        SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( dst != SIXTRL_NULLPTR ) && ( src != SIXTRL_NULLPTR ) )
    {
        if( dst != src )
        {
            status  = NS(BeamMonitor_set_num_stores)(
                dst, NS(BeamMonitor_num_stores)( src ) );

            status |= NS(BeamMonitor_set_start)(
                dst, NS(BeamMonitor_start)( src ) );

            status |= NS(BeamMonitor_set_skip)(
                dst, NS(BeamMonitor_skip)( src ) );

            status |= NS(BeamMonitor_set_out_address)(
                dst, NS(BeamMonitor_out_address)( src ) );

            status |= NS(BeamMonitor_set_max_particle_id)(
                dst, NS(BeamMonitor_max_particle_id)( src ) );

            status |= NS(BeamMonitor_set_min_particle_id)(
                dst, NS(BeamMonitor_min_particle_id)( src ) );

            status |= NS(BeamMonitor_set_is_rolling)(
                dst, NS(BeamMonitor_is_rolling)( src ) );

            status |= NS(BeamMonitor_set_is_turn_ordered)(
                dst, NS(BeamMonitor_is_turn_ordered)( src ) );
        }
        else
        {
            status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
        }
    }

    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t)
NS(BeamMonitor_get_beam_monitor_indices_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_be_monitors,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( buffer != SIXTRL_NULLPTR ) && ( slot_size > ZERO ) &&
        ( !NS(ManagedBuffer_needs_remapping)( buffer, slot_size ) ) &&
        ( NS(ManagedBuffer_get_num_objects)( buffer, slot_size ) > ZERO ) )
    {
        status = NS(BeamMonitor_get_beam_monitor_indices_from_index_range)(
            NS(ManagedBuffer_get_const_objects_index_begin)(
                buffer, slot_size ),
            NS(ManagedBuffer_get_const_objects_index_end)(
                buffer, slot_size ), ZERO, max_num_of_indices, indices_begin,
                    ptr_num_be_monitors );
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t)
NS(BeamMonitor_get_beam_monitor_indices_from_index_range)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_end,
    NS(buffer_size_t) const start_index,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT ptr_num_be_monitors ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t)    buf_size_t;
    typedef NS(buffer_addr_t)    address_t;
    typedef NS(object_type_id_t) obj_type_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor) const* ptr_be_monitor_t;

    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR buf_size_t const MON_SIZE = sizeof( NS(BeamMonitor) );
    SIXTRL_STATIC_VAR address_t  const ZERO_ADDR = ( address_t )0u;
    buf_size_t num_be_monitors = ZERO;

    if( ( obj_it != SIXTRL_NULLPTR ) && ( obj_end != SIXTRL_NULLPTR ) &&
        ( ( ( uintptr_t )obj_end ) >= ( ( uintptr_t )obj_it ) ) &&
        ( max_num_of_indices > ZERO ) &&
        ( indices_begin != SIXTRL_NULLPTR ) )
    {
        buf_size_t next_index = start_index;

        status = SIXTRL_ARCH_STATUS_SUCCESS;

        while( ( num_be_monitors < max_num_of_indices ) &&
               ( obj_it != obj_end ) )
        {
            obj_type_t const type_id = NS(Object_get_type_id)( obj_it );
            address_t const addr = NS(Object_get_begin_addr)( obj_it );

            if( ( type_id == NS(OBJECT_TYPE_BEAM_MONITOR ) ) &&
                ( addr > ZERO_ADDR ) &&
                ( NS(Object_get_size)( obj_it ) >= MON_SIZE ) )
            {
                ptr_be_monitor_t mon = ( ptr_be_monitor_t )( uintptr_t )addr;

                if( mon != SIXTRL_NULLPTR )
                {
                    indices_begin[ num_be_monitors++ ] = next_index;
                }
            }
            else if( type_id == NS(OBJECT_TYPE_BEAM_MONITOR) )
            {
                status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
                break;
            }

            ++next_index;
            ++obj_it;
        }
    }

    if( ( status == SIXTRL_ARCH_STATUS_SUCCESS ) &&
        ( num_be_monitors == max_num_of_indices ) && ( obj_it != obj_end ) )
    {
        status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    }

    if(  ptr_num_be_monitors != SIXTRL_NULLPTR )
    {
        *ptr_num_be_monitors = num_be_monitors;
    }

    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor) const*
NS(BeamMonitor_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
    NS(Object) *const SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor) const* ptr_elem_t;
    ptr_elem_t elem = SIXTRL_NULLPTR;

    if( ( obj != SIXTRL_NULLPTR ) &&
        ( NS(Object_get_type_id)( obj ) == NS(OBJECT_TYPE_BEAM_MONITOR) ) &&
        ( NS(Object_get_size)( obj ) >= sizeof( NS(BeamMonitor) ) ) )
    {
        elem = ( ptr_elem_t )( uintptr_t )NS(Object_get_begin_addr)( obj );
    }

    return elem;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*
NS(BeamMonitor_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
    SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*
        )NS(BeamMonitor_const_from_obj_index)( obj );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor) const*
NS(BeamMonitor_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*
NS(BeamMonitor_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_from_obj_index)( NS(ManagedBuffer_get_object)(
        buffer_begin, index, slot_size ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE bool NS(BeamMonitor_are_present_in_obj_index_range)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_end
) SIXTRL_NOEXCEPT
{
    bool are_available = false;

    if( ( obj_it != SIXTRL_NULLPTR ) && ( obj_end != SIXTRL_NULLPTR ) )
    {
        SIXTRL_ASSERT( ( ( uintptr_t )( obj_it ) ) <= ( uintptr_t )obj_end );

        for( ; obj_it != obj_end ; ++obj_it )
        {
            if( ( NS(Object_get_type_id)( obj_it ) ==
                  NS(OBJECT_TYPE_BEAM_MONITOR) ) &&
                ( NS(BeamMonitor_const_from_obj_index)( obj_it ) !=
                  SIXTRL_NULLPTR ) )
            {
                are_available = true;
                break;
            }
        }
    }

    return are_available;
}

SIXTRL_INLINE bool NS(BeamMonitor_are_present_in_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_are_present_in_obj_index_range)(
        NS(ManagedBuffer_get_const_objects_index_begin)( buffer, slot_size ),
        NS(ManagedBuffer_get_const_objects_index_end)( buffer, slot_size ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(BeamMonitor_num_monitors_in_obj_index_range)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_end
) SIXTRL_NOEXCEPT
{
    NS(buffer_size_t) num_be_monitors = ( NS(buffer_size_t) )0u;

    if( ( obj_it != SIXTRL_NULLPTR ) && ( obj_end != SIXTRL_NULLPTR ) )
    {
        SIXTRL_ASSERT( ( ( uintptr_t )( obj_it ) ) <= ( uintptr_t )obj_end );

        for( ; obj_it != obj_end ; ++obj_it )
        {
            if( ( NS(Object_get_type_id)( obj_it ) ==
                  NS(OBJECT_TYPE_BEAM_MONITOR) ) &&
                ( NS(BeamMonitor_const_from_obj_index)( obj_it ) !=
                  SIXTRL_NULLPTR ) )
            {
                ++num_be_monitors;
            }
        }
    }

    return num_be_monitors;
}

SIXTRL_INLINE NS(buffer_size_t) NS(BeamMonitor_num_monitors_in_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_num_monitors_in_obj_index_range)(
        NS(ManagedBuffer_get_const_objects_index_begin)( buffer, slot_size ),
        NS(ManagedBuffer_get_const_objects_index_end)( buffer, slot_size ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t)
NS(BeamMonitor_monitor_indices_from_obj_index_range)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT obj_end,
    NS(buffer_size_t) const start_index ) SIXTRL_NOEXCEPT
{
    NS(buffer_size_t) num_be_monitors = ( NS(buffer_size_t) )0;
    if( ( obj_it != SIXTRL_NULLPTR ) && ( obj_end != SIXTRL_NULLPTR ) &&
        ( indices_begin != SIXTRL_NULLPTR ) &&
        ( max_num_of_indices > ( NS(buffer_size_t) )0 ) )
    {
        NS(buffer_size_t) be_index = start_index;
        SIXTRL_ASSERT( ( ( uintptr_t )obj_it ) <= ( uintptr_t )obj_end );

        for( ; obj_it != obj_end ; ++obj_it, ++be_index )
        {
            if( ( NS(Object_get_type_id)( obj_it ) ==
                  NS(OBJECT_TYPE_BEAM_MONITOR) ) &&
                ( NS(BeamMonitor_const_from_obj_index)( obj_it ) !=
                  SIXTRL_NULLPTR ) )
            {
                if( num_be_monitors < max_num_of_indices )
                {
                    indices_begin[ num_be_monitors ] = be_index;
                }

                ++num_be_monitors;
            }
        }
    }

    return num_be_monitors;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(BeamMonitor_monitor_indices_from_managed_buffer)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_monitor_indices_from_obj_index_range)( indices_begin,
        max_num_of_indices,
        NS(ManagedBuffer_get_const_objects_index_begin)( buffer, slot_size ),
        NS(ManagedBuffer_get_const_objects_index_end)( buffer, slot_size ),
        ( NS(buffer_size_t) )0 );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t) NS(BeamMonitor_reset)( SIXTRL_BE_ARGPTR_DEC
    NS(BeamMonitor)* SIXTRL_RESTRICT monitor ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( monitor != SIXTRL_NULLPTR )
    {
        status = NS(BeamMonitor_set_out_address)(
            monitor, ( NS(be_monitor_addr_t) )0 );

        status |= NS(BeamMonitor_set_min_particle_id)(
            monitor, ( NS(be_monitor_index_t) )-1 );

        status |= NS(BeamMonitor_set_max_particle_id)(
            monitor, ( NS(be_monitor_index_t) )-1 );
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(BeamMonitor_reset_all_in_obj_index_range)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT obj_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
        SIXTRL_RESTRICT obj_end ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    if( ( obj_it != SIXTRL_NULLPTR ) && ( obj_end != SIXTRL_NULLPTR ) )
    {
        SIXTRL_ASSERT( ( ( uintptr_t )obj_it ) <= ( uintptr_t )obj_end );
        for( ; obj_it != obj_end ; ++obj_it )
        {
            if( NS(Object_get_type_id)( obj_it ) ==
                NS(OBJECT_TYPE_BEAM_MONITOR) )
            {
                SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* be_monitor =
                    NS(BeamMonitor_from_obj_index)( obj_it );

                SIXTRL_ASSERT( be_monitor != SIXTRL_NULLPTR );
                status |= NS(BeamMonitor_reset)( be_monitor );
            }
        }
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(BeamMonitor_reset_all_in_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_reset_all_in_obj_index_range)(
        NS(ManagedBuffer_get_objects_index_begin)( buffer_begin, slot_size ),
        NS(ManagedBuffer_get_objects_index_end)( buffer_begin, slot_size ) );
}

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor) const*
NS(BeamMonitor_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_const_from_obj_index)(
        NS(Buffer_get_const_object)( buffer, index ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* NS(BeamMonitor_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_from_obj_index)(
        NS(Buffer_get_object)( buffer, index ) );
}

SIXTRL_INLINE bool NS(BeamMonitor_are_present_in_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT buffer ) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_are_present_in_obj_index_range)(
        NS(Buffer_get_const_objects_begin)( buffer ),
        NS(Buffer_get_const_objects_end)( buffer ) );
}

SIXTRL_INLINE NS(buffer_size_t) NS(BeamMonitor_num_monitors_in_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT buffer ) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_num_monitors_in_obj_index_range)(
        NS(Buffer_get_const_objects_begin)( buffer ),
        NS(Buffer_get_const_objects_end)( buffer ) );
}

SIXTRL_INLINE NS(buffer_size_t) NS(BeamMonitor_monitor_indices_from_buffer)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer
) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_monitor_indices_from_obj_index_range)(
        indices_begin, max_num_of_indices,
        NS(Buffer_get_const_objects_begin)( buffer ),
        NS(Buffer_get_const_objects_end)( buffer ),
        ( NS(buffer_size_t) )0 );
}

SIXTRL_INLINE NS(arch_status_t) NS(BeamMonitor_reset_all_in_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer ) SIXTRL_NOEXCEPT
{
    return NS(BeamMonitor_reset_all_in_obj_index_range)(
        NS(Buffer_get_objects_begin)( buffer ),
        NS(Buffer_get_objects_end)( buffer ) );
}

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */
#endif /* SIXTRACKL_COMMON_BE_MONITOR_BE_MONITOR_C99_HEADER_H__ */
