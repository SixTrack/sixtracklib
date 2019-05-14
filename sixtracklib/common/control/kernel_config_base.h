#ifndef SIXTRACKLIB_COMMON_CONTROL_KERNEL_CONFIG_BASE_C99_H__
#define SIXTRACKLIB_COMMON_CONTROL_KERNEL_CONFIG_BASE_C99_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdlib.h>
    #include <stdio.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/kernel_config_base.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++ */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_id_t) NS(KernelConfig_get_arch_id)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(KernelConfig_has_arch_string)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN char const* NS(KernelConfig_get_arch_string)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(KernelConfig_has_kernel_id)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_kernel_id_t)
NS(KernelConfig_get_kernel_id)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(KernelConfig_set_kernel_id)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config,
    NS(ctrl_kernel_id_t) const kernel_id );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(KernelConfig_has_name)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN char const* NS(KernelConfig_get_ptr_name_string)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(KernelConfig_set_name)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config,
    char const* SIXTRL_RESTRICT kernel_name );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_size_t)
NS(KernelConfig_get_num_arguments)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(KernelConfig_set_num_arguments)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const num_kernel_args );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_size_t)
NS(KernelConfig_get_work_items_dim)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN  NS(ctrl_size_t)
NS(KernelConfig_get_num_work_items_by_dim)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const index );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_size_t)
NS(KernelConfig_get_total_num_work_items)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN  NS(ctrl_size_t) const*
NS(KernelConfig_get_const_work_items_begin)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN  NS(ctrl_size_t) const*
NS(KernelConfig_get_const_work_items_end)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_size_t)*
NS(KernelConfig_get_work_items_begin)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_size_t)*
NS(KernelConfig_get_work_items_end)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(KernelConfig_set_num_work_items_1d)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const work_items_a );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(KernelConfig_set_num_work_items_2d)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const work_items_a,
    NS(ctrl_size_t) const work_items_b );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(KernelConfig_set_num_work_items_3d)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const work_items_a,
    NS(ctrl_size_t) const work_items_b,
    NS(ctrl_size_t) const work_item_c );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(KernelConfig_set_num_work_items)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const work_items_dim,
    NS(ctrl_size_t) const* SIXTRL_RESTRICT work_itms_begin );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN  NS(ctrl_size_t)
NS(KernelConfig_get_work_item_offset_by_dim)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const index );

SIXTRL_EXTERN SIXTRL_HOST_FN  NS(ctrl_size_t) const*
NS(KernelConfig_get_const_work_item_offsets_begin)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN  NS(ctrl_size_t) const*
NS(KernelConfig_get_const_work_item_offsets_end)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_size_t)*
NS(KernelConfig_work_item_offsets_begin)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_size_t)*
NS(KernelConfig_work_item_offsets_end)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(KernelConfig_set_work_item_offset_1d)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const offset_a );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(KernelConfig_set_work_item_offset_2d)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const offset_a,
    NS(ctrl_size_t) const offset_b );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(KernelConfig_set_work_item_offset_3d)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const offset_a,
    NS(ctrl_size_t) const offset_b,
    NS(ctrl_size_t) const offset_c );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(KernelConfig_set_work_item_offset)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const offset_dim,
    NS(ctrl_size_t) const* SIXTRL_RESTRICT offsets_begin );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN  NS(ctrl_size_t)
NS(KernelConfig_get_work_group_size_by_dim)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const index );

SIXTRL_EXTERN SIXTRL_HOST_FN  NS(ctrl_size_t)
NS(KernelConfig_get_work_groups_dim)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN  NS(ctrl_size_t) const*
NS(KernelConfig_get_const_work_group_sizes_begin)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN  NS(ctrl_size_t) const*
NS(KernelConfig_get_const_work_group_sizes_end)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_size_t)*
NS(KernelConfig_get_work_group_sizes_begin)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_size_t)*
NS(KernelConfig_work_group_sizes_end)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(KernelConfig_set_work_group_sizes_1d)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const work_groups_a );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(KernelConfig_set_work_group_sizes_2d)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const work_groups_a,
    NS(ctrl_size_t) const work_groups_b );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(KernelConfig_set_work_group_sizes_3d)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const work_groups_a,
    NS(ctrl_size_t) const work_groups_b,
    NS(ctrl_size_t) const work_groups_c );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(KernelConfig_set_work_group_sizes)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const work_groups_dim,
    NS(ctrl_size_t) const* SIXTRL_RESTRICT work_grps_begin );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN  NS(ctrl_size_t)
NS(KernelConfig_get_preferred_work_group_multiple_by_dim)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const index );

SIXTRL_EXTERN SIXTRL_HOST_FN  NS(ctrl_size_t) const*
NS(KernelConfig_get_const_preferred_work_group_multiples_begin)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN  NS(ctrl_size_t) const*
NS(KernelConfig_get_const_preferred_work_group_multiples_end)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_size_t)*
NS(KernelConfig_get_preferred_work_group_multiples_begin)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_size_t)*
NS(KernelConfig_get_preferred_work_group_multiples_end)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(KernelConfig_set_preferred_work_group_multiple_1d)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const work_groups_a );

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(KernelConfig_set_preferred_work_group_multiple_2d)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const work_groups_a,
    NS(ctrl_size_t) const work_groups_b );

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(KernelConfig_set_preferred_work_group_multiple_3d)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const work_groups_a,
    NS(ctrl_size_t) const work_groups_b,
    NS(ctrl_size_t) const work_groups_c );

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(KernelConfig_set_preferred_work_group_multiple)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config,
    NS(ctrl_size_t) const work_groups_dim,
    NS(ctrl_size_t) const* SIXTRL_RESTRICT pref_work_groups_multiple );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(KernelConfig_clear)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(KernelConfig_reset)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config,
    NS(ctrl_size_t) work_items_dim,
    NS(ctrl_size_t) work_groups_dim );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(KernelConfig_needs_update)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(KernelConfig_update)(
    NS(KernelConfigBase)* SIXTRL_RESTRICT config );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(KernelConfig_print)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config,
    FILE* SIXTRL_RESTRICT output );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(KernelConfig_print_out)(
    const NS(KernelConfigBase) *const SIXTRL_RESTRICT config );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_CONTROL_KERNEL_CONFIG_BASE_C99_H__ */
/* end: sixtracklib/common/control/kernel_config_base.h */
