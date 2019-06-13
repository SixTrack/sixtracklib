#include "sixtracklib/common/control/kernel_config_base.h"

#include <cstddef>
#include <cstdlib>
#include <cstdio>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

#include "sixtracklib/common/control/kernel_config_base.hpp"

void NS(KernelConfig_delete)( ::NS(KernelConfigBase)* SIXTRL_RESTRICT config )
{
    if( config != nullptr ) delete config;
    return;
}

/* ------------------------------------------------------------------------- */

::NS(arch_id_t) NS(KernelConfig_get_arch_id)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT config )
{
    return ( config != nullptr )
        ? config->archId() : ::NS(ARCHITECTURE_ILLEGAL);
}

bool NS(KernelConfig_has_arch_string)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT config )
{
    return ( ( config != nullptr ) && ( config->hasArchStr() ) );
}

char const* NS(KernelConfig_get_ptr_arch_string)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT config )
{
    return ( config != nullptr ) ? config->ptrArchStr() : nullptr;
}

/* ------------------------------------------------------------------------- */

bool NS(KernelConfig_has_kernel_id)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf )
{
    return ( ( conf != nullptr ) && ( conf->hasKernelId() ) );
}

NS(ctrl_kernel_id_t) NS(KernelConfig_get_kernel_id)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr )
        ? conf->kernelId() : ::NS(ARCH_ILLEGAL_KERNEL_ID);
}

void NS(KernelConfig_set_kernel_id)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf,
    NS(ctrl_kernel_id_t) const kernel_id )
{
    if( conf != nullptr ) conf->setKernelId( kernel_id );
}

/* ------------------------------------------------------------------------- */

bool NS(KernelConfig_has_name)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf )
{
    return ( ( conf != nullptr ) && ( conf->hasName() ) );
}

char const* NS(KernelConfig_get_ptr_name_string)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr ) ? conf->ptrNameStr() : nullptr;
}

void NS(KernelConfig_set_name)( ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf,
    char const* SIXTRL_RESTRICT kernel_name )
{
    if( conf != nullptr ) conf->setName( kernel_name );
}

/* ------------------------------------------------------------------------- */

::NS(ctrl_size_t) NS(KernelConfig_get_num_arguments)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr ) ? conf->numArguments() : ::NS(ctrl_size_t){ 0 };
}

void NS(KernelConfig_set_num_arguments)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const num_kernel_args )
{
    if( conf != nullptr ) conf->setNumArguments( num_kernel_args );
}

/* ------------------------------------------------------------------------- */

::NS(ctrl_size_t) NS(KernelConfig_get_work_items_dim)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr ) ? conf->workItemsDim() : ::NS(ctrl_size_t){ 0 };
}

::NS(ctrl_size_t) NS(KernelConfig_get_num_work_items_by_dim)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const index )
{
    return ( conf != nullptr )
        ? conf->numWorkItems( index ) : ::NS(ctrl_size_t){ 0 };
}

::NS(ctrl_size_t) NS(KernelConfig_get_total_num_work_items)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr )
        ? conf->totalNumWorkItems() : ::NS(ctrl_size_t){ 0 };
}

::NS(ctrl_size_t) const* NS(KernelConfig_get_const_work_items_begin)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr ) ? conf->workItemsBegin() : nullptr;
}

::NS(ctrl_size_t) const* NS(KernelConfig_get_const_work_items_end)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr ) ? conf->workItemsEnd() : nullptr;
}

::NS(ctrl_size_t)* NS(KernelConfig_get_work_items_begin)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr ) ? conf->workItemsBegin() : nullptr;
}

::NS(ctrl_size_t)* NS(KernelConfig_get_work_items_end)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr ) ? conf->workItemsEnd() : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(ctrl_status_t) NS(KernelConfig_set_num_work_items_1d)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const work_items_a )
{
    return ( conf != nullptr )
        ? conf->setNumWorkItems( work_items_a )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

::NS(ctrl_status_t) NS(KernelConfig_set_num_work_items_2d)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const work_items_a,
    ::NS(ctrl_size_t) const work_items_b )
{
    return ( conf != nullptr )
        ? conf->setNumWorkItems( work_items_a, work_items_b )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

::NS(ctrl_status_t) NS(KernelConfig_set_num_work_items_3d)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const work_items_a, ::NS(ctrl_size_t) const work_items_b,
    ::NS(ctrl_size_t) const work_items_c )
{
    return ( conf != nullptr )
        ? conf->setNumWorkItems( work_items_a, work_items_b, work_items_c )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

::NS(ctrl_status_t) NS(KernelConfig_set_num_work_items)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const work_items_dim,
    ::NS(ctrl_size_t) const* SIXTRL_RESTRICT work_itms_begin )
{
    return ( conf != nullptr )
        ? conf->setNumWorkItems( work_items_dim, work_itms_begin )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

/* ------------------------------------------------------------------------- */

::NS(ctrl_size_t) NS(KernelConfig_get_work_item_offset_by_dim)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const index )
{
    return ( conf != nullptr )
        ? conf->workItemOffset( index ) : ::NS(ctrl_size_t){ 0 };
}

::NS(ctrl_size_t) const* NS(KernelConfig_get_const_work_item_offsets_begin)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr ) ? conf->workItemOffsetsBegin() : nullptr;
}

::NS(ctrl_size_t) const* NS(KernelConfig_get_const_work_item_offsets_end)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr ) ? conf->workItemOffsetsEnd() : nullptr;
}

::NS(ctrl_size_t)* NS(KernelConfig_work_item_offsets_begin)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr ) ? conf->workItemOffsetsBegin() : nullptr;
}

::NS(ctrl_size_t)* NS(KernelConfig_work_item_offsets_end)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr ) ? conf->workItemOffsetsEnd() : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(ctrl_status_t) NS(KernelConfig_set_work_item_offset_1d)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const offset_a )
{
    return ( conf != nullptr )
        ? conf->setWorkItemOffset( offset_a )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

::NS(ctrl_status_t) NS(KernelConfig_set_work_item_offset_2d)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const offset_a,
    ::NS(ctrl_size_t) const offset_b )
{
    return ( conf != nullptr )
        ? conf->setWorkItemOffset( offset_a, offset_b )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

::NS(ctrl_status_t) NS(KernelConfig_set_work_item_offset_3d)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const offset_a, ::NS(ctrl_size_t) const offset_b,
    ::NS(ctrl_size_t) const offset_c )
{
    return ( conf != nullptr )
        ? conf->setWorkItemOffset( offset_a, offset_b, offset_c )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

::NS(ctrl_status_t) NS(KernelConfig_set_work_item_offset)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const offset_dim,
    ::NS(ctrl_size_t) const* SIXTRL_RESTRICT offsets_begin )
{
    return ( conf != nullptr )
        ? conf->setWorkItemOffset( offset_dim, offsets_begin )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

/* ------------------------------------------------------------------------- */

::NS(ctrl_size_t) NS(KernelConfig_get_work_group_size_by_dim)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const index )
{
    return ( conf != nullptr )
        ? conf->workGroupSize( index ) : ::NS(ctrl_size_t){ 0 };
}

::NS(ctrl_size_t) NS(KernelConfig_get_work_groups_dim)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr )
        ? conf->workGroupsDim() : ::NS(ctrl_size_t){ 0 };
}

::NS(ctrl_size_t) const* NS(KernelConfig_get_const_work_group_sizes_begin)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr ) ? conf->workGroupSizesBegin() : nullptr;
}

::NS(ctrl_size_t) const* NS(KernelConfig_get_const_work_group_sizes_end)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr ) ? conf->workGroupSizesEnd() : nullptr;
}

::NS(ctrl_size_t)* NS(KernelConfig_get_work_group_sizes_begin)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr ) ? conf->workGroupSizesBegin() : nullptr;
}

::NS(ctrl_size_t)* NS(KernelConfig_work_group_sizes_end)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr ) ? conf->workGroupSizesEnd() : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(ctrl_status_t) NS(KernelConfig_set_work_group_sizes_1d)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const work_groups_a )
{
    return ( conf != nullptr )
        ? conf->setWorkGroupSizes( work_groups_a )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

::NS(ctrl_status_t) NS(KernelConfig_set_work_group_sizes_2d)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const work_groups_a,
    ::NS(ctrl_size_t) const work_groups_b )
{
    return ( conf != nullptr )
        ? conf->setWorkGroupSizes( work_groups_a, work_groups_b )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

::NS(ctrl_status_t) NS(KernelConfig_set_work_group_sizes_3d)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const work_groups_a,
    ::NS(ctrl_size_t) const work_groups_b,
    ::NS(ctrl_size_t) const work_groups_c )
{
    return ( conf != nullptr )
        ? conf->setWorkGroupSizes( work_groups_a, work_groups_b, work_groups_c )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

::NS(ctrl_status_t) NS(KernelConfig_set_work_group_sizes)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const work_groups_dim,
    ::NS(ctrl_size_t) const* SIXTRL_RESTRICT work_grps_begin )
{
    return ( conf != nullptr )
        ? conf->setWorkGroupSizes( work_groups_dim, work_grps_begin )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

/* ------------------------------------------------------------------------- */

::NS(ctrl_size_t) NS(KernelConfig_get_preferred_work_group_multiple_by_dim)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const index )
{
    return ( conf != nullptr )
        ? conf->preferredWorkGroupMultiple( index ) : ::NS(ctrl_size_t){ 0 };
}

::NS(ctrl_size_t) const*
NS(KernelConfig_get_const_preferred_work_group_multiples_begin)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr )
        ? conf->preferredWorkGroupMultiplesBegin() : nullptr;
}

::NS(ctrl_size_t) const*
NS(KernelConfig_get_const_preferred_work_group_multiples_end)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr )
        ? conf->preferredWorkGroupMultiplesEnd() : nullptr;
}

::NS(ctrl_size_t)* NS(KernelConfig_get_preferred_work_group_multiples_begin)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr )
        ? conf->preferredWorkGroupMultiplesBegin() : nullptr;
}

::NS(ctrl_size_t)* NS(KernelConfig_get_preferred_work_group_multiples_end)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr )
        ? conf->preferredWorkGroupMultiplesEnd() : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(ctrl_status_t) NS(KernelConfig_set_preferred_work_group_multiple_1d)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const work_groups_a )
{
    return ( conf != nullptr )
        ? conf->setPreferredWorkGroupMultiple( work_groups_a )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

::NS(ctrl_status_t) NS(KernelConfig_set_preferred_work_group_multiple_2d)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const work_groups_a,
    ::NS(ctrl_size_t) const work_groups_b )
{
    return ( conf != nullptr )
        ? conf->setPreferredWorkGroupMultiple( work_groups_a, work_groups_b )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

::NS(ctrl_status_t) NS(KernelConfig_set_preferred_work_group_multiple_3d)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const work_groups_a,
    ::NS(ctrl_size_t) const work_groups_b,
    ::NS(ctrl_size_t) const work_groups_c )
{
    return ( conf != nullptr )
        ? conf->setPreferredWorkGroupMultiple(
                work_groups_a, work_groups_b, work_groups_c )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

::NS(ctrl_status_t) NS(KernelConfig_set_preferred_work_group_multiple)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf,
    ::NS(ctrl_size_t) const work_groups_dim,
    ::NS(ctrl_size_t) const* SIXTRL_RESTRICT pref_work_groups_multiple )
{
    return ( conf != nullptr )
        ? conf->setPreferredWorkGroupMultiple(
                work_groups_dim, pref_work_groups_multiple )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

/* ------------------------------------------------------------------------- */

void NS(KernelConfig_clear)( ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf )
{
    if( conf != nullptr ) conf->clear();
}

void NS(KernelConfig_reset)( ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf,
    NS(ctrl_size_t) work_items_dim, NS(ctrl_size_t) work_groups_dim )
{
    if( conf != nullptr ) conf->reset( work_items_dim, work_groups_dim );
}

bool NS(KernelConfig_needs_update)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf )
{
    return ( ( conf != nullptr ) && ( conf->needsUpdate() ) );
}

::NS(ctrl_status_t) NS(KernelConfig_update)(
    ::NS(KernelConfigBase)* SIXTRL_RESTRICT conf )
{
    return ( conf != nullptr )
        ? conf->update() : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

/* ------------------------------------------------------------------------- */

void NS(KernelConfig_print)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf,
    ::FILE* SIXTRL_RESTRICT output )
{
    if( conf != nullptr ) conf->print( output );
}

void NS(KernelConfig_print_out)(
    const ::NS(KernelConfigBase) *const SIXTRL_RESTRICT conf )
{
    if( conf != nullptr ) conf->printOut();
}

#endif /* C++, Host */

/* end: sixtracklib/common/control/kernel_config_base_c99.cpp */
