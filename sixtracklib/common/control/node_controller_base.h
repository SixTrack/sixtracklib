#ifndef SIXTRACKLIB_COMMON_CONTROL_NODE_CONTROLLER_BASE_H__
#define SIXTRACKLIB_COMMON_CONTROL_NODE_CONTROLLER_BASE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/node_id.h"
    #include "sixtracklib/common/control/node_info.h"
    #include "sixtracklib/common/control/controller_base.h"
    #include "sixtracklib/common/control/node_controller_base.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, Host */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(node_index_t)
NS(Controller_get_num_available_nodes)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(Controller_has_default_node)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(node_index_t)
NS(Controller_get_default_node_index)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(NodeId) const*
NS(Controller_get_default_node_id)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(NodeInfoBase) const*
NS(Controller_get_default_node_info_base)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Controller_is_default_node_by_index)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    NS(node_index_t) const node_index );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Controller_is_default_node_by_node_id)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    const NS(NodeId) *const SIXTRL_RESTRICT node_id );

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(Controller_is_default_node_by_platform_device_ids)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Controller_is_default_node)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT node_id_str );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Controller_is_node_available_by_index)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    NS(node_index_t) const node_index );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Controller_is_node_available_by_node_id)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    const NS(NodeId) *const SIXTRL_RESTRICT node_id );

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(Controller_is_node_available_by_platform_device_ids)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Controller_is_node_available)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT node_id_str );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(NodeId) const*
NS(Controller_get_ptr_node_id_by_index)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    NS(node_index_t) const node_index );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(NodeId) const*
NS(Controller_get_ptr_node_id_by_platform_device_ids)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(NodeId) const* NS(Controller_get_ptr_node_id)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT node_id_str );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(NodeInfoBase) const*
NS(Controller_get_ptr_node_info_base_by_index)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    NS(ctrl_size_t) const node_index );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(NodeInfoBase) const*
NS(Controller_get_ptr_node_info_base_by_node_id)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    const NS(NodeId) *const SIXTRL_RESTRICT node_id );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(NodeInfoBase) const*
NS(Controller_get_ptr_node_info_base_by_platform_device_ids)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(NodeInfoBase) const*
NS(Controller_get_ptr_node_info_base)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT node_id_str );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Controller_has_selected_node)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_size_t)
NS(Controller_get_selected_node_index)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(NodeId) const*
NS(Controller_get_ptr_selected_node_id)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(NodeInfoBase) const*
NS(Controller_get_ptr_selected_node_info_base)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl );

SIXTRL_EXTERN SIXTRL_HOST_FN char const*
NS(Controller_get_selected_node_id_str)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Controller_copy_selected_node_id_str)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    char* SIXTRL_RESTRICT node_id_str,
    NS(ctrl_size_t) const max_str_length );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Controller_select_node)(
    NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT node_id_str );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Controller_select_node_by_node_id)(
    NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    const NS(NodeId) *const SIXTRL_RESTRICT node_id );

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(Controller_select_node_by_platform_device_ids)(
    NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Controller_select_node_by_index)(
    NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    NS(node_index_t) const node_index );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Controller_print_available_nodes_info)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    FILE* SIXTRL_RESTRICT fp );

SIXTRL_EXTERN SIXTRL_HOST_FN void
NS(Controller_print_out_available_nodes_info)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl );

SIXTRL_EXTERN SIXTRL_HOST_FN void
NS(Controller_store_available_nodes_info_to_string)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    char* SIXTRL_RESTRICT nodes_info,
    NS(ctrl_size_t) const max_str_length,
    NS(ctrl_size_t)* SIXTRL_RESTRICT ptr_required_max_str_length );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_CONTROL_NODE_CONTROLLER_BASE_H__ */

/* end: sixtracklib/common/control/node_controller_base.h */
