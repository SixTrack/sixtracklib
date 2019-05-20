#include "sixtracklib/common/control/node_controller_base.h"

#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "sixtracklib/common/control/definitions.h"

namespace st = SIXTRL_CXX_NAMESPACE;

/* ========================================================================= */

::NS(node_index_t) NS(Controller_get_num_available_nodes)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->numAvailableNodes() : ::NS(node_index_t){ 0 };
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Controller_has_default_node)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->hasDefaultNode() ) );
}

::NS(node_index_t) NS(Controller_get_default_node_index)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->defaultNodeIndex()
        : st::NODE_UNDEFINED_INDEX;
}

::NS(NodeId) const* NS(Controller_get_default_node_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrDefaultNodeId() : nullptr;
}

::NS(NodeInfoBase) const* NS(Controller_get_default_node_info_base)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrDefaultNodeInfoBase() : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Controller_is_default_node_index)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    ::NS(arch_size_t)const node_index )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->isDefaultNode( node_index ) ) );
}

bool NS(Controller_is_default_node_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    const NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) && ( node_id != nullptr ) &&
             ( ptr_nodes_ctx->isDefaultNode( *node_id ) ) );
}

bool NS(Controller_is_default_platform_id_and_device_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    ::NS(node_platform_id_t) const platform_id,
    ::NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->isDefaultNode( platform_id, device_id ) ) );
}

bool NS(Controller_is_default_node)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->isDefaultNode( node_id_str ) ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Controller_is_node_available_by_index)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    ::NS(arch_size_t)const node_index )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->isNodeAvailable( node_index ) ) );
}

bool NS(Controller_is_node_available_by_node_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    const ::NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) && ( node_id != nullptr ) &&
             ( ptr_nodes_ctx->isNodeAvailable( *node_id ) ) );
}

bool NS(Controller_is_node_available_by_platform_id_and_device_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    ::NS(node_platform_id_t) const platform_id,
    ::NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->isNodeAvailable( platform_id, device_id ) ) );
}

bool NS(Controller_is_node_available)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->isNodeAvailable( node_id_str ) ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(node_index_t) NS(Controller_get_node_index_by_node_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    const ::NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    
    return ( ( ptr_nodes_ctx != nullptr ) && ( node_id != nullptr ) )
        ? ptr_nodes_ctx->nodeIndex( *node_id ) : st::NODE_UNDEFINED_INDEX;
}

::NS(node_index_t) NS(Controller_get_node_index_by_platform_id_and_device_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    ::NS(node_platform_id_t) const platform_id, 
    ::NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    
    return ( ptr_nodes_ctx != nullptr ) 
        ? ptr_nodes_ctx->nodeIndex( platform_id, device_id ) 
        : st::NODE_UNDEFINED_INDEX;
}

::NS(node_index_t) NS(Controller_get_node_index_by_node_info)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT ptr_node_info )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    
    return ( ptr_nodes_ctx != nullptr ) 
        ? ptr_nodes_ctx->nodeIndex( ptr_node_info ) 
        : st::NODE_UNDEFINED_INDEX;
}

::NS(node_index_t) NS(Controller_get_node_index)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    
    return ( ptr_nodes_ctx != nullptr ) 
        ? ptr_nodes_ctx->nodeIndex( node_id_str ) 
        : st::NODE_UNDEFINED_INDEX;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(NodeId) const* NS(Controller_get_ptr_node_id_by_index)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    ::NS(arch_size_t)const node_index )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrNodeId( node_index ) : nullptr;
}

::NS(NodeId) const* NS(Controller_get_ptr_node_id_by_platform_id_and_device_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    ::NS(node_platform_id_t) const platform_id,
    ::NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrNodeId( platform_id, device_id ) : nullptr;
}

::NS(NodeId) const* NS(Controller_get_ptr_node_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrNodeId( node_id_str ) : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(NodeInfo) const* NS(Controller_get_ptr_node_info_base_begin)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    return ( ctrl != nullptr ) ? ctrl->nodeInfoBaseBegin() : nullptr;
}

::NS(NodeInfo) const* NS(Controller_get_ptr_node_info_base_end)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    return ( ctrl != nullptr ) ? ctrl->nodeInfoBaseEnd() : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(NodeInfoBase) const* NS(Controller_get_ptr_node_info_base_by_index)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    ::NS(arch_size_t)const node_index )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrNodeInfoBase( node_index ) : nullptr;
}

::NS(NodeInfoBase) const* NS(Controller_get_ptr_node_info_base_by_node_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    const ::NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) && ( node_id != nullptr ) )
        ? ptr_nodes_ctx->ptrNodeInfoBase( *node_id ) : nullptr;
}

::NS(NodeInfoBase) const*
NS(Controller_get_ptr_node_info_base_by_platform_id_and_device_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrNodeInfoBase( platform_id, device_id ) : nullptr;
}

::NS(NodeInfoBase) const* NS(Controller_get_ptr_node_info_base)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrNodeInfoBase( node_id_str ) : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Controller_has_selected_node)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->hasSelectedNode() ) );
}

::NS(arch_size_t) NS(Controller_get_selected_node_index)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->selectedNodeIndex()
        : st::NODE_UNDEFINED_INDEX;
}

::NS(NodeId) const* NS(Controller_get_ptr_selected_node_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrSelectedNodeId() : nullptr;
}

::NS(NodeInfoBase) const* NS(Controller_get_ptr_selected_node_info_base)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrSelectedNodeInfoBase() : nullptr;
}

char const* NS(Controller_get_selected_node_id_str)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrSelectedNodeIdStr() : nullptr;
}

bool NS(Controller_copy_selected_node_id_str)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    char* SIXTRL_RESTRICT node_id_str,
    ::NS(arch_size_t) const max_str_length )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->selectedNodeIdStr( node_id_str, max_str_length )
        : false;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Controller_select_node)( ::NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->selectNode( node_id_str ) ) );
}

bool NS(Controller_select_node_by_node_id)(
    NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    const NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) && ( node_id != nullptr ) &&
             ( ptr_nodes_ctx->selectNode( *node_id ) ) );
}

bool NS(Controller_select_node_by_plaform_id_and_device_id)(
    NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->selectNode( platform_id, device_id ) ) );
}

bool NS(Controller_select_node_by_index)( NS(ControllerBase)* SIXTRL_RESTRICT ctx,
    ::NS(arch_size_t)const node_index )
{
    auto ptr_nodes_ctx = st::asNodeController( ctx );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->selectNode( node_index ) ) );
}

/* ========================================================================= */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Controller_can_change_selected_node)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->canChangeSelectedNode() ) );
}

bool NS(Controller_can_directly_change_selected_node)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->canDirectlyChangeSelectedNode() ) );
}

bool NS(Controller_change_selected_node)(
    ::NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    ::NS(node_index_t) const current_selected_node_index,
    ::NS(node_index_t) const new_selected_node_index )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) &&
    ( ptr_nodes_ctx->changeSelectedNode(
        current_selected_node_index, new_selected_node_index ) ) );
}

/* ========================================================================= */

bool NS(Controller_can_unselect_node)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->canUnselectNode() ) );
}

bool NS(Controller_unselect_node)(
    NS(ControllerBase)* SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->unselectNode() ) );
}

bool NS(Controller_unselect_node_by_index)(
    NS(ControllerBase)* SIXTRL_RESTRICT ctrl, NS(node_index_t) const index )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->unselectNode( index ) ) );
}

bool NS(Controller_unselect_node_by_node_id)(
    NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    const NS(NodeId) *const SIXTRL_RESTRICT ptr_node_id )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) && ( ptr_node_id != nullptr ) &&
             ( ptr_nodes_ctx->unselectNode( *ptr_node_id ) ) );
}

bool NS(Controller_unselect_node_by_platform_id_and_device_id)(
    NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->unselectNode( platform_id, device_id ) ) );
}

bool NS(Controller_unselect_node_by_node_id_str)(
    NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->unselectNode( node_id_str ) ) );
}

/* ========================================================================= */

void NS(Controller_print_available_nodes_info)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    FILE* SIXTRL_RESTRICT fp )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    if( ptr_nodes_ctx != nullptr )
    {
        ptr_nodes_ctx->printAvailableNodesInfo( fp );
    }
}

void NS(Controller_print_out_available_nodes_info)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    if( ptr_nodes_ctx != nullptr )
    {
        ptr_nodes_ctx->printAvailableNodesInfo();
    }
}

void NS(Controller_store_available_nodes_info_to_string)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    char* SIXTRL_RESTRICT nodes_info_str,
    ::NS(arch_size_t)const nodes_info_str_capacity,
    ::NS(arch_size_t)* SIXTRL_RESTRICT ptr_required_max_str_length )
{
    auto ptr_nodes_ctx = st::asNodeController( ctrl );
    if( ptr_nodes_ctx != nullptr )
    {
        ptr_nodes_ctx->storeAvailableNodesInfoToCString(
            nodes_info_str, nodes_info_str_capacity,
                ptr_required_max_str_length );
    }
}

/* end: sixtracklib/common/control/node_controller_base.cpp */
