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
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ptr_nodes_ctrl != nullptr )
        ? ptr_nodes_ctrl->numAvailableNodes() : ::NS(node_index_t){ 0 };
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Controller_has_default_node)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) &&
             ( ptr_nodes_ctrl->hasDefaultNode() ) );
}

::NS(node_index_t) NS(Controller_get_default_node_index)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ptr_nodes_ctrl != nullptr )
        ? ptr_nodes_ctrl->defaultNodeIndex()
        : st::NODE_UNDEFINED_INDEX;
}

::NS(NodeId) const* NS(Controller_get_default_node_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ptr_nodes_ctrl != nullptr )
        ? ptr_nodes_ctrl->ptrDefaultNodeId() : nullptr;
}

::NS(NodeInfoBase) const* NS(Controller_get_default_node_info_base)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ptr_nodes_ctrl != nullptr )
        ? ptr_nodes_ctrl->ptrDefaultNodeInfoBase() : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Controller_is_default_node_index)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    ::NS(arch_size_t)const node_index )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) &&
             ( ptr_nodes_ctrl->isDefaultNode( node_index ) ) );
}

bool NS(Controller_is_default_node_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    const NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) && ( node_id != nullptr ) &&
             ( ptr_nodes_ctrl->isDefaultNode( *node_id ) ) );
}

bool NS(Controller_is_default_platform_id_and_device_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    ::NS(node_platform_id_t) const platform_id,
    ::NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) &&
             ( ptr_nodes_ctrl->isDefaultNode( platform_id, device_id ) ) );
}

bool NS(Controller_is_default_node)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) &&
             ( ptr_nodes_ctrl->isDefaultNode( node_id_str ) ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Controller_is_node_available_by_index)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    ::NS(arch_size_t)const node_index )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) &&
             ( ptr_nodes_ctrl->isNodeAvailable( node_index ) ) );
}

bool NS(Controller_is_node_available_by_node_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    const ::NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) && ( node_id != nullptr ) &&
             ( ptr_nodes_ctrl->isNodeAvailable( *node_id ) ) );
}

bool NS(Controller_is_node_available_by_platform_id_and_device_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    ::NS(node_platform_id_t) const platform_id,
    ::NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) &&
             ( ptr_nodes_ctrl->isNodeAvailable( platform_id, device_id ) ) );
}

bool NS(Controller_is_node_available)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) &&
             ( ptr_nodes_ctrl->isNodeAvailable( node_id_str ) ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(node_index_t) NS(Controller_get_min_available_node_index)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ptr_nodes_ctrl != nullptr ) 
        ? ptr_nodes_ctrl->minAvailableNodeIndex() 
        : st::NODE_UNDEFINED_INDEX;
}

::NS(node_index_t) NS(Controller_get_max_available_node_index)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ptr_nodes_ctrl != nullptr ) 
        ? ptr_nodes_ctrl->maxAvailableNodeIndex() 
        : st::NODE_UNDEFINED_INDEX;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(ctrl_size_t) NS(Controller_get_available_node_indices)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    ::NS(ctrl_size_t) const max_num_node_indices,
    ::NS(node_index_t)* SIXTRL_RESTRICT node_index_begin )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ptr_nodes_ctrl != nullptr )
        ? ptr_nodes_ctrl->availableNodeIndices( 
            max_num_node_indices, node_index_begin )
        : ::NS(ctrl_size_t){ 0 };
}

::NS(ctrl_size_t) NS(Controller_get_available_node_ids)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    ::NS(ctrl_size_t) const max_num_node_ids,
    ::NS(NodeId)* SIXTRL_RESTRICT node_ids_begin )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ptr_nodes_ctrl != nullptr )
        ? ptr_nodes_ctrl->availableNodeIds( max_num_node_ids, node_ids_begin )
        : ::NS(ctrl_size_t){ 0 };
}

::NS(ctrl_size_t) NS(Controller_get_available_base_node_infos)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    ::NS(ctrl_size_t) const max_num_node_infos,
    ::NS(NodeInfoBase) const** SIXTRL_RESTRICT ptr_node_infos_begin )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ptr_nodes_ctrl != nullptr )
        ? ptr_nodes_ctrl->availableBaseNodeInfos( 
            max_num_node_infos, ptr_node_infos_begin )
        : ::NS(ctrl_size_t){ 0 };
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(node_index_t) NS(Controller_get_node_index_by_node_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    const ::NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    
    return ( ( ptr_nodes_ctrl != nullptr ) && ( node_id != nullptr ) )
        ? ptr_nodes_ctrl->nodeIndex( *node_id ) : st::NODE_UNDEFINED_INDEX;
}

::NS(node_index_t) NS(Controller_get_node_index_by_platform_id_and_device_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    ::NS(node_platform_id_t) const platform_id, 
    ::NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    
    return ( ptr_nodes_ctrl != nullptr ) 
        ? ptr_nodes_ctrl->nodeIndex( platform_id, device_id ) 
        : st::NODE_UNDEFINED_INDEX;
}

::NS(node_index_t) NS(Controller_get_node_index_by_node_info)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT ptr_node_info )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    
    return ( ptr_nodes_ctrl != nullptr ) 
        ? ptr_nodes_ctrl->nodeIndex( ptr_node_info ) 
        : st::NODE_UNDEFINED_INDEX;
}

::NS(node_index_t) NS(Controller_get_node_index)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    
    return ( ptr_nodes_ctrl != nullptr ) 
        ? ptr_nodes_ctrl->nodeIndex( node_id_str ) 
        : st::NODE_UNDEFINED_INDEX;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(NodeId) const* NS(Controller_get_ptr_node_id_by_index)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    ::NS(arch_size_t)const node_index )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ptr_nodes_ctrl != nullptr )
        ? ptr_nodes_ctrl->ptrNodeId( node_index ) : nullptr;
}

::NS(NodeId) const* NS(Controller_get_ptr_node_id_by_platform_id_and_device_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    ::NS(node_platform_id_t) const platform_id,
    ::NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ptr_nodes_ctrl != nullptr )
        ? ptr_nodes_ctrl->ptrNodeId( platform_id, device_id ) : nullptr;
}

::NS(NodeId) const* NS(Controller_get_ptr_node_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ptr_nodes_ctrl != nullptr )
        ? ptr_nodes_ctrl->ptrNodeId( node_id_str ) : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(NodeInfoBase) const* NS(Controller_get_ptr_node_info_base_by_index)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    ::NS(arch_size_t)const node_index )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ptr_nodes_ctrl != nullptr )
        ? ptr_nodes_ctrl->ptrNodeInfoBase( node_index ) : nullptr;
}

::NS(NodeInfoBase) const* NS(Controller_get_ptr_node_info_base_by_node_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    const ::NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) && ( node_id != nullptr ) )
        ? ptr_nodes_ctrl->ptrNodeInfoBase( *node_id ) : nullptr;
}

::NS(NodeInfoBase) const*
NS(Controller_get_ptr_node_info_base_by_platform_id_and_device_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ptr_nodes_ctrl != nullptr )
        ? ptr_nodes_ctrl->ptrNodeInfoBase( platform_id, device_id ) : nullptr;
}

::NS(NodeInfoBase) const* NS(Controller_get_ptr_node_info_base)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ptr_nodes_ctrl != nullptr )
        ? ptr_nodes_ctrl->ptrNodeInfoBase( node_id_str ) : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Controller_has_selected_node)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) &&
             ( ptr_nodes_ctrl->hasSelectedNode() ) );
}

::NS(arch_size_t) NS(Controller_get_selected_node_index)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ptr_nodes_ctrl != nullptr )
        ? ptr_nodes_ctrl->selectedNodeIndex()
        : st::NODE_UNDEFINED_INDEX;
}

::NS(NodeId) const* NS(Controller_get_ptr_selected_node_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ptr_nodes_ctrl != nullptr )
        ? ptr_nodes_ctrl->ptrSelectedNodeId() : nullptr;
}

::NS(NodeInfoBase) const* NS(Controller_get_ptr_selected_node_info_base)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ptr_nodes_ctrl != nullptr )
        ? ptr_nodes_ctrl->ptrSelectedNodeInfoBase() : nullptr;
}

char const* NS(Controller_get_selected_node_id_str)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ptr_nodes_ctrl != nullptr )
        ? ptr_nodes_ctrl->ptrSelectedNodeIdStr() : nullptr;
}

bool NS(Controller_copy_selected_node_id_str)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    char* SIXTRL_RESTRICT node_id_str,
    ::NS(arch_size_t) const max_str_length )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ptr_nodes_ctrl != nullptr )
        ? ptr_nodes_ctrl->selectedNodeIdStr( node_id_str, max_str_length )
        : false;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Controller_select_node)( ::NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) &&
             ( ptr_nodes_ctrl->selectNode( node_id_str ) ) );
}

bool NS(Controller_select_node_by_node_id)(
    NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    const NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) && ( node_id != nullptr ) &&
             ( ptr_nodes_ctrl->selectNode( *node_id ) ) );
}

bool NS(Controller_select_node_by_plaform_id_and_device_id)(
    NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) &&
             ( ptr_nodes_ctrl->selectNode( platform_id, device_id ) ) );
}

bool NS(Controller_select_node_by_index)( 
    NS(ControllerBase)* SIXTRL_RESTRICT ctx,
    ::NS(node_index_t) const node_index )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctx );
    return ( ( ptr_nodes_ctrl != nullptr ) &&
             ( ptr_nodes_ctrl->selectNode( node_index ) ) );
}

/* ========================================================================= */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Controller_can_change_selected_node)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) &&
             ( ptr_nodes_ctrl->canChangeSelectedNode() ) );
}

bool NS(Controller_can_directly_change_selected_node)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) &&
             ( ptr_nodes_ctrl->canDirectlyChangeSelectedNode() ) );
}

bool NS(Controller_change_selected_node)(
    ::NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    ::NS(node_index_t) const current_selected_node_index,
    ::NS(node_index_t) const new_selected_node_index )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) &&
    ( ptr_nodes_ctrl->changeSelectedNode(
        current_selected_node_index, new_selected_node_index ) ) );
}

/* ========================================================================= */

bool NS(Controller_can_unselect_node)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) &&
             ( ptr_nodes_ctrl->canUnselectNode() ) );
}

bool NS(Controller_unselect_node)(
    NS(ControllerBase)* SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) &&
             ( ptr_nodes_ctrl->unselectNode() ) );
}

bool NS(Controller_unselect_node_by_index)(
    NS(ControllerBase)* SIXTRL_RESTRICT ctrl, NS(node_index_t) const index )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) &&
             ( ptr_nodes_ctrl->unselectNode( index ) ) );
}

bool NS(Controller_unselect_node_by_node_id)(
    NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    const NS(NodeId) *const SIXTRL_RESTRICT ptr_node_id )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) && ( ptr_node_id != nullptr ) &&
             ( ptr_nodes_ctrl->unselectNode( *ptr_node_id ) ) );
}

bool NS(Controller_unselect_node_by_platform_id_and_device_id)(
    NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) &&
             ( ptr_nodes_ctrl->unselectNode( platform_id, device_id ) ) );
}

bool NS(Controller_unselect_node_by_node_id_str)(
    NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    return ( ( ptr_nodes_ctrl != nullptr ) &&
             ( ptr_nodes_ctrl->unselectNode( node_id_str ) ) );
}

/* ========================================================================= */

void NS(Controller_print_available_nodes_info)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    FILE* SIXTRL_RESTRICT fp )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    if( ptr_nodes_ctrl != nullptr )
    {
        ptr_nodes_ctrl->printAvailableNodesInfo( fp );
    }
}

void NS(Controller_print_out_available_nodes_info)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    if( ptr_nodes_ctrl != nullptr )
    {
        ptr_nodes_ctrl->printAvailableNodesInfo();
    }
}

void NS(Controller_store_available_nodes_info_to_string)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl,
    char* SIXTRL_RESTRICT nodes_info_str,
    ::NS(arch_size_t)const nodes_info_str_capacity,
    ::NS(arch_size_t)* SIXTRL_RESTRICT ptr_required_max_str_length )
{
    auto ptr_nodes_ctrl = st::asNodeController( ctrl );
    if( ptr_nodes_ctrl != nullptr )
    {
        ptr_nodes_ctrl->storeAvailableNodesInfoToCString(
            nodes_info_str, nodes_info_str_capacity,
                ptr_required_max_str_length );
    }
}

/* end: sixtracklib/common/control/node_controller_base.cpp */
