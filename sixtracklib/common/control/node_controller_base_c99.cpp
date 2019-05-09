#include "sixtracklib/common/control/node_controller_base.h"

#if !defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace st = SIXTRL_CXX_NAMESPACE;

/* ========================================================================= */

::NS(ctrl_size_t) NS(Context_get_num_available_nodes)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->numAvailableNodes() : ::NS(ctrl_size_t){ 0 };
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Context_has_default_node)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->hasDefaultNode() ) );
}

NS(ctrl_size_t) NS(Context_get_default_node_index)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->defaultNodeIndex()
        : st::NODE_UNDEFINED_INDEX;
}

::NS(NodeId) const* NS(Context_get_default_node_id)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrDefaultNodeId() : nullptr;
}

::NS(NodeInfoBase) const* NS(Context_get_default_node_info_base)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrDefaultNodeInfoBase() : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Context_is_default_node_by_index)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    ::NS(ctrl_size_t)const node_index )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->isDefaultNode( node_index ) ) );
}

bool NS(Context_is_default_node_by_node_id)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    const NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ( ptr_nodes_ctx != nullptr ) && ( node_id != nullptr ) &&
             ( ptr_nodes_ctx->isDefaultNode( *node_id ) ) );
}

bool NS(Context_is_default_node_by_platform_device_ids)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->isDefaultNode( platform_id, device_id ) ) );
}

bool NS(Context_is_default_node)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->isDefaultNode( node_id_str ) ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Context_is_node_available_by_index)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    ::NS(ctrl_size_t)const node_index )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->isNodeAvailable( node_index ) ) );
}

bool NS(Context_is_node_available_by_node_id)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    const NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ( ptr_nodes_ctx != nullptr ) && ( node_id != nullptr ) &&
             ( ptr_nodes_ctx->isNodeAvailable( *node_id ) ) );
}

bool NS(Context_is_node_available_by_platform_device_ids)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->isNodeAvailable( platform_id, device_id ) ) );
}

bool NS(Context_is_node_available)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->isNodeAvailable( node_id_str ) ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(NodeId) const* NS(Context_get_ptr_node_id_by_index)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    ::NS(ctrl_size_t)const node_index )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrNodeId( node_index ) : nullptr;
}

::NS(NodeId) const* NS(Context_get_ptr_node_id_by_platform_device_ids)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrNodeId( platform_id, device_id ) : nullptr;
}

::NS(NodeId) const* NS(Context_get_ptr_node_id)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrNodeId( node_id_str ) : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(NodeInfoBase) const* NS(Context_get_ptr_node_info_base_by_index)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    ::NS(ctrl_size_t)const node_index )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrNodeInfoBase( node_index ) : nullptr;
}

::NS(NodeInfoBase) const* NS(Context_get_ptr_node_info_base_by_node_id)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    const NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrNodeInfoBase( node_id ) : nullptr;
}

::NS(NodeInfoBase) const*
NS(Context_get_ptr_node_info_base_by_platform_device_ids)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrNodeInfoBase( platform_id, device_id ) : nullptr;
}

::NS(NodeInfoBase) const* NS(Context_get_ptr_node_info_base)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrNodeInfoBase( node_id_str ) : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Context_has_selected_node)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->hasSelectedNode() ) );
}

::NS(ctrl_size_t) NS(Context_get_selected_node_index)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->selectedNodeIndex()
        : st::NODE_UNDEFINED_INDEX;
}

::NS(NodeId) const* NS(Context_get_ptr_selected_node_id)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrSelectedNodeId() : nullptr;
}

::NS(NodeInfoBase) const* NS(Context_get_ptr_selected_node_info_base)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrSelectedNodeInfoBase() : nullptr;
}

char const* NS(Context_get_selected_node_id_str)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrSelectedNodeIdStr() : nullptr;
}

bool NS(Context_copy_selected_node_id_str)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    char* SIXTRL_RESTRICT node_id_str,
    ::NS(ctrl_size_t) const max_str_length )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->selectedNodeIdStr( node_id_str, max_str_length )
        : false;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Context_select_node)( ::NS(ContextBase)* SIXTRL_RESTRICT context,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->selectNode( node_id_str ) ) );
}

bool NS(Context_select_node_by_node_id)(
    NS(ContextBase)* SIXTRL_RESTRICT context,
    const NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ( ptr_nodes_ctx != nullptr ) && ( node_id != nullptr ) &&
             ( ptr_nodes_ctx->selectNode( *node_id ) ) );
}

bool NS(Context_select_node_by_platform_device_ids)(
    NS(ContextBase)* SIXTRL_RESTRICT context,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->selectNode( platform_id, device_id ) ) );
}

bool NS(Context_select_node_by_index)( NS(ContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(ctrl_size_t)const node_index )
{
    auto ptr_nodes_ctx = st::asNodeController( ctx );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->selectNode( node_index ) ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

void NS(Context_print_available_nodes_info)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    FILE* SIXTRL_RESTRICT fp )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    if( ptr_nodes_ctx != nullptr )
    {
        ptr_nodes_ctx->printAvailableNodesInfo( fp );
    }
}

void NS(Context_print_out_available_nodes_info)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    if( ptr_nodes_ctx != nullptr )
    {
        ptr_nodes_ctx->printAvailableNodesInfo();
    }
}

void NS(Context_store_available_nodes_info_to_string)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    char* SIXTRL_RESTRICT nodes_info_str,
    ::NS(ctrl_size_t)const nodes_info_str_capacity,
    ::NS(ctrl_size_t)* SIXTRL_RESTRICT ptr_required_max_str_length )
{
    auto ptr_nodes_ctx = st::asNodeController( context );
    if( ptr_nodes_ctx != nullptr )
    {
        ptr_nodes_ctx->storeAvailableNodesInfoToCString(
            nodes_info_str, nodes_info_str_capacity,
                ptr_required_max_str_length );
    }
}

#endif /* C, Host */

/* end: sixtracklib/common/control/context_base_with_nodes_c99.cpp */
