#include "sixtracklib/common/control/controller_on_nodes_base.h"

#if !defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_HOST_FN ControllerOnNodesBase const*
    ContextBase_convert_to_context_with_nodes(
        ContextBase const* SIXTRL_RESTRICT ptr_base_context )
    {
        return ( ( ptr_base_context != nullptr ) &&
                 ( ptr_base_context->usesNodes() ) )
            ? static_cast< ControllerOnNodesBase const* >( ptr_base_context )
            : nullptr;
    }

    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_HOST_FN ControllerOnNodesBase*
    ContextBase_convert_to_context_with_nodes(
        ContextBase* SIXTRL_RESTRICT ptr_base_context )
    {
        ContextBase const* ptr_const_base_context = ptr_base_context;

        return const_cast< ControllerOnNodesBase* >(
            ContextBase_convert_to_context_with_nodes(
                ptr_const_base_context ) );
    }
}

/* ========================================================================= */

using SIXTRL_CXX_NAMESPACE::ContextBase_convert_to_context_with_nodes;

::NS(context_size_t) NS(Context_get_num_available_nodes)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->numAvailableNodes() : ::NS(context_size_t){ 0 };
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Context_has_default_node)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->hasDefaultNode() ) );
}

NS(context_size_t) NS(Context_get_default_node_index)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->defaultNodeIndex()
        : SIXTRL_CXX_NAMESPACE::NODE_UNDEFINED_INDEX;
}

::NS(NodeId) const* NS(Context_get_default_node_id)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrDefaultNodeId() : nullptr;
}

::NS(NodeInfoBase) const* NS(Context_get_default_node_info_base)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrDefaultNodeInfoBase() : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Context_is_default_node_by_index)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    ::NS(context_size_t)const node_index )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->isDefaultNode( node_index ) ) );
}

bool NS(Context_is_default_node_by_node_id)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    const NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ( ptr_nodes_ctx != nullptr ) && ( node_id != nullptr ) &&
             ( ptr_nodes_ctx->isDefaultNode( *node_id ) ) );
}

bool NS(Context_is_default_node_by_platform_device_ids)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->isDefaultNode( platform_id, device_id ) ) );
}

bool NS(Context_is_default_node)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->isDefaultNode( node_id_str ) ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Context_is_node_available_by_index)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    ::NS(context_size_t)const node_index )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->isNodeAvailable( node_index ) ) );
}

bool NS(Context_is_node_available_by_node_id)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    const NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ( ptr_nodes_ctx != nullptr ) && ( node_id != nullptr ) &&
             ( ptr_nodes_ctx->isNodeAvailable( *node_id ) ) );
}

bool NS(Context_is_node_available_by_platform_device_ids)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->isNodeAvailable( platform_id, device_id ) ) );
}

bool NS(Context_is_node_available)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->isNodeAvailable( node_id_str ) ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(NodeId) const* NS(Context_get_ptr_node_id_by_index)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    ::NS(context_size_t)const node_index )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrNodeId( node_index ) : nullptr;
}

::NS(NodeId) const* NS(Context_get_ptr_node_id_by_platform_device_ids)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrNodeId( platform_id, device_id ) : nullptr;
}

::NS(NodeId) const* NS(Context_get_ptr_node_id)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrNodeId( node_id_str ) : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(NodeInfoBase) const* NS(Context_get_ptr_node_info_base_by_index)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    ::NS(context_size_t)const node_index )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrNodeInfoBase( node_index ) : nullptr;
}

::NS(NodeInfoBase) const* NS(Context_get_ptr_node_info_base_by_node_id)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    const NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrNodeInfoBase( node_id ) : nullptr;
}

::NS(NodeInfoBase) const*
NS(Context_get_ptr_node_info_base_by_platform_device_ids)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrNodeInfoBase( platform_id, device_id ) : nullptr;
}

::NS(NodeInfoBase) const* NS(Context_get_ptr_node_info_base)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrNodeInfoBase( node_id_str ) : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Context_has_selected_node)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->hasSelectedNode() ) );
}

::NS(context_size_t) NS(Context_get_selected_node_index)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->selectedNodeIndex()
        : SIXTRL_CXX_NAMESPACE::NODE_UNDEFINED_INDEX;
}

::NS(NodeId) const* NS(Context_get_ptr_selected_node_id)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrSelectedNodeId() : nullptr;
}

::NS(NodeInfoBase) const* NS(Context_get_ptr_selected_node_info_base)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrSelectedNodeInfoBase() : nullptr;
}

char const* NS(Context_get_selected_node_id_str)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->ptrSelectedNodeIdStr() : nullptr;
}

bool NS(Context_copy_selected_node_id_str)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    char* SIXTRL_RESTRICT node_id_str,
    ::NS(context_size_t) const max_str_length )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ptr_nodes_ctx != nullptr )
        ? ptr_nodes_ctx->selectedNodeIdStr( node_id_str, max_str_length )
        : false;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Context_select_node)( ::NS(ContextBase)* SIXTRL_RESTRICT context,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->selectNode( node_id_str ) ) );
}

bool NS(Context_select_node_by_node_id)(
    NS(ContextBase)* SIXTRL_RESTRICT context,
    const NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ( ptr_nodes_ctx != nullptr ) && ( node_id != nullptr ) &&
             ( ptr_nodes_ctx->selectNode( *node_id ) ) );
}

bool NS(Context_select_node_by_platform_device_ids)(
    NS(ContextBase)* SIXTRL_RESTRICT context,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->selectNode( platform_id, device_id ) ) );
}

bool NS(Context_select_node_by_index)( NS(ContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(context_size_t)const node_index )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( ctx );
    return ( ( ptr_nodes_ctx != nullptr ) &&
             ( ptr_nodes_ctx->selectNode( node_index ) ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

void NS(Context_print_available_nodes_info)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    FILE* SIXTRL_RESTRICT fp )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    if( ptr_nodes_ctx != nullptr )
    {
        ptr_nodes_ctx->printAvailableNodesInfo( fp );
    }
}

void NS(Context_print_out_available_nodes_info)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    if( ptr_nodes_ctx != nullptr )
    {
        ptr_nodes_ctx->printAvailableNodesInfo();
    }
}

void NS(Context_store_available_nodes_info_to_string)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT context,
    char* SIXTRL_RESTRICT nodes_info_str,
    ::NS(context_size_t)const nodes_info_str_capacity,
    NS(context_size_t)* SIXTRL_RESTRICT ptr_required_max_str_length )
{
    auto ptr_nodes_ctx = ContextBase_convert_to_context_with_nodes( context );
    if( ptr_nodes_ctx != nullptr )
    {
        ptr_nodes_ctx->storeAvailableNodesInfoToCString(
            nodes_info_str, nodes_info_str_capacity,
                ptr_required_max_str_length );
    }
}

#endif /* C, Host */

/* end: sixtracklib/common/control/context_base_with_nodes_c99.cpp */
