#include "sixtracklib/common/context.h"

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/context/context_base.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

/* ------------------------------------------------------------------------- */
/* NS(ContextBase) related public API */

void NS(Context_delete)( ::NS(ContextBase)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) delete ctx;
    return;
}

void NS(Context_clear)( ::NS(ContextBase)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) ctx->clear();
    return;
}

::NS(context_type_id_t) NS(Context_get_type_id)(
    ::NS(ContextBase)* SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->type() : ::NS(context_type_id_t){ -1 };
}

char const* NS(Context_get_type_str)( ::NS(ContextBase)* SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->ptrTypeStr() : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Context_has_config_str)( ::NS(ContextBase)* SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->hasConfigStr() : false;
}

char const* NS(Context_get_config_str)(
    ::NS(ContextBase)* SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->ptrConfigStr() : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Context_uses_nodes)( ::NS(ContextBase)* SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->usesNodes() : false;
}

/* ------------------------------------------------------------------------- */
/* NS(ContextOnNodesBase) related public API */

::NS(context_size_t) NS(Context_get_num_available_nodes)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr ) ? ctx->numAvailableNodes()
        : ::NS(context_size_t){ 0 };
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- */

::NS(context_node_info_t) const* NS(Context_get_available_nodes_info_begin)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr ) ? ctx->availableNodesInfoBegin() : nullptr;
}

::NS(context_node_info_t) const* NS(Context_get_available_nodes_info_end)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr ) ? ctx->availableNodesInfoEnd() : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- */

::NS(context_node_id_t) NS(Context_get_default_node_id)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );

    if( ctx != nullptr )
    {
        return ctx->defaultNodeId();
    }

    ::NS(context_node_id_t) node_id;
    ::NS(ComputeNodeId_preset)( &node_id );

    return node_id;
}

::NS(context_node_info_t) const* NS(Context_get_default_node_info)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr ) ? ctx->defaultNodeInfo() : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- */

bool NS(Context_is_node_available_by_index)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_size_t) const node_index )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr ) ? ctx->isNodeAvailable( node_index ) : false;
}

bool NS(Context_is_node_available_by_node_id)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_node_id_t) const node_id )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr ) ? ctx->isNodeAvailable( node_id ) : false;
}

bool NS(Context_is_node_available_by_platform_device_ids)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_platform_id_t) const platform_index,
    ::NS(context_device_id_t) const device_index )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr )
        ? ctx->isNodeAvailable( platform_index, device_index ) : false;
}

bool NS(Context_is_node_available)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT node_id_str )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr ) ? ctx->isNodeAvailable( node_id_str ) : false;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- */

bool NS(Context_is_default_node)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    char const* node_id_str )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr ) ? ctx->isDefaultNode( node_id_str ) : false;
}

bool NS(Context_is_default_node_by_node_id)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_node_id_t) const node_id )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr ) ? ctx->isDefaultNode( node_id ) : false;
}

bool NS(Context_is_default_node_by_platform_device_ids)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_platform_id_t) const platform_index,
    ::NS(context_device_id_t) const device_index )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr )
        ? ctx->isDefaultNode( platform_index, device_index ) : false;
}

bool NS(Context_is_default_node_by_index)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_size_t) const node_index )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr ) ? ctx->isDefaultNode( node_index ) : false;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- */

::NS(context_node_id_t) const* NS(Context_get_ptr_available_nodes_id_by_index)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,(
    ::NS(context_size_t) const index )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr ) ? ctx->ptrAvailableNodesId( index ) : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- */

::NS(context_node_info_t) const*
NS(Context_get_ptr_available_nodes_info_by_index)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_size_t) const index )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr ) ? ctx->ptrAvailableNodesInfo( index ) : nullptr;
}

::NS(context_node_info_t) const*
NS(Context_get_ptr_available_nodes_info_by_platform_device_ids)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_platform_id_t) const platform_idx,
    ::NS(context_device_id_t) const device_idx )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr )
        ? ctx->ptrAvailableNodesInfo( platform_idx, device_idx ) : nullptr;
}

::NS(context_node_info_t) const*
NS(Context_get_ptr_available_nodes_info_by_node_id)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_node_id_t) const node_id )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr )
        ? ctx->ptrAvailableNodesInfo( node_id ) : nullptr;
}

::NS(context_node_info_t) const* NS(Context_get_ptr_available_nodes_info)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT node_id_str )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr )
        ? ctx->ptrAvailableNodesInfo( node_id_str ) : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- */

bool NS(Context_has_selected_node)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr ) ? ctx->hasSelectedNode() : false;
}

::NS(context_node_id_t) const* NS(Context_get_ptr_selected_node_id)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr ) ? ctx->ptrSelectedNodeId() : nullptr;
}

::NS(context_node_info_t) const* NS(Context_get_ptr_selected_node_info)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr ) ? ctx->ptrSelectedNodeInfo() : nullptr;
}

char const* NS(Context_selected_node_id_str)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr ) ? ctx->ptrSelectedNodeIdStr() : nullptr;
}

bool NS(Context_get_copy_of_selected_node_id_str)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    char* SIXTRL_RESTRICT node_id_str, ::NS(context_size_t) const max_length )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr )
        ? ctx->selectedNodeIdStr( node_id_str, max_length ) : false;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- */

bool NS(Context_select_node_by_node_id)(
    ::NS(ContextNodeBase)* SIXTRL_RESTRICT ctx,
    ::NS(context_node_id_t) const node_id )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr ) ? ctx->selectNode( node_id ) : false;
}

bool NS(Context_select_node_by_platform_device_ids)(
    ::NS(ContextNodeBase)* SIXTRL_RESTRICT ctx,
    ::NS(context_platform_id_t) const platform_idx,
    ::NS(context_device_id_t)  const device_idx )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr )
        ? ctx->selectNode( platform_idx, device_idx ) : false;
}

bool NS(Context_select_node)( ::NS(ContextNodeBase)* SIXTRL_RESTRICT ctx,
    const char *const SIXTRL_RESTRICT node_id_str )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr ) ? ctx->selectNode( node_id_str ) : false;
}

bool NS(Contet_select_node_by_index)(
    ::NS(ContextNodeBase)* SIXTRL_RESTRICT ctx,
    ::NS(context_size_t) const index )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    return ( ctx != nullptr ) ? ctx->selectNode( index ) : false;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- */

void NS(Context_print_out_nodes_info)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    if( ctx != nullptr ) ctx->printNodesInfo();
}

void NS(Context_print_out_nodes_info_by_index)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_size_t) const index )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    if( ctx != nullptr ) ctx->printNodesInfo( index );
}

void NS(Context_print_out_nodes_info_by_node_id)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_node_id_t) const node_id )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    if( ctx != nullptr ) ctx->printNodesInfo( node_id );
}

void NS(Context_print_out_nodes_info)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT node_id_str )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    if( ctx != nullptr ) ctx->printNodesInfo( node_id_str );
}

void NS(Context_print_out_selected_nodes_info)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx )
{
    SIXTRL_ASSERT( ::NS(Context_uses_nodes)( ctx ) );
    if( ctx != nullptr ) ctx->printSelectedNodesInfo();
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- */

void ::NS(Context_print_nodes_info)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp )
{
    auto it  = ::NS(Context_get_available_nodes_info_begin)( ctx );
    auto end = ::NS(Context_get_available_nodes_info_end)( ctx );

    if( it != nullptr )
    {
        auto default_node_id = ::NS(Context_get_default_node_id)( ctx );

        for( ; it != end ; ++it )
        {
            ::NS(ComputeNodeInfo_print)( fp, it, &default_node_id );
        }
    }

    return;
}

void NS(Context_print_nodes_info_by_index)(
    const ::NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    ::NS(context_size_t) const index )
{
    auto default_node_id = ::NS(Context_get_default_node_id)( ctx );

    ::NS(ComputeNodeInfo_print)( fp,
        ::NS(Context_get_ptr_available_nodes_info_by_index)( ctx, index ),
            &default_node_id );
}

void NS(Context_print_nodes_info_by_node_id)(
    const NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    NS(context_node_id_t) const node_id )
{
    auto default_node_id = ::NS(Context_get_default_node_id)( ctx );

    ::NS(ComputeNodeInfo_print)( fp,
        ::NS(Context_get_ptr_available_nodes_info_by_node_id)( ctx, node_id ),
            &default_node_id );
}

void NS(Context_print_nodes_info)(
    const NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    char const* SIXTRL_RESTRICT node_id_str )
{
    auto default_node_id = ::NS(Context_get_default_node_id)( ctx );

    ::NS(ComputeNodeInfo_print)( fp,
        ::NS(Context_get_ptr_available_nodes_info)( ctx, node_id_str ),
            &default_node_id );
}

void NS(Context_print_selected_nodes_info)(
    const NS(ContextNodeBase) *const SIXTRL_RESTRICT ctx,
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp )
{
    ::NS(ComputeNodeInfo_print)( fp,
        ::NS(Context_get_ptr_selected_node_info)( ctx ), nullptr );
}

/* end: sixtracklib/common/context/context.cpp */
