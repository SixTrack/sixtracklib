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
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->numAvailableNodes();
    }

    return ::NS(context_size_t){ 0 };
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- */

::NS(context_node_info_t) const* NS(Context_get_available_nodes_info_begin)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->availableNodesInfoBegin();
    }

    return nullptr;
}

::NS(context_node_info_t) const* NS(Context_get_available_nodes_info_end)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->availableNodesInfoEnd();
    }

    return nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- */

::NS(context_node_id_t) NS(Context_get_default_node_id)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->defaultNodeId();
    }

    ::NS(context_node_id_t) node_id;
    ::NS(ComputeNodeId_preset)( &node_id );

    return node_id;
}

::NS(context_node_info_t) const* NS(Context_get_default_node_info)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->defaultNodeInfo();
    }

    return nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- */

bool NS(Context_is_node_available_by_index)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_size_t) const node_index )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->isNodeAvailable( node_index );
    }

    return false;
}

bool NS(Context_is_node_available_by_node_id)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_node_id_t) const node_id )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->isNodeAvailable( node_id );
    }

    return false;
}

bool NS(Context_is_node_available_by_platform_device_ids)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_platform_id_t) const platform_idx,
    ::NS(context_device_id_t) const device_idx )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->isNodeAvailable( platform_idx, device_idx );
    }

    return false;
}

bool NS(Context_is_node_available)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT node_id_str )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->isNodeAvailable( node_id_str );
    }

    return false;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- */

bool NS(Context_is_default_node)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    char const* node_id_str )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->isDefaultNode( node_id_str );
    }

    return false;
}

bool NS(Context_is_default_node_by_node_id)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_node_id_t) const node_id )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->isDefaultNode( node_id );
    }

    return false;
}

bool NS(Context_is_default_node_by_platform_device_ids)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_platform_id_t) const platform_idx,
    ::NS(context_device_id_t) const device_idx )
{
     using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->isDefaultNode( platform_idx, device_idx );
    }

    return false;
}

bool NS(Context_is_default_node_by_index)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_size_t) const node_index )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->isDefaultNode( node_index );
    }

    return false;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- */

::NS(context_node_id_t) const* NS(Context_get_ptr_available_nodes_id_by_index)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_size_t) const index )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->ptrAvailableNodesId( index );
    }

    return nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- */

::NS(context_node_info_t) const*
NS(Context_get_ptr_available_nodes_info_by_index)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_size_t) const index )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->ptrAvailableNodesInfo( index );
    }

    return nullptr;
}

::NS(context_node_info_t) const*
NS(Context_get_ptr_available_nodes_info_by_platform_device_ids)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_platform_id_t) const platform_idx,
    ::NS(context_device_id_t) const device_idx )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->ptrAvailableNodesInfo(
            platform_idx, device_idx );
    }

    return nullptr;
}

::NS(context_node_info_t) const*
NS(Context_get_ptr_available_nodes_info_by_node_id)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_node_id_t) const node_id )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->ptrAvailableNodesInfo( node_id );
    }

    return nullptr;
}

::NS(context_node_info_t) const* NS(Context_get_ptr_available_nodes_info)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT node_id_str )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->ptrAvailableNodesInfo( node_id_str );
    }

    return nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- */

bool NS(Context_has_selected_node)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->hasSelectedNode();
    }

    return false;
}

::NS(context_node_id_t) const* NS(Context_get_ptr_selected_node_id)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->ptrSelectedNodeId();
    }

    return nullptr;
}

::NS(context_node_info_t) const* NS(Context_get_ptr_selected_node_info)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->ptrSelectedNodeInfo();
    }

    return nullptr;
}

char const* NS(Context_selected_node_id_str)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->ptrSelectedNodeIdStr();
    }

    return nullptr;
}

bool NS(Context_get_copy_of_selected_node_id_str)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    char* SIXTRL_RESTRICT node_id_str, ::NS(context_size_t) const max_length )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->selectedNodeIdStr( node_id_str, max_length );
    }

    return false;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- */

bool NS(Context_select_node_by_node_id)(
    ::NS(ContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(context_node_id_t) const node_id )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase)*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->selectNode( node_id );
    }

    return false;
}

bool NS(Context_select_node_by_platform_device_ids)(
    ::NS(ContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(context_platform_id_t) const platform_idx,
    ::NS(context_device_id_t)  const device_idx )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase)*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->selectNode( platform_idx, device_idx );
    }

    return false;
}

bool NS(Context_select_node)( ::NS(ContextBase)* SIXTRL_RESTRICT ctx,
    const char *const SIXTRL_RESTRICT node_id_str )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase)*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->selectNode( node_id_str );
    }

    return false;
}

bool NS(Contet_select_node_by_index)(
    ::NS(ContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(context_size_t) const index )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase)*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        return ptr_nodes_ctx->selectNode( index );
    }

    return false;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- */

void NS(Context_print_out_all_nodes_info)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        ptr_nodes_ctx->printNodesInfo();
    }
}

void NS(Context_print_out_nodes_info_by_index)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_size_t) const index )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        ptr_nodes_ctx->printNodesInfo( index );
    }
}

void NS(Context_print_out_nodes_info_by_node_id)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    ::NS(context_node_id_t) const node_id )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        ptr_nodes_ctx->printNodesInfo( node_id );
    }
}

void NS(Context_print_out_nodes_info)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT node_id_str )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        ptr_nodes_ctx->printNodesInfo( node_id_str );
    }
}

void NS(Context_print_out_selected_nodes_info)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );
        ptr_nodes_ctx->printSelectedNodesInfo();
    }
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- */

void NS(Context_print_all_nodes_info)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );

        auto it  = ptr_nodes_ctx->availableNodesInfoBegin();
        auto end = ptr_nodes_ctx->availableNodesInfoEnd();

        if( it != nullptr )
        {
            auto default_node_id = ptr_nodes_ctx->defaultNodeId();

            for( ; it != end ; ++it )
            {
                ::NS(ComputeNodeInfo_print)( fp, it, &default_node_id );
            }
        }
    }
}

void NS(Context_print_nodes_info_by_index)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    ::NS(context_size_t) const index )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );

        auto default_node_id = ptr_nodes_ctx->defaultNodeId();
        auto node_info = ptr_nodes_ctx->ptrAvailableNodesInfo( index );
        ::NS(ComputeNodeInfo_print)( fp, node_info, &default_node_id );
    }
}

void NS(Context_print_nodes_info_by_node_id)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    ::NS(context_node_id_t) const node_id )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );

        auto default_node_id = ptr_nodes_ctx->defaultNodeId();
        auto node_info = ptr_nodes_ctx->ptrAvailableNodesInfo( node_id );
        ::NS(ComputeNodeInfo_print)( fp, node_info, &default_node_id );
    }
}

void NS(Context_print_nodes_info)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    char const* SIXTRL_RESTRICT node_id_str )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );

        auto default_node_id = ptr_nodes_ctx->defaultNodeId();
        auto node_info = ptr_nodes_ctx->ptrAvailableNodesInfo( node_id_str );
        ::NS(ComputeNodeInfo_print)( fp, node_info, &default_node_id );
    }
}

void NS(Context_print_selected_nodes_info)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx,
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp )
{
    using ptr_nodes_ctx_t = ::NS(ContextOnNodesBase) const*;

    if( ( ctx != nullptr ) && ( ctx->usesNodes() ) )
    {
        ptr_nodes_ctx_t ptr_nodes_ctx = static_cast< ptr_nodes_ctx_t >( ctx );

        auto default_node_id = ptr_nodes_ctx->defaultNodeId();
        auto node_info = ptr_nodes_ctx->ptrSelectedNodeInfo();
        ::NS(ComputeNodeInfo_print)( fp, node_info, &default_node_id );
    }
}

/* end: sixtracklib/common/context/context.cpp */
