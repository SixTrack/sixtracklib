#include "sixtracklib/common/control/node_controller_base.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/node_id.h"
#include "sixtracklib/common/control/node_info.h"

namespace st = SIXTRL_CXX_NAMESPACE;

namespace SIXTRL_CXX_NAMESPACE
{
    NodeControllerBase::node_index_t
    NodeControllerBase::numAvailableNodes() const SIXTRL_NOEXCEPT
    {
        return this->m_available_nodes.size();
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool NodeControllerBase::hasDefaultNode() const SIXTRL_NOEXCEPT
    {
        bool const has_default_node = (
            ( this->m_ptr_default_node_id != nullptr ) &&
            ( this->m_ptr_default_node_id->hasIndex() ) &&
            ( this->m_ptr_default_node_id->index() <
            this->numAvailableNodes() ) );

        SIXTRL_ASSERT( ( !has_default_node ) ||
        ( ( this->m_ptr_default_node_id != nullptr ) &&
          ( this->m_ptr_default_node_id->hasIndex() ) &&
          ( this->numAvailableNodes() >
            this->m_ptr_default_node_id->index() ) &&
          ( this->m_available_nodes[
                this->m_ptr_default_node_id->index() ].get() != nullptr ) &&
          ( this->m_available_nodes[
                this->m_ptr_default_node_id->index() ].get()->ptrNodeId() ==
                this->m_ptr_default_node_id ) ) );

        return has_default_node;
    }

    NodeControllerBase::node_id_t const*
    NodeControllerBase::ptrDefaultNodeId() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_default_node_id;
    }

    NodeControllerBase::node_index_t
    NodeControllerBase::defaultNodeIndex() const SIXTRL_NOEXCEPT
    {
        using node_index_t = NodeControllerBase::node_index_t;
        node_index_t default_node_index = NodeId::UNDEFINED_INDEX;

        if( ( this->ptrDefaultNodeId() != nullptr ) &&
            ( this->ptrDefaultNodeId()->hasIndex() ) &&
            ( this->ptrDefaultNodeId()->index() <
              this->numAvailableNodes() ) )
        {
            default_node_index = this->ptrDefaultNodeId()->index();
        }

        return default_node_index;
    }

    NodeControllerBase::node_info_base_t const*
    NodeControllerBase::ptrDefaultNodeInfoBase() const SIXTRL_NOEXCEPT
    {
        using node_index_t = NodeControllerBase::node_index_t;
        using ptr_node_info_t = NodeControllerBase::node_info_base_t const*;

        node_index_t const default_node_idx = this->defaultNodeIndex();
        ptr_node_info_t ptr_node_info_base = nullptr;

        if( default_node_idx < this->numAvailableNodes() )
        {
            ptr_node_info_base =
                this->m_available_nodes[ default_node_idx ].get();

            if( ptr_node_info_base != nullptr )
            {
                if( ptr_node_info_base->ptrNodeId() !=
                    this->ptrDefaultNodeId() )
                {
                    ptr_node_info_base = nullptr;
                }
            }
        }

        return ptr_node_info_base;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool NodeControllerBase::isNodeAvailable(
        NodeControllerBase::node_index_t const idx ) const SIXTRL_NOEXCEPT
    {
        return ( ( idx != NodeId::UNDEFINED_INDEX ) &&
            ( idx <  this->numAvailableNodes() ) &&
            ( this->m_available_nodes[ idx ].get() != nullptr ) );
    }

    bool NodeControllerBase::isNodeAvailable(
        NodeControllerBase::node_id_t const& node_id ) const SIXTRL_NOEXCEPT
    {
        return this->isNodeAvailable( this->doFindAvailableNodesIndex(
            node_id.platformId(), node_id.deviceId() ) );
    }

    bool NodeControllerBase::isNodeAvailable(
        NodeControllerBase::platform_id_t const platform_id,
        NodeControllerBase::device_id_t const dev_id ) const SIXTRL_NOEXCEPT
    {
        return this->isNodeAvailable( this->doFindAvailableNodesIndex(
            platform_id, dev_id ) );
    }

    bool NodeControllerBase::isNodeAvailable(
        char const* node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->isNodeAvailable( this->doFindAvailableNodesIndex(
            node_id_str ) );
    }

    bool NodeControllerBase::isNodeAvailable(
        std::string const& node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->isNodeAvailable( this->doFindAvailableNodesIndex(
            node_id_str.c_str() ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool NodeControllerBase::isDefaultNode(
        char const* node_id_str ) const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasDefaultNode() ) &&
            ( this->defaultNodeIndex() == this->doFindAvailableNodesIndex(
                node_id_str ) ) );
    }

    bool NodeControllerBase::isDefaultNode( std::string const&
        SIXTRL_RESTRICT_REF node_id_str ) const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasDefaultNode() ) &&
            ( this->defaultNodeIndex() == this->doFindAvailableNodesIndex(
                node_id_str.c_str() ) ) );
    }

    bool NodeControllerBase::isDefaultNode( NodeControllerBase::node_id_t
        const& node_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasDefaultNode() ) &&
            ( this->defaultNodeIndex() == this->doFindAvailableNodesIndex(
                node_id.platformId(), node_id.deviceId() ) ) );
    }

    bool NodeControllerBase::isDefaultNode(
        NodeControllerBase::platform_id_t const platform_id,
        NodeControllerBase::device_id_t const dev_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasDefaultNode() ) &&
            ( this->defaultNodeIndex() == this->doFindAvailableNodesIndex(
                platform_id, dev_id ) ) );
    }

    bool NodeControllerBase::isDefaultNode(
        NodeControllerBase::node_index_t const idx ) const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasDefaultNode() ) &&
                 ( this->defaultNodeIndex() == idx ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool NodeControllerBase::isSelectedNode(
        char const* node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->isSelectedNode( this->doFindAvailableNodesIndex(
            node_id_str ) );
    }

    bool NodeControllerBase::isSelectedNode( std::string const&
        SIXTRL_RESTRICT_REF node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->isSelectedNode( this->doFindAvailableNodesIndex(
            node_id_str.c_str() ) );
    }

    bool NodeControllerBase::isSelectedNode(
        NodeControllerBase::node_id_t const& node_id ) const SIXTRL_NOEXCEPT
    {
        return this->isSelectedNode( this->doFindAvailableNodesIndex(
            node_id.platformId(), node_id.deviceId() ) );
    }

    bool NodeControllerBase::isSelectedNode(
        NodeControllerBase::platform_id_t const platform_id,
        NodeControllerBase::device_id_t const dev_id ) const SIXTRL_NOEXCEPT
    {
        return this->isSelectedNode( this->doFindAvailableNodesIndex(
            platform_id, dev_id ) );
    }

    bool NodeControllerBase::isSelectedNode(
        NodeControllerBase::node_index_t const idx ) const SIXTRL_NOEXCEPT
    {
        return ( ( idx != NodeControllerBase::UNDEFINED_INDEX ) &&
                 ( this->hasSelectedNode() ) &&
                 ( this->selectedNodeIndex() == idx ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    NodeControllerBase::node_index_t
    NodeControllerBase::minAvailableNodeIndex() const SIXTRL_NOEXCEPT
    {
        return this->m_min_available_node_index;
    }

    NodeControllerBase::node_index_t
    NodeControllerBase::maxAvailableNodeIndex() const SIXTRL_NOEXCEPT
    {
        return this->m_max_available_node_index;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    NodeControllerBase::size_type NodeControllerBase::availableNodeIndices(
        NodeControllerBase::size_type const max_num_node_indices,
        NodeControllerBase::node_index_t*
            SIXTRL_RESTRICT node_indices_begin ) const SIXTRL_NOEXCEPT
    {
        using _this_t      = st::NodeControllerBase;
        using size_t       = _this_t::size_type;
        using node_index_t = _this_t::node_index_t;

        size_t num_avail_elements = size_t{ 0 };

        if( ( node_indices_begin != nullptr ) &&
            ( this->numAvailableNodes() > node_index_t{ 0 } ) &&
            ( this->numAvailableNodes() != st::NODE_UNDEFINED_INDEX ) )
        {
            node_index_t ii = this->minAvailableNodeIndex();
            node_index_t const max_node_idx = this->maxAvailableNodeIndex();

            std::fill( node_indices_begin,
                       node_indices_begin + max_num_node_indices,
                       st::NODE_UNDEFINED_INDEX );

            SIXTRL_ASSERT( ii != st::NODE_UNDEFINED_INDEX );
            SIXTRL_ASSERT( max_node_idx != st::NODE_UNDEFINED_INDEX );
            SIXTRL_ASSERT( ii <= max_node_idx );
            SIXTRL_ASSERT( ( node_index_t{ 1 } + ( max_node_idx - ii ) ) <=
                this->numAvailableNodes() );

            while( ( ii <= max_node_idx ) &&
                   ( ii != st::NODE_UNDEFINED_INDEX ) &&
                   ( num_avail_elements < max_num_node_indices ) )
            {
                if( this->isNodeAvailable( ii ) )
                {
                    node_indices_begin[ num_avail_elements++ ] = ii;
                }

                ++ii;
            }
        }

        return num_avail_elements;
    }

    NodeControllerBase::size_type NodeControllerBase::availableNodeIds(
        NodeControllerBase::size_type const max_num_node_ids,
        NodeControllerBase::node_id_t* SIXTRL_RESTRICT
            node_ids_begin ) const SIXTRL_NOEXCEPT
    {
        using _this_t = st::NodeControllerBase;
        using size_t = _this_t::size_type;
        using node_id_t = _this_t::node_id_t;
        using node_index_t = _this_t::node_index_t;

        size_t num_avail_elements = size_t{ 0 };

        if( ( node_ids_begin != nullptr ) &&
            ( this->numAvailableNodes() > node_index_t{ 0 } ) &&
            ( this->numAvailableNodes() != st::NODE_UNDEFINED_INDEX ) )
        {
            node_index_t ii = this->minAvailableNodeIndex();
            node_index_t const max_node_idx = this->maxAvailableNodeIndex();

            SIXTRL_ASSERT( ii != st::NODE_UNDEFINED_INDEX );
            SIXTRL_ASSERT( max_node_idx != st::NODE_UNDEFINED_INDEX );
            SIXTRL_ASSERT( ii <= max_node_idx );
            SIXTRL_ASSERT( ( node_index_t{ 1 } + ( max_node_idx - ii ) ) <=
                this->numAvailableNodes() );

            std::fill( node_ids_begin,
                       node_ids_begin + max_num_node_ids, node_id_t{} );

            while( ( ii <= max_node_idx ) &&
                   ( ii != st::NODE_UNDEFINED_INDEX ) &&
                   ( num_avail_elements < max_num_node_ids ) )
            {
                if( this->isNodeAvailable( ii ) )
                {
                    node_id_t const* ptr_node_id = this->ptrNodeId( ii );
                    SIXTRL_ASSERT( ptr_node_id != nullptr );

                    node_ids_begin[ num_avail_elements++ ] = *ptr_node_id;
                }

                ++ii;
            }
        }

        return num_avail_elements;
    }

    NodeControllerBase::size_type NodeControllerBase::availableBaseNodeInfos(
        NodeControllerBase::size_type const max_num_node_infos,
        NodeControllerBase::node_info_base_t const**
            SIXTRL_RESTRICT ptr_node_infos_begin ) const SIXTRL_NOEXCEPT
    {
        using _this_t = st::NodeControllerBase;
        using size_t = _this_t::size_type;
        using node_index_t = _this_t::node_index_t;

        size_t num_avail_elements = size_t{ 0 };

        if( ( ptr_node_infos_begin != nullptr ) &&
            ( this->numAvailableNodes() > node_index_t{ 0 } ) &&
            ( this->numAvailableNodes() != st::NODE_UNDEFINED_INDEX ) )
        {
            node_index_t ii = this->minAvailableNodeIndex();
            node_index_t const max_node_idx = this->maxAvailableNodeIndex();

            std::fill( ptr_node_infos_begin,
                       ptr_node_infos_begin + max_num_node_infos, nullptr );

            SIXTRL_ASSERT( ii != st::NODE_UNDEFINED_INDEX );
            SIXTRL_ASSERT( max_node_idx != st::NODE_UNDEFINED_INDEX );
            SIXTRL_ASSERT( ii <= max_node_idx );
            SIXTRL_ASSERT( ( node_index_t{ 1 } + ( max_node_idx - ii ) ) <=
                this->numAvailableNodes() );

            while( ( ii <= max_node_idx ) &&
                   ( ii != st::NODE_UNDEFINED_INDEX ) &&
                   ( num_avail_elements < max_num_node_infos ) )
            {
                if( this->isNodeAvailable( ii ) )
                {
                    ptr_node_infos_begin[ num_avail_elements++ ] =
                        this->ptrNodeInfoBase( ii );
                }

                ++ii;
            }
        }

        return num_avail_elements;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    NodeControllerBase::node_index_t NodeControllerBase::nodeIndex(
        NodeControllerBase::node_id_t const&
            SIXTRL_RESTRICT_REF node_id ) const SIXTRL_NOEXCEPT
    {
        return this->doFindAvailableNodesIndex(
            node_id.platformId(), node_id.deviceId() );
    }

    NodeControllerBase::node_index_t NodeControllerBase::nodeIndex(
        NodeControllerBase::platform_id_t const platform_id,
        NodeControllerBase::device_id_t const device_id ) const SIXTRL_NOEXCEPT
    {
        return this->doFindAvailableNodesIndex( platform_id, device_id );
    }

    NodeControllerBase::node_index_t NodeControllerBase::nodeIndex(
        NodeControllerBase::node_info_base_t const* SIXTRL_RESTRICT
            ptr_node_info ) const SIXTRL_NOEXCEPT
    {
        NodeControllerBase::node_id_t const* ptr_node_id =
            ( ptr_node_info != nullptr ) ? &ptr_node_info->nodeId() : nullptr;

        return ( ptr_node_id != nullptr )
            ? this->doFindAvailableNodesIndex(
                ptr_node_id->platformId(), ptr_node_id->deviceId() )
            : st::NODE_UNDEFINED_INDEX;
    }

    NodeControllerBase::node_index_t NodeControllerBase::nodeIndex(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->doFindAvailableNodesIndex( node_id_str );
    }

    NodeControllerBase::node_index_t NodeControllerBase::nodeIndex( std::string
        const& SIXTRL_RESTRICT_REF node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->doFindAvailableNodesIndex( node_id_str.c_str() );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    NodeControllerBase::node_id_t const* NodeControllerBase::ptrNodeId(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->ptrNodeId( this->doFindAvailableNodesIndex(
            node_id_str ) );
    }

    NodeControllerBase::node_id_t const*
    NodeControllerBase::ptrNodeId( std::string const&
        SIXTRL_RESTRICT_REF node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->ptrNodeId( this->doFindAvailableNodesIndex(
            node_id_str.c_str() ) );
    }

    NodeControllerBase::node_id_t const* NodeControllerBase::ptrNodeId(
        NodeControllerBase::platform_id_t const platform_id,
        NodeControllerBase::device_id_t const dev_id ) const SIXTRL_NOEXCEPT
    {
        return this->ptrNodeId( this->doFindAvailableNodesIndex(
            platform_id, dev_id ) );
    }

    NodeControllerBase::node_id_t const* NodeControllerBase::ptrNodeId(
        NodeControllerBase::node_index_t const idx ) const SIXTRL_NOEXCEPT
    {
        NodeControllerBase::node_id_t const* ptr_node_id = nullptr;

        if( ( idx < this->numAvailableNodes() ) &&
            ( this->m_available_nodes[ idx ].get() != nullptr ) )
        {
            ptr_node_id = this->m_available_nodes[ idx ]->ptrNodeId();
        }

        return ptr_node_id;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    NodeControllerBase::node_info_base_t const*
    NodeControllerBase::ptrNodeInfoBase(
        NodeControllerBase::node_index_t const index ) const SIXTRL_NOEXCEPT
    {
        NodeControllerBase::node_info_base_t const*
            ptr_node_info_base = nullptr;

        if( ( index < this->numAvailableNodes() ) &&
            ( this->m_available_nodes[ index ].get() != nullptr ) )
        {
            ptr_node_info_base = this->m_available_nodes[ index ].get();
        }

        return ptr_node_info_base;
    }

    NodeControllerBase::node_info_base_t const*
    NodeControllerBase::ptrNodeInfoBase(
        NodeControllerBase::platform_id_t const platform_id,
        NodeControllerBase::device_id_t const dev_id ) const SIXTRL_NOEXCEPT
        {
        return this->ptrNodeInfoBase( this->doFindAvailableNodesIndex(
            platform_id, dev_id ) );
        }

    NodeControllerBase::node_info_base_t const*
    NodeControllerBase::ptrNodeInfoBase(
        NodeControllerBase::node_id_t const& node_id ) const SIXTRL_NOEXCEPT
    {
        return this->ptrNodeInfoBase( this->doFindAvailableNodesIndex(
            node_id.platformId(), node_id.deviceId() ) );
    }

    NodeControllerBase::node_info_base_t const*
    NodeControllerBase::ptrNodeInfoBase(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->ptrNodeInfoBase( this->doFindAvailableNodesIndex(
            node_id_str ) );
    }

    NodeControllerBase::node_info_base_t const*
    NodeControllerBase::ptrNodeInfoBase( std::string const&
        SIXTRL_RESTRICT_REF node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->ptrNodeInfoBase( this->doFindAvailableNodesIndex(
            node_id_str.c_str() ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool NodeControllerBase::hasSelectedNode() const SIXTRL_NOEXCEPT
    {
        bool const has_selected_node = (
            ( this->m_ptr_selected_node_id != nullptr ) &&
            ( this->numAvailableNodes() > this->selectedNodeIndex() ) );

        SIXTRL_ASSERT( ( !has_selected_node ) ||
            ( ( this->m_ptr_selected_node_id != nullptr ) &&
              ( this->selectedNodeIndex() != NodeId::UNDEFINED_INDEX ) &&
              ( this->m_ptr_selected_node_id ==
                this->ptrNodeId( this->selectedNodeIndex() ) ) ) );

        return has_selected_node;
    }

    NodeControllerBase::node_index_t
    NodeControllerBase::selectedNodeIndex() const SIXTRL_NOEXCEPT
    {
        return ( ( this->m_ptr_selected_node_id != nullptr ) &&
                    ( this->m_ptr_selected_node_id->hasIndex() ) )
            ? this->m_ptr_selected_node_id->index()
            : NodeId::UNDEFINED_INDEX;
    }

    NodeControllerBase::node_id_t const*
    NodeControllerBase::ptrSelectedNodeId() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_selected_node_id;
    }

    NodeControllerBase::node_info_base_t const*
    NodeControllerBase::ptrSelectedNodeInfoBase() const SIXTRL_NOEXCEPT
    {
        return ( this->hasSelectedNode() )
            ? this->m_available_nodes[ this->selectedNodeIndex() ].get()
            : nullptr;
    }

    std::string
    NodeControllerBase::selectedNodeIdStr() const SIXTRL_NOEXCEPT
    {
        return std::string{ this->m_selected_node_id_str.data() };
    }

    char const*
    NodeControllerBase::ptrSelectedNodeIdStr() const SIXTRL_NOEXCEPT
    {
        return ( this->hasSelectedNode() )
            ? this->m_selected_node_id_str.data() : nullptr;
    }

    NodeControllerBase::status_t NodeControllerBase::selectedNodeIdStr(
        char* SIXTRL_RESTRICT node_id_str, NodeControllerBase::size_type
            const max_str_length ) const SIXTRL_NOEXCEPT
    {
        NodeControllerBase::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        using size_t = NodeControllerBase::size_type;

        if( this->hasSelectedNode() )
        {
            SIXTRL_ASSERT( this->ptrSelectedNodeId() != nullptr );
            status = this->ptrSelectedNodeId()->toString(
                node_id_str, max_str_length );
        }
        else if( ( node_id_str != nullptr ) &&
                 ( max_str_length > size_t{ 0 } ) )
        {
            std::memset( node_id_str, static_cast< int >( '\0' ),
                         max_str_length );
        }

        return status;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool NodeControllerBase::usesAutoSelect() const SIXTRL_NOEXCEPT
    {
        return this->m_use_autoselect;
    }

    NodeControllerBase::status_t NodeControllerBase::selectNode(
        NodeControllerBase::node_index_t const new_selected_node_index )
    {
        NodeControllerBase::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        using node_index_t = NodeControllerBase::node_index_t;

        if( !this->hasSelectedNode() )
        {
            status = this->doSelectNode( new_selected_node_index );
        }
        else if( this->canChangeSelectedNode() )
        {
            node_index_t const current_selected_node_index =
                this->selectedNodeIndex();

            SIXTRL_ASSERT( current_selected_node_index !=
                           NodeControllerBase::UNDEFINED_INDEX );

            status = this->doChangeSelectedNode(
                current_selected_node_index, new_selected_node_index );
        }

        return status;
    }

    NodeControllerBase::status_t NodeControllerBase::selectNode(
        NodeControllerBase::node_id_t const& node_id )
    {
        return this->selectNode( this->doFindAvailableNodesIndex(
            node_id.platformId(), node_id.deviceId() ) );
    }

    NodeControllerBase::status_t NodeControllerBase::selectNode(
        NodeControllerBase::platform_id_t const platform_idx,
        NodeControllerBase::device_id_t const device_idx )
    {
        return this->selectNode( this->doFindAvailableNodesIndex(
            platform_idx, device_idx ) );
    }

    NodeControllerBase::status_t NodeControllerBase::selectNode(
        char const* node_id_str )
    {
        return this->selectNode( this->doFindAvailableNodesIndex(
            node_id_str ) );
    }

    NodeControllerBase::status_t NodeControllerBase::selectNode(
        std::string const& node_id_str )
    {
        return this->selectNode( this->doFindAvailableNodesIndex(
            node_id_str.c_str() ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool NodeControllerBase::canChangeSelectedNode() const SIXTRL_NOEXCEPT
    {
        return ( ( this->canDirectlyChangeSelectedNode() ) ||
                 ( this->canUnselectNode() ) );
    }

    bool NodeControllerBase::canDirectlyChangeSelectedNode()
        const SIXTRL_NOEXCEPT
    {
        return this->m_can_directly_change_selected_node;
    }

    NodeControllerBase::status_t NodeControllerBase::changeSelectedNode(
        NodeControllerBase::node_index_t const new_selected_node_index )
    {
        return this->doChangeSelectedNode(
            this->selectedNodeIndex(), new_selected_node_index );
    }

    NodeControllerBase::status_t NodeControllerBase::changeSelectedNode(
        NodeControllerBase::node_index_t const current_selected_node_index,
        NodeControllerBase::node_index_t const new_selected_node_index )
    {
        return this->doChangeSelectedNode(
            current_selected_node_index, new_selected_node_index );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool NodeControllerBase::canUnselectNode() const SIXTRL_NOEXCEPT
    {
        return this->m_can_unselect_node;
    }

    NodeControllerBase::status_t NodeControllerBase::unselectNode()
    {
        return this->doUnselectNode( this->selectedNodeIndex() );
    }

    NodeControllerBase::status_t NodeControllerBase::unselectNode(
        NodeControllerBase::node_index_t const index )
    {
        bool const is_currently_default_node =
            this->isDefaultNode( index );

        NodeControllerBase::status_t status = this->doUnselectNode( index );

        if( ( status == st::ARCH_STATUS_SUCCESS ) &&
            ( this->usesAutoSelect() ) && ( !is_currently_default_node ) &&
            ( this->hasDefaultNode() ) )
        {
            status = this->doSelectNode( this->defaultNodeIndex() );
        }

        return status;
    }

    NodeControllerBase::status_t NodeControllerBase::unselectNode(
        NodeControllerBase::node_id_t const& node_id )
    {
        return this->unselectNode( this->doFindAvailableNodesIndex(
            node_id.platformId(), node_id.deviceId() ) );
    }

    NodeControllerBase::status_t NodeControllerBase::unselectNode(
        NodeControllerBase::platform_id_t const platform_idx,
        NodeControllerBase::device_id_t const device_idx )
    {
        return this->unselectNode( this->doFindAvailableNodesIndex(
            platform_idx, device_idx ) );
    }

    NodeControllerBase::status_t NodeControllerBase::unselectNode(
        char const* node_id_str )
    {
        return this->unselectNode( this->doFindAvailableNodesIndex(
            node_id_str ) );
    }

    NodeControllerBase::status_t NodeControllerBase::unselectNode(
        std::string const& node_id_str )
    {
        return this->unselectNode( this->doFindAvailableNodesIndex(
            node_id_str.c_str() ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    void NodeControllerBase::printAvailableNodesInfo() const
    {
        this->printAvailableNodesInfo( std::cout );
    }

    void NodeControllerBase::printAvailableNodesInfo(
        std::ostream& SIXTRL_RESTRICT_REF output ) const
    {
        using node_index_t = NodeControllerBase::node_index_t;
        using node_info_base_t = NodeControllerBase::node_info_base_t;
        using ptr_node_info_t  = node_info_base_t const*;

        node_index_t const num_avail_nodes = this->numAvailableNodes();

        for( node_index_t ii = node_index_t{ 0 } ; ii < num_avail_nodes ; ++ii )
        {
            ptr_node_info_t ptr_node_info = this->ptrNodeInfoBase( ii );

            if( ptr_node_info != nullptr )
            {
                if( num_avail_nodes > node_index_t{ 1 } )
                {
                    output << "Node: "
                            << std::setw( 6 ) << ii + node_index_t{ 1 }
                            << " / " << num_avail_nodes << "\r\n";
                }

                output << *ptr_node_info << "\r\n";
            }
        }
    }

    std::string NodeControllerBase::availableNodesInfoToString() const
    {
        using _this_t = NodeControllerBase;

        if( this->numAvailableNodes() > _this_t::node_index_t{ 0 } )
        {
            std::ostringstream a2str;
            this->printAvailableNodesInfo( a2str );
            return a2str.str();
        }

        return std::string{};
    }

    void NodeControllerBase::printAvailableNodesInfo(
        ::FILE* SIXTRL_RESTRICT output ) const
    {
        using _this_t = NodeControllerBase;

        if( ( output != nullptr ) &&
            ( this->numAvailableNodes() > _this_t::node_index_t{ 0 } ) )
        {
            std::string const strinfo = this->availableNodesInfoToString();

            if( !strinfo.empty() )
            {
                int const ret = std::fprintf(
                    output, "%s", strinfo.c_str() );

                SIXTRL_ASSERT( ret > 0 );
                ( void )ret;
            }
        }

        return;
    }

    void NodeControllerBase::storeAvailableNodesInfoToCString(
        char* SIXTRL_RESTRICT nodes_info_str,
        NodeControllerBase::size_type const nodes_info_str_capacity,
        NodeControllerBase::size_type* SIXTRL_RESTRICT
            ptr_required_max_nodes_info_str_length ) const
    {
        using size_t = NodeControllerBase::size_type;
        size_t info_str_length = size_t{ 0 };

        if( ( nodes_info_str != nullptr ) &&
            ( nodes_info_str_capacity > size_t{ 0 } ) )
        {
            size_t const max_str_length =
                nodes_info_str_capacity - size_t{ 1 };

            std::string const info_str(
                this->availableNodesInfoToString() );

            info_str_length = info_str.size();

            size_t const chars_to_copy =
                std::min( max_str_length, info_str_length );

            auto info_str_begin = info_str.begin();
            auto info_str_end = info_str_begin;
            std::advance( info_str_end, chars_to_copy );

            std::memset( nodes_info_str, static_cast< int >( '\0' ),
                         nodes_info_str_capacity );

            std::copy( info_str_begin, info_str_end, nodes_info_str );
        }
        else if( ptr_required_max_nodes_info_str_length != nullptr )
        {
            std::ostringstream a2str;
            this->printAvailableNodesInfo( a2str );
            a2str.seekp( 0, std::ios::end );
            info_str_length = a2str.tellp();
        }

        if( ptr_required_max_nodes_info_str_length != nullptr )
        {
            *ptr_required_max_nodes_info_str_length = info_str_length;
        }

        return;
    }

    NodeControllerBase::~NodeControllerBase() SIXTRL_NOEXCEPT
    {

    }

    NodeControllerBase::NodeControllerBase(
        NodeControllerBase::arch_id_t const arch_id,
        const char *const SIXTRL_RESTRICT arch_str,
        const char *const SIXTRL_RESTRICT config_str ) :
        ControllerBase( arch_id, arch_str, config_str ),
        m_available_nodes(),
        m_selected_node_id_str(
            NodeControllerBase::NODE_ID_STR_CAPACITY, char{ '\0' } ),
        m_ptr_default_node_id( nullptr ),
        m_ptr_selected_node_id( nullptr ),
        m_min_available_node_index( NodeControllerBase::UNDEFINED_INDEX ),
        m_max_available_node_index( NodeControllerBase::UNDEFINED_INDEX ),
        m_can_directly_change_selected_node( false ),
        m_can_unselect_node( false ),
        m_use_autoselect( false )
    {
        this->doSetUsesNodesFlag( true );
    }

    void NodeControllerBase::doClear()
    {
        this->doClearNodeControllerBaseImpl();
        this->doClearAvailableNodes();
        ControllerBase::doClear();

        return;
    }

    NodeControllerBase::status_t NodeControllerBase::doSelectNode(
        NodeControllerBase::node_index_t const node_index )
    {
        return this->doSelectNodeNodeControllerBaseImpl( node_index );
    }

    NodeControllerBase::status_t NodeControllerBase::doChangeSelectedNode(
        NodeControllerBase::node_index_t const current_selected_node_index,
        NodeControllerBase::node_index_t const new_selected_node_index )
    {
        NodeControllerBase::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->hasSelectedNode() ) &&
            ( current_selected_node_index != new_selected_node_index ) &&
            ( this->isNodeAvailable( new_selected_node_index ) ) &&
            ( this->isSelectedNode( current_selected_node_index ) ) &&
            ( !this->isSelectedNode( new_selected_node_index ) ) &&
            ( this->canChangeSelectedNode() ) )
        {
            if( this->canDirectlyChangeSelectedNode() )
            {
                this->doRemoveNodeFromSelection( current_selected_node_index );

                if( !this->isSelectedNode( current_selected_node_index ) )
                {
                    status = this->doSelectNode( new_selected_node_index );
                }
            }
            else if( this->canUnselectNode() )
            {
                status = this->doUnselectNode( this->selectedNodeIndex() );

                if( status == st::ARCH_STATUS_SUCCESS )
                {
                    status = this->doSelectNode( new_selected_node_index );
                }
            }
        }

        return status;
    }

    NodeControllerBase::status_t NodeControllerBase::doUnselectNode(
        NodeControllerBase::node_index_t const selected_node_index )
    {
        NodeControllerBase::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->isSelectedNode( selected_node_index ) ) &&
            ( this->canUnselectNode() ) )
        {
            this->doRemoveNodeFromSelection( selected_node_index );

            if( !this->isSelectedNode( selected_node_index ) )
            {
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    NodeControllerBase::node_index_t
    NodeControllerBase::doFindAvailableNodesIndex(
        NodeControllerBase::platform_id_t const platform_id,
        NodeControllerBase::device_id_t const dev_id ) const SIXTRL_NOEXCEPT
    {
        using node_index_t = NodeControllerBase::node_index_t;
        using node_info_base_t = NodeControllerBase::node_info_base_t;
        using ptr_node_info_t  = node_info_base_t const*;

        node_index_t node_index = NodeId::UNDEFINED_INDEX;

        if( ( platform_id != NodeId::ILLEGAL_PLATFORM_ID ) &&
            ( dev_id != NodeId::ILLEGAL_DEVICE_ID ) &&
            ( this->numAvailableNodes() > node_index_t{ 0 } ) )
        {
            node_index_t const num_avail_nodes = this->numAvailableNodes();

            for( node_index_t ii = node_index_t{ 0 } ;
                    ii < num_avail_nodes ; ++ii )
            {
                ptr_node_info_t try_node = this->ptrNodeInfoBase( ii );

                if( ( try_node != nullptr ) &&
                    ( try_node->platformId() == platform_id ) &&
                    ( try_node->deviceId() == dev_id ) )
                {
                    SIXTRL_ASSERT( ( !try_node->hasNodeIndex() ) ||
                        ( ( try_node->hasNodeIndex() ) &&
                            ( ii == try_node->nodeIndex() ) ) );

                    node_index = ii;
                    break;
                }
            }
        }

        return node_index;
    }

    NodeControllerBase::node_index_t
    NodeControllerBase::doFindAvailableNodesIndex(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        using node_index_t = NodeControllerBase::node_index_t;
        using node_id_t = NodeControllerBase::node_id_t;

        node_index_t node_index = NodeId::UNDEFINED_INDEX;

        if( this->numAvailableNodes() > node_index_t{ 0 } )
        {
            node_id_t const cmp_node_id( node_id_str );

            if( cmp_node_id.valid() )
            {
                node_index = this->doFindAvailableNodesIndex(
                    cmp_node_id.platformId(), cmp_node_id.deviceId() );
            }
        }

        return node_index;
    }

    void NodeControllerBase::doClearAvailableNodes() SIXTRL_NOEXCEPT
    {
        this->m_available_nodes.clear();
        this->m_min_available_node_index = st::NodeId::UNDEFINED_INDEX;
        this->m_max_available_node_index = st::NodeId::UNDEFINED_INDEX;
        return;
    }

    void NodeControllerBase::doRemoveNodeFromSelection(
         NodeControllerBase::node_index_t const node_index ) SIXTRL_NOEXCEPT
    {
        if( this->isSelectedNode( node_index ) )
        {
            SIXTRL_ASSERT( this->canUnselectNode() );

            std::fill( this->m_selected_node_id_str.begin(),
                       this->m_selected_node_id_str.end(), '\0' );

            this->m_ptr_selected_node_id = nullptr;
        }
    }

    NodeControllerBase::node_index_t
    NodeControllerBase::doAppendAvailableNodeInfoBase(
        NodeControllerBase::ptr_node_info_base_t&& ptr_node_info_base )
    {
        using node_index_t = NodeControllerBase::node_index_t;
        using node_id_t = NodeControllerBase::node_id_t;

        node_index_t new_index = node_id_t::UNDEFINED_INDEX;

        if( ( ptr_node_info_base.get() != nullptr ) &&
            ( ptr_node_info_base->ptrNodeId() != nullptr ) )
        {
            using platform_id_t = node_id_t::platform_id_t;
            using device_id_t = node_id_t::device_id_t;

            node_id_t const* ptr_node_id = ptr_node_info_base->ptrNodeId();
            bool const node_id_valid = ptr_node_id->valid();

            platform_id_t platform_id = ptr_node_id->platformId();
            device_id_t dev_id = ptr_node_id->deviceId();

            if( ( node_id_valid ) && ( node_id_t::UNDEFINED_INDEX ==
                    this->doFindAvailableNodesIndex( platform_id, dev_id ) ) )
            {
                new_index = this->numAvailableNodes();
            }
            else if( !ptr_node_id->valid() )
            {
                if( ptr_node_info_base->hasPlatformName() )
                {
                    platform_id = this->doGetPlatformIdByPlatformName(
                        ptr_node_info_base->platformName().c_str() );

                    SIXTRL_ASSERT(
                        ( platform_id == node_id_t::ILLEGAL_PLATFORM_ID ) ||
                        ( platform_id == ptr_node_id->platformId() ) );

                    if( platform_id != node_id_t::ILLEGAL_PLATFORM_ID )
                    {
                        dev_id = this->doGetNextDeviceIdForPlatform(
                            platform_id );
                    }
                }

                if( platform_id == node_id_t::ILLEGAL_PLATFORM_ID )
                {
                    platform_id = this->doGetNextPlatformId();
                    dev_id = device_id_t{ 0 };
                }

                if( ( platform_id != node_id_t::ILLEGAL_PLATFORM_ID ) &&
                    ( dev_id != node_id_t::ILLEGAL_DEVICE_ID ) )
                {
                    ptr_node_info_base->setPlatformId( platform_id );
                    ptr_node_info_base->setDeviceId( dev_id );
                    new_index = this->numAvailableNodes();
                }
            }

            if( new_index != node_id_t::UNDEFINED_INDEX )
            {
                ptr_node_info_base->setNodeIndex( new_index );

                this->m_available_nodes.push_back(
                    std::move( ptr_node_info_base ) );

                if( ( this->minAvailableNodeIndex() ==
                        node_id_t::UNDEFINED_INDEX ) ||
                    ( this->minAvailableNodeIndex() > new_index ) )
                {
                    this->m_min_available_node_index = new_index;
                }

                if( ( this->maxAvailableNodeIndex() ==
                      node_id_t::UNDEFINED_INDEX ) ||
                    ( this->maxAvailableNodeIndex() < new_index ) )
                {
                    this->m_max_available_node_index = new_index;
                }
            }
        }

        return new_index;
    }

    NodeControllerBase::node_info_base_t*
    NodeControllerBase::doGetPtrNodeInfoBase(
        NodeControllerBase::node_index_t const node_index ) SIXTRL_NOEXCEPT
    {
        return const_cast< NodeControllerBase::node_info_base_t* >(
            static_cast< NodeControllerBase const& >(
                *this ).ptrNodeInfoBase( node_index ) );
    }

    void NodeControllerBase::doSetDefaultNodeIndex(
        NodeControllerBase::node_index_t const node_index ) SIXTRL_NOEXCEPT
    {
        using node_info_base_t = NodeControllerBase::node_info_base_t;

        if( node_index != NodeId::UNDEFINED_INDEX )
        {
            node_info_base_t* ptr_node_info_base =
                this->doGetPtrNodeInfoBase( node_index );

            if( ptr_node_info_base != nullptr)
            {
                if( !ptr_node_info_base->hasNodeIndex() )
                {
                    ptr_node_info_base->setNodeIndex( node_index );
                }

                SIXTRL_ASSERT(
                    ( ptr_node_info_base->hasNodeIndex() ) &&
                    ( ptr_node_info_base->nodeIndex() == node_index ) );

                ptr_node_info_base->setIsDefaultNode( true );

                this->m_ptr_default_node_id =
                    ptr_node_info_base->ptrNodeId();
            }
        }
        else
        {
            this->m_ptr_default_node_id = nullptr;
        }

        return;
    }

    NodeControllerBase::platform_id_t
    NodeControllerBase::doGetNextPlatformId() const SIXTRL_NOEXCEPT
    {
        using node_info_base_t = NodeControllerBase::node_info_base_t;
        using platform_id_t = NodeControllerBase::platform_id_t;
        using node_index_t = NodeControllerBase::node_index_t;

        platform_id_t next_platform_id = platform_id_t{ 0 };
        node_index_t const num_avail_nodes = this->numAvailableNodes();

        if( num_avail_nodes > node_index_t{ 0 } )
        {
            node_index_t ii = node_index_t{ 0 };

            for( ; ii < num_avail_nodes ; ++ii )
            {
                node_info_base_t const* nodeinfo = this->ptrNodeInfoBase( ii );

                if( ( nodeinfo != nullptr ) &&
                    ( nodeinfo->ptrNodeId() != nullptr ) &&
                    ( nodeinfo->ptrNodeId()->valid() ) &&
                    ( nodeinfo->platformId() >= next_platform_id ) )
                {
                    next_platform_id = nodeinfo->platformId();
                    ++next_platform_id;
                }
            }
        }

        return next_platform_id;
    }

    NodeControllerBase::device_id_t
    NodeControllerBase::doGetNextDeviceIdForPlatform(
        NodeControllerBase::platform_id_t const platform_id
        ) const SIXTRL_NOEXCEPT
    {
        using node_info_base_t = NodeControllerBase::node_info_base_t;
        using node_index_t = NodeControllerBase::node_index_t;
        using device_id_t = NodeControllerBase::device_id_t;
        using node_id_t = NodeControllerBase::node_id_t;

        device_id_t next_device_id = node_id_t::ILLEGAL_DEVICE_ID;

        node_index_t const num_avail_nodes = this->numAvailableNodes();

        if( num_avail_nodes > node_index_t{ 0 } )
        {
            node_index_t ii  = node_index_t{ 0 };

            for( ; ii < num_avail_nodes ; ++ii )
            {
                node_info_base_t const* nodeinfo = this->ptrNodeInfoBase( ii );

                if( ( nodeinfo != nullptr ) &&
                    ( nodeinfo->ptrNodeId() != nullptr ) &&
                    ( nodeinfo->ptrNodeId()->valid() ) &&
                    ( nodeinfo->platformId() == platform_id ) )
                {
                    if( ( next_device_id == node_id_t::ILLEGAL_DEVICE_ID ) ||
                        ( next_device_id <= nodeinfo->deviceId() ) )
                    {
                        next_device_id = nodeinfo->deviceId();
                        ++next_device_id;
                    }
                }
            }
        }

        return next_device_id;
    }

    NodeControllerBase::platform_id_t
    NodeControllerBase::doGetPlatformIdByPlatformName(
        char const* SIXTRL_RESTRICT platform_name ) const SIXTRL_NOEXCEPT
    {
        using node_info_base_t = NodeControllerBase::node_info_base_t;
        using platform_id_t = NodeControllerBase::platform_id_t;
        using node_index_t = NodeControllerBase::node_index_t;
        using node_id_t = NodeControllerBase::node_id_t;

        platform_id_t platform_id = node_id_t::ILLEGAL_PLATFORM_ID;

        node_index_t const num_avail_nodes = this->numAvailableNodes();

        if( ( num_avail_nodes > node_index_t{ 0 } ) &&
            ( platform_name != nullptr ) &&
            ( std::strlen( platform_name ) > std::size_t{ 0 } ) )
        {
            node_index_t ii = node_index_t{ 0 };

            for( ; ii < num_avail_nodes ; ++ii )
            {
                node_info_base_t const* nodeinfo = this->ptrNodeInfoBase( ii );

                if( ( nodeinfo != nullptr ) &&
                    ( nodeinfo->ptrNodeId() != nullptr ) &&
                    ( nodeinfo->ptrNodeId()->valid() ) &&
                    ( nodeinfo->hasPlatformName() ) &&
                    ( 0 == nodeinfo->platformName().compare(
                        platform_name ) ) )
                {
                    platform_id = nodeinfo->platformId();
                    break;
                }
            }
        }

        return platform_id;
    }

    void NodeControllerBase::doSetCanDirectlyChangeSelectedNodeFlag(
        bool const can_directly_change_selected_node ) SIXTRL_NOEXCEPT
    {
        this->m_can_directly_change_selected_node =
            can_directly_change_selected_node;
    }

    void NodeControllerBase::doSetCanUnselectNodeFlag(
        bool const can_unselect_node ) SIXTRL_NOEXCEPT
    {
        this->m_can_unselect_node = can_unselect_node;
    }

    void NodeControllerBase::doSetUseAutoSelectFlag(
        bool const use_autoselect ) SIXTRL_NOEXCEPT
    {
        this->m_use_autoselect = use_autoselect;
    }

    /* ===================================================================== */


    void NodeControllerBase::doClearNodeControllerBaseImpl() SIXTRL_NOEXCEPT
    {
        if( this->hasSelectedNode() )
        {
            using _this_t = NodeControllerBase;
            using node_index_t = _this_t::node_index_t;

            node_index_t const selected_node_index = this->selectedNodeIndex();
            SIXTRL_ASSERT( selected_node_index != NodeId::UNDEFINED_INDEX );
            SIXTRL_ASSERT( selected_node_index < this->numAvailableNodes() );

            NodeControllerBase::node_info_base_t* ptr_info_base =
                this->m_available_nodes[ selected_node_index ].get();

            if( ptr_info_base != nullptr )
            {
                SIXTRL_ASSERT( ptr_info_base->isSelectedNode() );
                ptr_info_base->setIsSelectedNode( false );
            }

            this->m_ptr_selected_node_id = nullptr;
            this->m_selected_node_id_str.resize(
                _this_t::NODE_ID_STR_CAPACITY, char{ '\0' } );
        }
    }

    NodeControllerBase::status_t NodeControllerBase::doSelectNodeNodeControllerBaseImpl(
        NodeControllerBase::node_index_t const node_index ) SIXTRL_NOEXCEPT
    {
        NodeControllerBase::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        using _this_t          = NodeControllerBase;
        using size_t           = _this_t::size_type;
        using ptr_node_id_t    = _this_t::node_id_t const*;
        using ptr_node_info_t  = _this_t::node_info_base_t*;

        if( ( !this->hasSelectedNode() ) &&
            ( node_index != st::NODE_UNDEFINED_INDEX ) &&
            ( node_index < this->numAvailableNodes() ) )
        {
            SIXTRL_ASSERT( this->m_ptr_selected_node_id == nullptr );
            SIXTRL_ASSERT( this->selectedNodeIndex() ==
                           NodeId::UNDEFINED_INDEX );

            size_t const max_id_str_len =
                _this_t::NODE_ID_STR_CAPACITY - size_t{ 1 };

            this->m_selected_node_id_str.resize(
                _this_t::NODE_ID_STR_CAPACITY, char{ '\0' } );

            ptr_node_info_t ptr_selected_node_info =
                this->doGetPtrNodeInfoBase( node_index );

            if( ( ptr_selected_node_info != nullptr ) &&
                ( ptr_selected_node_info->ptrNodeId() != nullptr ) )
            {
                ptr_node_id_t ptr_selected_node_id =
                    ptr_selected_node_info->ptrNodeId();

                if( ( ptr_selected_node_id != nullptr ) &&
                    ( st::ARCH_STATUS_SUCCESS ==
                      ptr_selected_node_id->toString(
                        this->m_selected_node_id_str.data(),
                        max_id_str_len ) ) )
                {
                    ptr_selected_node_info->setIsSelectedNode( true );
                    ptr_selected_node_info->setNodeIndex( node_index );

                    SIXTRL_ASSERT( ptr_selected_node_id->hasIndex() );
                    SIXTRL_ASSERT( ptr_selected_node_id->index() ==
                                   node_index );

                    this->m_ptr_selected_node_id = ptr_selected_node_id;
                    status = st::ARCH_STATUS_SUCCESS;
                }
            }
        }

        return status;
    }
}

/* end: sixtracklib/common/control/node_controller_base.cpp */
