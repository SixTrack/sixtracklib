#include "sixtracklib/common/control/controller_on_nodes_base.hpp"

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

namespace SIXTRL_CXX_NAMESPACE
{
    ControllerOnNodesBase::node_index_t
    ControllerOnNodesBase::numAvailableNodes() const SIXTRL_NOEXCEPT
    {
        return this->m_available_nodes.size();
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ControllerOnNodesBase::hasDefaultNode() const SIXTRL_NOEXCEPT
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

    ControllerOnNodesBase::node_id_t const*
    ControllerOnNodesBase::ptrDefaultNodeId() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_default_node_id;
    }

    ControllerOnNodesBase::node_index_t
    ControllerOnNodesBase::defaultNodeIndex() const SIXTRL_NOEXCEPT
    {
        using node_index_t = ControllerOnNodesBase::node_index_t;
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

    ControllerOnNodesBase::node_info_base_t const*
    ControllerOnNodesBase::ptrDefaultNodeInfoBase() const SIXTRL_NOEXCEPT
    {
        using node_index_t = ControllerOnNodesBase::node_index_t;
        using ptr_node_info_t = ControllerOnNodesBase::node_info_base_t const*;

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

    bool ControllerOnNodesBase::isNodeAvailable(
        ControllerOnNodesBase::node_index_t const idx ) const SIXTRL_RESTRICT
    {
        return ( ( idx != NodeId::UNDEFINED_INDEX ) &&
            ( idx <  this->numAvailableNodes() ) &&
            ( this->m_available_nodes[ idx ].get() != nullptr ) );
    }

    bool ControllerOnNodesBase::isNodeAvailable(
        ControllerOnNodesBase::node_id_t const& node_id ) const SIXTRL_NOEXCEPT
    {
        return this->isNodeAvailable( this->doFindAvailableNodesIndex(
            node_id.platformId(), node_id.deviceId() ) );
    }

    bool ControllerOnNodesBase::isNodeAvailable(
        ControllerOnNodesBase::platform_id_t const platform_id,
        ControllerOnNodesBase::device_id_t const dev_id ) const SIXTRL_NOEXCEPT
    {
        return this->isNodeAvailable( this->doFindAvailableNodesIndex(
            platform_id, dev_id ) );
    }

    bool ControllerOnNodesBase::isNodeAvailable(
        char const* node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->isNodeAvailable( this->doFindAvailableNodesIndex(
            node_id_str ) );
    }

    bool ControllerOnNodesBase::isNodeAvailable(
        std::string const& node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->isNodeAvailable( this->doFindAvailableNodesIndex(
            node_id_str.c_str() ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ControllerOnNodesBase::isDefaultNode(
        char const* node_id_str ) const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasDefaultNode() ) &&
            ( this->defaultNodeIndex() == this->doFindAvailableNodesIndex(
                node_id_str ) ) );
    }

    bool ControllerOnNodesBase::isDefaultNode( std::string const&
        SIXTRL_RESTRICT_REF node_id_str ) const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasDefaultNode() ) &&
            ( this->defaultNodeIndex() == this->doFindAvailableNodesIndex(
                node_id_str.c_str() ) ) );
    }

    bool ControllerOnNodesBase::isDefaultNode( ControllerOnNodesBase::node_id_t
        const& node_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasDefaultNode() ) &&
            ( this->defaultNodeIndex() == this->doFindAvailableNodesIndex(
                node_id.platformId(), node_id.deviceId() ) ) );
    }

    bool ControllerOnNodesBase::isDefaultNode(
        ControllerOnNodesBase::platform_id_t const platform_id,
        ControllerOnNodesBase::device_id_t const dev_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasDefaultNode() ) &&
            ( this->defaultNodeIndex() == this->doFindAvailableNodesIndex(
                platform_id, dev_id ) ) );
    }

    bool ControllerOnNodesBase::isDefaultNode(
        ControllerOnNodesBase::node_index_t const idx ) const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasDefaultNode() ) &&
                 ( this->defaultNodeIndex() == idx ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ControllerOnNodesBase::node_id_t const* ControllerOnNodesBase::ptrNodeId(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->ptrNodeId( this->doFindAvailableNodesIndex(
            node_id_str ) );
    }

    ControllerOnNodesBase::node_id_t const*
    ControllerOnNodesBase::ptrNodeId( std::string const&
        SIXTRL_RESTRICT_REF node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->ptrNodeId( this->doFindAvailableNodesIndex(
            node_id_str.c_str() ) );
    }

    ControllerOnNodesBase::node_id_t const* ControllerOnNodesBase::ptrNodeId(
        ControllerOnNodesBase::platform_id_t const platform_id,
        ControllerOnNodesBase::device_id_t const dev_id ) const SIXTRL_NOEXCEPT
    {
        return this->ptrNodeId( this->doFindAvailableNodesIndex(
            platform_id, dev_id ) );
    }

    ControllerOnNodesBase::node_id_t const* ControllerOnNodesBase::ptrNodeId(
        ControllerOnNodesBase::node_index_t const idx ) const SIXTRL_NOEXCEPT
    {
        ControllerOnNodesBase::node_id_t const* ptr_node_id = nullptr;

        if( ( idx < this->numAvailableNodes() ) &&
            ( this->m_available_nodes[ idx ].get() != nullptr ) )
        {
            ptr_node_id = this->m_available_nodes[ idx ]->ptrNodeId();
        }

        return ptr_node_id;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ControllerOnNodesBase::node_info_base_t const*
    ControllerOnNodesBase::ptrNodeInfoBase(
        ControllerOnNodesBase::node_index_t const index ) const SIXTRL_NOEXCEPT
    {
        ControllerOnNodesBase::node_info_base_t const*
            ptr_node_info_base = nullptr;

        if( ( index < this->numAvailableNodes() ) &&
            ( this->m_available_nodes[ index ].get() != nullptr ) )
        {
            ptr_node_info_base = this->m_available_nodes[ index ].get();
        }

        return ptr_node_info_base;
    }

    ControllerOnNodesBase::node_info_base_t const*
    ControllerOnNodesBase::ptrNodeInfoBase(
        ControllerOnNodesBase::platform_id_t const platform_id,
        ControllerOnNodesBase::device_id_t const dev_id ) const SIXTRL_NOEXCEPT
        {
        return this->ptrNodeInfoBase( this->doFindAvailableNodesIndex(
            platform_id, dev_id ) );
        }

    ControllerOnNodesBase::node_info_base_t const*
    ControllerOnNodesBase::ptrNodeInfoBase(
        ControllerOnNodesBase::node_id_t const& node_id ) const SIXTRL_NOEXCEPT
    {
        return this->ptrNodeInfoBase( this->doFindAvailableNodesIndex(
            node_id.platformId(), node_id.deviceId() ) );
    }

    ControllerOnNodesBase::node_info_base_t const*
    ControllerOnNodesBase::ptrNodeInfoBase(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->ptrNodeInfoBase( this->doFindAvailableNodesIndex(
            node_id_str ) );
    }

    ControllerOnNodesBase::node_info_base_t const*
    ControllerOnNodesBase::ptrNodeInfoBase( std::string const&
        SIXTRL_RESTRICT_REF node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->ptrNodeInfoBase( this->doFindAvailableNodesIndex(
            node_id_str.c_str() ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ControllerOnNodesBase::hasSelectedNode() const SIXTRL_NOEXCEPT
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

    ControllerOnNodesBase::node_index_t
    ControllerOnNodesBase::selectedNodeIndex() const SIXTRL_NOEXCEPT
    {
        return ( ( this->m_ptr_selected_node_id != nullptr ) &&
                    ( this->m_ptr_selected_node_id->hasIndex() ) )
            ? this->m_ptr_selected_node_id->index()
            : NodeId::UNDEFINED_INDEX;
    }

    ControllerOnNodesBase::node_id_t const*
    ControllerOnNodesBase::ptrSelectedNodeId() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_selected_node_id;
    }

    ControllerOnNodesBase::node_info_base_t const*
    ControllerOnNodesBase::ptrSelectedNodeInfoBase() const SIXTRL_NOEXCEPT
    {
        return ( this->hasSelectedNode() )
            ? this->m_available_nodes[ this->selectedNodeIndex() ].get()
            : nullptr;
    }

    std::string
    ControllerOnNodesBase::selectedNodeIdStr() const SIXTRL_NOEXCEPT
    {
        return std::string{ this->m_selected_node_id_str.data() };
    }

    char const*
    ControllerOnNodesBase::ptrSelectedNodeIdStr() const SIXTRL_NOEXCEPT
    {
        return ( this->hasSelectedNode() )
            ? this->m_selected_node_id_str.data() : nullptr;
    }

    bool ControllerOnNodesBase::selectedNodeIdStr(
        char* SIXTRL_RESTRICT node_id_str, ControllerOnNodesBase::size_type
            const max_str_length ) const SIXTRL_NOEXCEPT
    {
        bool success = false;

        using size_t = ControllerOnNodesBase::size_type;

        if( this->hasSelectedNode() )
        {
            SIXTRL_ASSERT( this->ptrSelectedNodeId() != nullptr );
            success = this->ptrSelectedNodeId()->toString(
                node_id_str, max_str_length );
        }
        else if( ( node_id_str != nullptr ) &&
                    ( max_str_length > size_t{ 0 } ) )
        {
            std::memset( node_id_str, static_cast< int >( '\0' ),
                         max_str_length );
        }

        return success;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ControllerOnNodesBase::selectNode(
        ControllerOnNodesBase::node_index_t const index )
    {
        return this->doSelectNode( index );
    }

    bool ControllerOnNodesBase::selectNode(
        ControllerOnNodesBase::node_id_t const& node_id )
    {
        return this->doSelectNode( this->doFindAvailableNodesIndex(
            node_id.platformId(), node_id.deviceId() ) );
    }

    bool ControllerOnNodesBase::selectNode(
        ControllerOnNodesBase::platform_id_t const platform_idx,
        ControllerOnNodesBase::device_id_t const device_idx )
    {
        return this->doSelectNode( this->doFindAvailableNodesIndex(
            platform_idx, device_idx ) );
    }

    bool ControllerOnNodesBase::selectNode( char const* node_id_str )
    {
        return this->doSelectNode( this->doFindAvailableNodesIndex(
            node_id_str ) );
    }

    bool ControllerOnNodesBase::selectNode( std::string const& node_id_str )
    {
        return this->doSelectNode( this->doFindAvailableNodesIndex(
            node_id_str.c_str() ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    void ControllerOnNodesBase::printAvailableNodesInfo() const
    {
        this->printAvailableNodesInfo( std::cout );
    }

    void ControllerOnNodesBase::printAvailableNodesInfo(
        std::ostream& SIXTRL_RESTRICT_REF output ) const
    {
        using node_index_t = ControllerOnNodesBase::node_index_t;
        using node_info_base_t = ControllerOnNodesBase::node_info_base_t;
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

    std::string ControllerOnNodesBase::availableNodesInfoToString() const
    {
        using _this_t = ControllerOnNodesBase;

        if( this->numAvailableNodes() > _this_t::node_index_t{ 0 } )
        {
            std::ostringstream a2str;
            this->printAvailableNodesInfo( a2str );
            return a2str.str();
        }

        return std::string{};
    }

    void ControllerOnNodesBase::printAvailableNodesInfo(
        ::FILE* SIXTRL_RESTRICT output ) const
    {
        using _this_t = ControllerOnNodesBase;

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

    void ControllerOnNodesBase::storeAvailableNodesInfoToCString(
        char* SIXTRL_RESTRICT nodes_info_str,
        ControllerOnNodesBase::size_type const nodes_info_str_capacity,
        ControllerOnNodesBase::size_type* SIXTRL_RESTRICT
            ptr_required_max_nodes_info_str_length ) const
    {
        using size_t = ControllerOnNodesBase::size_type;
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

    ControllerOnNodesBase::~ControllerOnNodesBase() SIXTRL_NOEXCEPT
    {

    }

    ControllerOnNodesBase::ControllerOnNodesBase(
        ControllerOnNodesBase::arch_id_t const arch_id,
        const char *const SIXTRL_RESTRICT arch_str,
        const char *const SIXTRL_RESTRICT config_str ) :
        ControllerBase( arch_id, arch_str, config_str ),
        m_available_nodes(),
        m_selected_node_id_str(
            ControllerOnNodesBase::NODE_ID_STR_CAPACITY, char{ '\0' } ),
        m_ptr_default_node_id( nullptr ),
        m_ptr_selected_node_id( nullptr )
    {

    }

    void ControllerOnNodesBase::doClear()
    {
        this->doClearOnNodesBaseImpl();
        this->doClearAvailableNodes();
        ControllerBase::doClear();

        return;
    }

    bool ControllerOnNodesBase::doSelectNode(
        ControllerOnNodesBase::node_index_t const node_index )
    {
        return this->doSelectNodeControllerOnNodesBaseImpl( node_index );
    }

    ControllerOnNodesBase::node_index_t
    ControllerOnNodesBase::doFindAvailableNodesIndex(
        ControllerOnNodesBase::platform_id_t const platform_id,
        ControllerOnNodesBase::device_id_t const dev_id ) const SIXTRL_NOEXCEPT
    {
        using node_index_t = ControllerOnNodesBase::node_index_t;
        using node_info_base_t = ControllerOnNodesBase::node_info_base_t;
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

    ControllerOnNodesBase::node_index_t
    ControllerOnNodesBase::doFindAvailableNodesIndex(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        using node_index_t = ControllerOnNodesBase::node_index_t;
        using node_id_t = ControllerOnNodesBase::node_id_t;

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

    void ControllerOnNodesBase::doClearAvailableNodes() SIXTRL_NOEXCEPT
    {
        this->m_available_nodes.clear();
        return;
    }

    ControllerOnNodesBase::node_index_t
    ControllerOnNodesBase::doAppendAvailableNodeInfoBase(
        ControllerOnNodesBase::ptr_node_info_base_t&& ptr_node_info_base )
    {
        using node_index_t = ControllerOnNodesBase::node_index_t;
        using node_id_t = ControllerOnNodesBase::node_id_t;

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
            }
        }

        return new_index;
    }

    ControllerOnNodesBase::node_info_base_t*
    ControllerOnNodesBase::doGetPtrNodeInfoBase(
        ControllerOnNodesBase::node_index_t const node_index ) SIXTRL_NOEXCEPT
    {
        return const_cast< ControllerOnNodesBase::node_info_base_t* >(
            static_cast< ControllerOnNodesBase const& >(
                *this ).ptrNodeInfoBase( node_index ) );
    }

    void ControllerOnNodesBase::doSetDefaultNodeIndex(
        ControllerOnNodesBase::node_index_t const node_index ) SIXTRL_NOEXCEPT
    {
        using node_info_base_t = ControllerOnNodesBase::node_info_base_t;

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

    ControllerOnNodesBase::platform_id_t
    ControllerOnNodesBase::doGetNextPlatformId() const SIXTRL_NOEXCEPT
    {
        using node_info_base_t = ControllerOnNodesBase::node_info_base_t;
        using platform_id_t = ControllerOnNodesBase::platform_id_t;
        using node_index_t = ControllerOnNodesBase::node_index_t;

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

    ControllerOnNodesBase::device_id_t
    ControllerOnNodesBase::doGetNextDeviceIdForPlatform(
        ControllerOnNodesBase::platform_id_t const platform_id
        ) const SIXTRL_NOEXCEPT
    {
        using node_info_base_t = ControllerOnNodesBase::node_info_base_t;
        using node_index_t = ControllerOnNodesBase::node_index_t;
        using device_id_t = ControllerOnNodesBase::device_id_t;
        using node_id_t = ControllerOnNodesBase::node_id_t;

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

    ControllerOnNodesBase::platform_id_t
    ControllerOnNodesBase::doGetPlatformIdByPlatformName(
        char const* SIXTRL_RESTRICT platform_name ) const SIXTRL_NOEXCEPT
    {
        using node_info_base_t = ControllerOnNodesBase::node_info_base_t;
        using platform_id_t = ControllerOnNodesBase::platform_id_t;
        using node_index_t = ControllerOnNodesBase::node_index_t;
        using node_id_t = ControllerOnNodesBase::node_id_t;

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

    void ControllerOnNodesBase::doClearOnNodesBaseImpl() SIXTRL_NOEXCEPT
    {
        if( this->hasSelectedNode() )
        {
            using _this_t = ControllerOnNodesBase;
            using node_index_t = _this_t::node_index_t;

            node_index_t const selected_node_index = this->selectedNodeIndex();
            SIXTRL_ASSERT( selected_node_index != NodeId::UNDEFINED_INDEX );
            SIXTRL_ASSERT( selected_node_index < this->numAvailableNodes() );

            ControllerOnNodesBase::node_info_base_t* ptr_info_base =
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

    bool ControllerOnNodesBase::doSelectNodeControllerOnNodesBaseImpl(
        ControllerOnNodesBase::node_index_t const node_index ) SIXTRL_NOEXCEPT
    {
        bool success = false;

        using _this_t          = ControllerOnNodesBase;
        using size_t           = _this_t::size_type;
        using ptr_node_id_t    = _this_t::node_id_t const*;
        using ptr_node_info_t  = _this_t::node_info_base_t*;

        if( ( !this->hasSelectedNode() ) &&
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
                    ( ptr_selected_node_id->toString(
                        this->m_selected_node_id_str.data(),
                        max_id_str_len ) ) )
                {
                    ptr_selected_node_info->setIsSelectedNode( true );
                    ptr_selected_node_info->setNodeIndex( node_index );

                    SIXTRL_ASSERT( ptr_selected_node_id->hasIndex() );
                    SIXTRL_ASSERT( ptr_selected_node_id->index() ==
                                   node_index );

                    this->m_ptr_selected_node_id = ptr_selected_node_id;
                    success = true;
                }
            }
        }

        return success;
    }
}

/* end: sixtracklib/common/control/context_base_with_nodes.cpp */
