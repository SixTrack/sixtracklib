#include "sixtracklib/common/context/context_base_with_nodes.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/context/compute_arch.h"
#include "sixtracklib/common/context/context_base.h"

namespace SIXTRL_CXX_NAMESPACE
{
    ContextOnNodesBase::size_type
    ContextOnNodesBase::numAvailableNodes() const SIXTRL_NOEXCEPT
    {
        return this->m_available_nodes.size();
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ContextOnNodesBase::hasDefaultNode() const SIXTRL_NOEXCEPT
    {
         bool const has_default_node = (
             ( this->m_ptr_default_node_id != nullptr ) &&
             ( this->m_ptr_default_node_id->hasIndex() ) &&
             ( this->m_ptr_default_node_id->index() <
               this->numAvailableNodes() ) );

         SIXTRL_ASSERT( ( !has_default_node ) ||
             ( this->m_ptr_default_node_id != nullptr ) &&
             ( this->numAvailableNodes() >
               this->m_ptr_default_node_id->index() ) &&
             ( this->m_available_nodes[
                 this->m_ptr_default_node_id->index() ].get() != nullptr ) &&
             ( this->m_available_nodes[
                 this->m_ptr_default_node_id->index() ].get()->ptrNodeId() ==
                 this->m_ptr_default_node_id ) );
    }

    ContextOnNodesBase::node_id_t const*
    ContextOnNodesBase::ptrDefaultNodeId() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_default_node_id;
    }

    ContextOnNodesBase::size_type
    ContextOnNodesBase::defaultNodeIndex() const SIXTRL_NOEXCEPT
    {
        using size_t = ContextOnNodesBase::size_type;

        size_t const default_node_id =
            SIXTRL_CXX_NAMESPACE::NODE_UNDEFINED_INDEX;

        if( ( this->ptrDefaultNodeId() != nullptr ) &&
            ( this->ptrDefaultNodeId()->hasIndex() ) &&
            ( this->ptrDefaultNodeId()->index() <
              this->numAvailableNodes() ) )
        {
            default_node_id = this->ptrDefaultNodeId()->index();
        }

        return default_node_id;
    }

    ContextOnNodesBase::node_info_base_t const*
    ContextOnNodesBase::defaultNodeInfoBase() const SIXTRL_NOEXCEPT
    {
        using size_t = ContextOnNodesBase::size_type;
        using ptr_node_info_t = ContextOnNodesBase::node_info_base_t const*;

        size_t const default_node_idx = this->defaultNodeIndex();
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

    bool ContextOnNodesBase::isNodeAvailable(
        ContextOnNodesBase::size_type const node_index ) const SIXTRL_RESTRICT
    {
        return (
            ( node_index != SIXTRL_CXX_NAMESPACE::NODE_UNDEFINED_INDEX ) &&
            ( node_index <  this->numAvailableNodes() ) &&
            ( this->m_available_nodes[ node_index ].get() != nullptr ) );

    }

        bool ContextOnNodesBase::isNodeAvailable(
            node_id_t const& node_id ) const SIXTRL_NOEXCEPT
        {
            return this->isNodeAvailable( this->doFindAvailableNodesIndex(
                node_id.platformId(), node_id.deviceId() ) );
        }

        bool ContextOnNodesBase::isNodeAvailable(
            ContextOnNodesBase::platform_id_t const platform_index,
            ContextOnNodesBase::device_id_t const device_index
            ) const SIXTRL_NOEXCEPT
        {
            return this->isNodeAvailable( this->doFindAvailableNodesIndex(
                platform_index, device_index ) );
        }

        bool ContextOnNodesBase::isNodeAvailable(
            char const* node_id_str ) const SIXTRL_NOEXCEPT
        {
            return this->isNodeAvailable( this->doFindAvailableNodesIndex(
                node_id_str ) );
        }

        bool ContextOnNodesBase::isNodeAvailable(
            std::string const& node_id_str ) const SIXTRL_NOEXCEPT
        {
            return this->isNodeAvailable( this->doFindAvailableNodesIndex(
                node_id_str.c_str() ) );
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        bool ContextOnNodesBase::isDefaultNode(
            char const* node_id_str ) const SIXTRL_NOEXCEPT
        {
            return ( ( this->hasDefaultNode() ) &&
                ( this->defaultNodeIndex() == this->doFindAvailableNodesIndex(
                    node_id_str ) ) );
        }

        bool ContextOnNodesBase::isDefaultNode( std::string const&
            SIXTRL_RESTRICT_REF node_id_str ) const SIXTRL_NOEXCEPT
        {
            return ( ( this->hasDefaultNode() ) &&
                ( this->defaultNodeIndex() == this->doFindAvailableNodesIndex(
                    node_id_str.c_str() ) ) );
        }

        bool ContextOnNodesBase::isDefaultNode( ContextOnNodesBase::node_id_t
            const& node_id ) const SIXTRL_NOEXCEPT
        {
            return ( ( this->hasDefaultNode() ) &&
                ( this->defaultNodeIndex() == this->doFindAvailableNodesIndex(
                    node_id.platformId(), node_id.deviceId() ) ) );
        }

        bool ContextOnNodesBase::isDefaultNode(
            ContextOnNodesBase::platform_id_t const platform_index,
            ContextOnNodesBase::device_id_t const device_index
            ) const SIXTRL_NOEXCEPT
        {
            return ( ( this->hasDefaultNode() ) &&
                ( this->defaultNodeIndex() == this->doFindAvailableNodesIndex(
                    platform_index, device_index ) ) );
        }

        bool ContextOnNodesBase::isDefaultNode( ContextOnNodesBase::size_type
            const node_index ) const SIXTRL_NOEXCEPT
        {
            return ( ( this->hasDefaultNode() ) &&
                     ( this->defaultNodeIndex() == node_index ) );
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        ContextOnNodesBase::node_id_t const* ContextOnNodesBase::ptrNodeId(
            char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
        {
            return this->ptrNodeId( this->doFindAvailableNodesIndex(
                node_id_str ) );
        }

        ContextOnNodesBase::node_id_t const*
        ContextOnNodesBase::ptrNodeId( std::string const&
            SIXTRL_RESTRICT_REF node_id_str ) const SIXTRL_NOEXCEPT
        {
            return this->ptrNodeId( this->doFindAvailableNodesIndex(
                node_id_str.c_str() ) );
        }

        ContextOnNodesBase::node_id_t const* ContextOnNodesBase::ptrNodeId(
            ContextOnNodesBase::platform_id_t const platform_idx,
            ContextOnNodesBase::device_id_t const device_idx
        ) const SIXTRL_NOEXCEPT
        {
            return this->ptrNodeId( this->doFindAvailableNodesIndex(
                platform_idx, device_idx ) );
        }

        ContextOnNodesBase::node_id_t const* ContextOnNodesBase::ptrNodeId(
            ContextOnNodesBase::size_type const index ) const SIXTRL_NOEXCEPT
        {
            ContextOnNodesBase::node_id_t const* ptr_node_id = nullptr;

            if( ( index < this->numAvailableNodes() ) &&
                ( this->m_available_nodes[ index ].get() != nullptr ) )
            {
                ptr_node_id = this->m_available_nodes[
                    index ].get()->ptrNodeId();
            }

            return ptr_node_id;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        ContextOnNodesBase::node_info_base_t const*
        ContextOnNodesBase::ptrNodesInfoBase(
            ContextOnNodesBase::size_type const index ) const SIXTRL_NOEXCEPT
        {
            ContextOnNodesBase::node_info_base_t const*
                ptr_node_info_base = nullptr;

            if( ( index < this->numAvailableNodes() ) &&
                ( this->m_available_nodes.get() != nullptr ) )
            {
                ptr_node_info_base = this->m_available_nodes[ index ].get();
            }

            return ptr_node_info_base;
        }

        ContextOnNodesBase::node_info_base_t const*
        ContextOnNodesBase::ptrNodesInfoBase(
            ContextOnNodesBase::platform_id_t const platform_idx,
            ContextOnNodesBase::device_id_t const device_idx
            ) const SIXTRL_NOEXCEPT
         {
            return this->ptrNodesInfoBase( this->doFindAvailableNodesIndex(
                platform_idx, device_idx ) );
         }

        ContextOnNodesBase::node_info_base_t const*
        ContextOnNodesBase::ptrNodesInfoBase(
            ContextOnNodesBase::node_id_t const& nodeid ) const SIXTRL_NOEXCEPT
        {
            return this->ptrNodesInfoBase( this->doFindAvailableNodesIndex(
                node_id.platformId(), node_id.deviceId() ) );
        }

        ContextOnNodesBase::node_info_base_t const*
        ContextOnNodesBase::ptrNodesInfoBase(
            char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
        {
            return this->ptrNodesInfoBase( this->doFindAvailableNodesIndex(
                node_id_str ) );
        }

        ContextOnNodesBase::node_info_base_t const*
        ContextOnNodesBase::ptrNodesInfoBase( std::string const&
            SIXTRL_RESTRICT_REF node_id_str ) const SIXTRL_NOEXCEPT
        {
            return this->ptrNodesInfoBase( this->doFindAvailableNodesIndex(
                node_id_str.c_str() ) );
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        bool ContextOnNodesBase::hasSelectedNode() const SIXTRL_NOEXCEPT
        {
            using size_t = ContextOnNodesBase::size_type;

            bool const has_selected_node = (
                ( this->m_ptr_selected_noded_id != nullptr ) &&
                ( this->numAvailableNodes() > this->selectedNodeIndex() ) );

            SIXTRL_ASSERT( ( !has_selected_node ) ||
                ( this->m_ptr_selected_noded_id != nullptr ) &&
                ( this->selectedNodeIndex() !=
                    SIXTRL_CXX_NAMESPACE::NODE_UNDEFINED_INDEX ) &&
                ( this->m_ptr_selected_noded_id ==
                    this->ptrNodeId( this->selectedNodeIndex() ) ) );

            return has_selected_node;
        }

        ContextOnNodesBase::size_type
        ContextOnNodesBase::selectedNodeIndex() const SIXTRL_NOEXCEPT
        {
            return this->m_selected_node_index;
        }

        ContextOnNodesBase::node_id_t const*
        ContextOnNodesBase::ptrSelectedNodeId() const SIXTRL_NOEXCEPT
        {
            return this->m_ptr_selected_noded_id;
        }

        ContextOnNodesBase::node_info_t const*
        ContextOnNodesBase::ptrSelectedNodeInfoBase() const SIXTRL_NOEXCEPT
        {
            return ( this->hasSelectedNode() )
                ? this->m_available_nodes[ this->selectedNodeIndex() ].get()
                : nullptr;
        }

        std::string ContextOnNodesBase::selectedNodeIdStr()
            const SIXTRL_NOEXCEPT
        {
            return std::string{ this->m_selected_node_id_str.data() };
        }

        char const* ContextOnNodesBase::ptrSelectedNodeIdStr()
            const SIXTRL_NOEXCEPT
        {
            return ( this->hasSelectedNode() )
                ? this->m_selected_node_id_str.data() : nullptr;
        }

        bool ContextOnNodesBase::selectedNodeIdStr(
            char* SIXTRL_RESTRICT node_id_str,
            ContextOnNodesBase::size_type const max_str_length ) const SIXTRL_NOEXCEPT
        {
            bool success = false;

            using size_t = ContextOnNodesBase::size_type;

            if( this->hasSelectedNode() )
            {
                SIXTRL_ASSERT( this->ptrSelectedNodeId() != nullptr );
                success = this->ptrSelectedNodeId()->toString(
                    node_id_str, max_str_length );
            }
            else if( ( node_id_str != nullptr ) &&
                     ( max_str_length > size_t{ 0 } ) )
            {
                std::memset( node_id_str, max_str_length, ( int )'\0' );
            }

            return success;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        bool selectNode( node_id_t const& node_id );
        bool selectNode( platform_id_t const platform_idx,
                         device_id_t const device_idx );

        bool selectNode( char const* node_id_str );
        bool selectNode( std::string const& node_id_str );
        bool selectNode( size_type const index );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        void printAvailableNodesInfo() const;

        void printAvailableNodesInfo(
            std::ostream& SIXTRL_RESTRICT_REF os ) const;

        void printAvailableNodesInfo(
            ::FILE* SIXTRL_RESTRICT output ) const;

        std::string availableNodesInfoToString() const;

        virtual ~ContextOnNodesBase() SIXTRL_NOEXCEPT;

        protected:

        using std::unique_ptr< node_info_base_t > ptr_node_info_base_t;

        ContextOnNodesBase(
            arch_id_t const arch_id, const char *const arch_str,
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        ContextOnNodesBase(
            ContextOnNodesBase const& other ) = default;

        ContextOnNodesBase(
            ContextOnNodesBase&& other ) = default;

        ContextOnNodesBase& operator=(
            ContextOnNodesBase const& rhs ) = default;

        ContextOnNodesBase& operator=(
            ContextOnNodesBase&& rhs ) = default;

        virtual void doClear() override;
        virtual bool doSelectNode( size_type node_index );

        virtual size_type doGetDefaultNodeIndex() const;

        size_type doFindAvailableNodesIndex(
            platform_id_t const platform_idx,
            device_id_t const device_idx ) const SIXTRL_NOEXCEPT;

        size_type doFindAvailableNodesIndex(
            char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT;

        void doClearAvailableNodes() SIXTRL_NOEXCEPT;

        size_type doAppendAvailableNodeInfoBase(
            ptr_node_info_base_t&& SIXTRL_RESTRICT_REF ptr_node_info_base );

        private:

        void doClearOnNodesBaseImpl() SIXTRL_NOEXCEPT;

        size_type
        doGetDefaultNodeIndexOnNodesBaseImpl() const SIXTRL_NOEXCEPT;

        bool doSelectNodeOnNodesBaseImpl(
            size_type const node_index ) SIXTRL_NOEXCEPT;











    ContextOnNodesBase::ContextOnNodesBase(
        ContextOnNodesBase::type_id_t const type_id,
        const char *const SIXTRL_RESTRICT type_id_str,
        const char *const SIXTRL_RESTRICT config_str ) :
        ContextBase( type_id, type_id_str, config_str ),
        m_available_nodes_id(),
        m_available_nodes_info(),
        m_selected_node_id_str( 32u, char{ '\0' } ),
        m_selected_node_index( int64_t{ -1 } )
    {
        this->doSetUsesNodesFlag( true );
    }

    ContextOnNodesBase::size_type
    ContextOnNodesBase::numAvailableNodes() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_available_nodes_id.size() ==
                       this->m_available_nodes_info.size() );

        return this->m_available_nodes_id.size();
    }

    ContextOnNodesBase::node_info_base_t const*
    ContextOnNodesBase::availableNodesInfoBaseBegin() const SIXTRL_NOEXCEPT
    {
        return this->m_available_nodes_info.data();
    }

    ContextOnNodesBase::node_info_base_t const*
    ContextOnNodesBase::availableNodesInfoBaseEnd() const SIXTRL_NOEXCEPT
    {
        using node_info_ptr_t = ContextOnNodesBase::node_info_base_t const*;
        node_info_ptr_t end_ptr = this->availableNodesInfoBaseBegin();

        if( end_ptr != nullptr )
        {
            std::advance( end_ptr, this->numAvailableNodes() );
        }

        return end_ptr;
    }

    ContextOnNodesBase::node_info_base_t const*
    ContextOnNodesBase::defaultNodeInfoBase() const SIXTRL_NOEXCEPT
    {
        using node_info_ptr_t = ContextOnNodesBase::node_info_base_t const*;
        using size_t = ContextOnNodesBase::size_type;

        node_info_ptr_t   ptr_node_info = nullptr;
        size_t const default_node_index = this->doGetDefaultNodeIndex();

        if( default_node_index < this->numAvailableNodes() )
        {
            ptr_node_info = this->availableNodesInfoBaseBegin();

            if( ptr_node_info != nullptr )
            {
                std::advance( ptr_node_info, default_node_index );
            }
        }

        return ptr_node_info;
    }

    ContextOnNodesBase::node_id_t
    ContextOnNodesBase::defaultNodeId() const SIXTRL_NOEXCEPT
    {
        using node_id_t = ContextOnNodesBase::node_id_t;
        using size_t    = ContextOnNodesBase::size_type;

        node_id_t default_node_id;
        size_t const default_node_index = this->doGetDefaultNodeIndex();

        ::NS(ComputeNodeId_preset)( &default_node_id );

        if( default_node_index < this->numAvailableNodes() )
        {
            default_node_id = this->m_available_nodes_id[ default_node_index ];
        }

        return default_node_id;
    }

    bool ContextOnNodesBase::isNodeAvailable(
        ContextOnNodesBase::size_type const node_index ) const SIXTRL_RESTRICT
    {
        return ( node_index < this->numAvailableNodes() );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ContextOnNodesBase::isNodeAvailable(
        ContextOnNodesBase::node_id_t const& node_id ) const SIXTRL_NOEXCEPT
    {
        return ( this->doFindAvailableNodesIndex( node_id.platformId(),
            node_id.deviceId() ) < this->numAvailableNodes() );
    }

    bool ContextOnNodesBase::isNodeAvailable(
        ContextOnNodesBase::platform_id_t const platform_idx,
        ContextOnNodesBase::device_id_t const device_idx
        ) const SIXTRL_NOEXCEPT
    {
        return ( this->doFindAvailableNodesIndex( platform_idx, device_idx ) <
                 this->numAvailableNodes() );
    }

    bool ContextOnNodesBase::isNodeAvailable(
        char const* node_id_str ) const SIXTRL_NOEXCEPT
    {
        return ( this->doFindAvailableNodesIndex( node_id_str ) <
                 this->numAvailableNodes() );
    }

    bool ContextOnNodesBase::isNodeAvailable(
        std::string const& node_id_str ) const SIXTRL_NOEXCEPT
    {
        return ( this->doFindAvailableNodesIndex( node_id_str.c_str() ) <
                 this->numAvailableNodes() );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ContextOnNodesBase::isDefaultNode(
        char const* node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->isDefaultNode(
            this->doFindAvailableNodesIndex( node_id_str ) );
    }

    bool ContextOnNodesBase::isDefaultNode( std::string const&
        SIXTRL_RESTRICT_REF node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->isDefaultNode(
            this->doFindAvailableNodesIndex( node_id_str.c_str() ) );
    }

    bool ContextOnNodesBase::isDefaultNode(
        ContextOnNodesBase::node_id_t const& node_id ) const SIXTRL_NOEXCEPT
    {
        return this->isDefaultNode( node_id.platformId(), node_id.deviceId() );
    }

    bool ContextOnNodesBase::isDefaultNode(
        ContextOnNodesBase::platform_id_t const platform_idx,
        ContextOnNodesBase::device_id_t const device_idx ) const SIXTRL_NOEXCEPT
    {
        return this->isDefaultNode(
            this->doFindAvailableNodesIndex( platform_idx, device_idx ) );
    }

    bool ContextOnNodesBase::isDefaultNode(
        ContextOnNodesBase::size_type const node_index ) const SIXTRL_NOEXCEPT
    {
        return ( ( this->doGetDefaultNodeIndex() == node_index ) &&
                 ( node_index < this->numAvailableNodes() ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ContextOnNodesBase::node_id_t const*
    ContextOnNodesBase::ptrAvailableNodesId(
        ContextOnNodesBase::size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ( index < this->numAvailableNodes() )
            ? &this->m_available_nodes_id[ index ] : nullptr;
    }

    ContextOnNodesBase::node_id_t const*
    ContextOnNodesBase::ptrAvailableNodesId(
            ContextOnNodesBase::platform_id_t const platform_index,
            ContextOnNodesBase::device_id_t const device_index
        ) const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesId( this->doFindAvailableNodesIndex(
            platform_index, device_index ) );
    }

    ContextOnNodesBase::node_id_t const*
    ContextOnNodesBase::ptrAvailableNodesId(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesId( this->doFindAvailableNodesIndex(
            node_id_str ) );
    }

    ContextOnNodesBase::node_id_t const*
    ContextOnNodesBase::ptrAvailableNodesId( std::string const&
        SIXTRL_RESTRICT_REF node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesId( this->doFindAvailableNodesIndex(
            node_id_str.c_str() ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ContextOnNodesBase::node_info_base_t const*
    ContextOnNodesBase::ptrAvailableNodesInfoBase(
        ContextOnNodesBase::size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ( index < this->numAvailableNodes() )
            ? &this->m_available_nodes_info[ index ].get() : nullptr;
    }

    ContextOnNodesBase::node_info_base_t const*
    ContextOnNodesBase::ptrAvailableNodesInfoBase(
            ContextOnNodesBase::platform_id_t const platform_idx,
            ContextOnNodesBase::device_id_t const device_idx
        ) const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesInfoBase(
            this->doFindAvailableNodesIndex( platform_idx, device_idx ) );
    }

    ContextOnNodesBase::node_info_base_t const*
    ContextOnNodesBase::ptrAvailableNodesInfoBase(
        ContextOnNodesBase::node_id_t const& node_id ) const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesInfoBase(
            node_id.platformId(), node_id.deviceId() );
    }

    ContextOnNodesBase::node_info_base_t const*
    ContextOnNodesBase::ptrAvailableNodesInfoBase(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesInfoBase(
            this->doFindAvailableNodesIndex( node_id_str ) );
    }

    ContextOnNodesBase::node_info_base_t const*
    ContextOnNodesBase::ptrAvailableNodesInfoBase( std::string const&
        SIXTRL_RESTRICT_REF node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesInfoBase( node_id_str.c_str() );
    }

    bool ContextOnNodesBase::hasSelectedNode() const SIXTRL_NOEXCEPT
    {
        return (
            ( this->m_selected_node_index >= int64_t{ 0 } ) &&
            ( static_cast< int64_t >( this->numAvailableNodes() ) >
                this->m_selected_node_index ) );
    }

    ContextOnNodesBase::node_id_t const*
    ContextOnNodesBase::ptrSelectedNodeId() const SIXTRL_NOEXCEPT
    {
        return ( this->hasSelectedNode() )
            ? this->ptrAvailableNodesId( this->m_selected_node_index )
            : nullptr;
    }

    ContextOnNodesBase::node_info_base_t const*
    ContextOnNodesBase::ptrSelectedNodeInfoBase() const SIXTRL_NOEXCEPT
    {
        return ( this->hasSelectedNode() )
            ? this->ptrAvailableNodesInfoBase(
                this->m_selected_node_index ) : nullptr;
    }

    std::string ContextOnNodesBase::selectedNodeIdStr() const SIXTRL_NOEXCEPT
    {
        return std::string{ this->m_selected_node_id_str.data() };
    }

    char const* ContextOnNodesBase::ptrSelectedNodeIdStr() const SIXTRL_NOEXCEPT
    {
        return this->m_selected_node_id_str.data();
    }

    bool ContextOnNodesBase::selectedNodeIdStr(
        char* SIXTRL_RESTRICT node_id_str,
        ContextOnNodesBase::size_type const max_len ) const SIXTRL_NOEXCEPT
    {
        bool success = false;
        using size_t = ContextOnNodesBase::size_type;

        if( ( this->hasSelectedNode() ) &&
            ( node_id_str != nullptr ) && ( max_len > size_t{ 0 } ) )
        {
            char const* selected_node_id_str =
                this->m_selected_node_id_str.data();

            if( ( selected_node_id_str != nullptr ) &&
                ( size_t{ 0 } < std::strlen( selected_node_id_str ) ) )
            {
                std::strncpy( node_id_str, selected_node_id_str, max_len );
                success = true;
            }
        }

        return success;
    }

    bool ContextOnNodesBase::selectNode(
        ContextOnNodesBase::node_id_t& const node_id )
    {
        return this->selectNode( this->doFindAvailableNodesIndex(
            node_id.platformId(), node_id.deviceId() ) );
    }

    bool ContextOnNodesBase::selectNode(
        ContextOnNodesBase::platform_id_t const platform_idx,
        ContextOnNodesBase::device_id_t const device_idx )
    {
        return this->selectNode( this->doFindAvailableNodesIndex(
            platform_idx, device_idx ) );
    }

    bool ContextOnNodesBase::selectNode( char const* node_id_str )
    {
        return this->selectNode( this->doFindAvailableNodesIndex(
            node_id_str ) );
    }

    bool ContextOnNodesBase::selectNode( std::string const& node_id_str )
    {
        return this->selectNode( this->doFindAvailableNodesIndex(
            node_id_str.c_str() ) );
    }

    bool ContextOnNodesBase::selectNode(
        ContextOnNodesBase::size_type const index )
    {
        return this->doSelectNode( index );
    }

    void ContextOnNodesBase::printAvailableNodesInfo() const
    {
        this->printAvailableNodesInfo( std::cout );
    }

    void ContextOnNodesBase::printAvailableNodesInfo(
            std::ostream& SIXTRL_RESTRICT_REF os ) const
    {
        using _this_t = SIXTRL_CXX_NAMESPACE::ContextOnNodesBase;
        using size_t  = _this_t::size_type;
        using ptr_t   = _this_t::node_info_base_t const*;

        size_t const num_avail_nodes = this->numAvailableNodes();

        for( size_t ii = size_t{ 0 } ; ii < num_avail_nodes ; ++ii )
        {
            ptr_t ptr_node_info = this->ptrAvailableNodesInfoBase( ii );

            if( ptr_node_info != nullptr )
            {
                os << *ptr_node_info << "\r\n";
            }
        }
    }

    std::string ContextOnNodesBase::availableNodesInfoToString() const
    {
        std::ostringstream a2str;
        this->printAvailableNodesInfo( a2str );
        return a2str.str();
    }

    void ContextOnNodesBase::printAvailableNodesInfo(
        ::FILE* SIXTRL_RESTRICT output ) const
    {
        if( output != nullptr )
        {
            std::string const avail_nodes_str =
                this->availableNodesInfoToString();

            if( !avail_nodes_str.empty() )
            {
                std::fprintf( output, "%s", avail_nodes_str.c_str() );
            }
        }
    }

    ContextOnNodesBase::~ContextOnNodesBase() SIXTRL_NOEXCEPT
    {

    }

    void ContextOnNodesBase::doClear()
    {
        this->doClearOnNodesBaseImpl();
        ContextBase::doClear();

        return;
    }

    bool ContextOnNodesBase::doSelectNode(
        ContextOnNodesBase::size_type node_index )
    {
        return this->doSelectNodeOnNodesBaseImpl( node_index );
    }

    ContextOnNodesBase::size_type
    ContextOnNodesBase::doGetDefaultNodeIndex() const
    {
        return this->doGetDefaultNodeIndexOnNodesBaseImpl();
    }

    ContextOnNodesBase::size_type
    ContextOnNodesBase::doFindAvailableNodesIndex(
        ContextOnNodesBase::platform_id_t const platform_idx,
        ContextOnNodesBase::device_id_t const device_idx ) const SIXTRL_NOEXCEPT
    {
        using size_t = ContextOnNodesBase::size_type;
        size_t index = this->numAvailableNodes();

        if( ( platform_idx !=
                SIXTRL_CXX_NAMESPACE::NODE_ILLEGAL_PLATFORM_ID ) &&
            ( device_idx   != SIXTRL_CXX_NAMESPACE::NODE_ILLEGAL_DEVICE_ID ) )
        {
            index = size_t{ 0 };

            for( auto const& cmp_node_id : this->m_available_nodes_id )
            {
                if( ( platform_idx == cmp_node_id.platformId() ) &&
                    ( device_idx == cmp_node_id.deviceId() ) )
                {
                    break;
                }

                ++index;
            }
        }

        SIXTRL_ASSERT( index <= this->numAvailableNodes() );
        return index;
    }

    ContextOnNodesBase::size_type
    ContextOnNodesBase::doFindAvailableNodesIndex(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        using size_t = ContextOnNodesBase::size_type;

        size_t index = this->numAvailableNodes();
        ContextOnNodesBase::node_id_t node_id;

        if( ( index > size_t{ 0 } ) && ( node_id.fromString( node_id_str ) ) )
        {
            index = this->doFindAvailableNodesIndex(
                node_id.platformId(), node_id.deviceId() );
        }

        return index;
    }

    void ContextOnNodesBase::doClearAvailableNodes() SIXTRL_NOEXCEPT
    {
        std::fill( this->m_selected_node_id_str.begin(),
                   this->m_selected_node_id_str.end(), char{ '\0' } );

        this->m_available_nodes_id.clear();
        this->m_available_nodes_info.clear();
    }

    ContextOnNodesBase::size_type ContextOnNodesBase::doAppendAvailableNode(
        ContextOnNodesBase::node_id_t const& SIXTRL_RESTRICT_REF node_id,
        ContextOnNodesBase::node_info_base_t const& SIXTRL_RESTRICT_REF node_info )
    {
        using _this_t     = ContextOnNodesBase;
        using node_id_t   = _this_t::node_id_t;
        using node_info_base_t = _this_t::node_info_base_t;
        using size_t      = _this_t::size_type;

        size_t const arch_str_len =
            ( ::NS(ComputeNodeInfo_get_arch)( &node_info ) != nullptr )
                ? std::strlen( ::NS(ComputeNodeInfo_get_arch)( &node_info ) )
                : size_t{ 0 };

        size_t const platform_str_len =
            ( ::NS(ComputeNodeInfo_get_platform)( &node_info ) != nullptr )
                ? std::strlen( ::NS(ComputeNodeInfo_get_platform)( &node_info ) )
                : size_t{ 0 };

        size_t const name_str_len =
            ( ::NS(ComputeNodeInfo_get_name)( &node_info ) != nullptr )
                ? std::strlen( ::NS(ComputeNodeInfo_get_name)( &node_info ) )
                : size_t{ 0 };

        size_t const descr_str_len =
            ( ::NS(ComputeNodeInfo_get_description)( &node_info ) != nullptr )
                ? std::strlen( ::NS(ComputeNodeInfo_get_description)(
                    &node_info ) ) : size_t{ 0 };

        size_t new_node_index = this->numAvailableNodes();

        this->m_available_nodes_id.push_back( node_id_t{} );
        this->m_available_nodes_info.push_back( node_info_base_t{} );

        node_id_t& new_node_id = this->m_available_nodes_id.back();
        node_info_base_t& new_node_info = this->m_available_nodes_info.back();

        ::NS(ComputeNodeId_preset)( &new_node_id );
        ::NS(ComputeNodeInfo_preset)( &new_node_info );

        if( nullptr != ::NS(ComputeNodeInfo_reserve)( &new_node_info,
            arch_str_len, platform_str_len, name_str_len, descr_str_len ) )
        {
            new_node_id      = node_id;
            new_node_info.id = node_id;

            if( arch_str_len > size_t{ 0 } )
            {
                SIXTRL_ASSERT( ::NS(ComputeNodeInfo_get_arch)(
                    &new_node_info ) != nullptr );

                SIXTRL_ASSERT( ::NS(ComputeNodeInfo_get_arch)(
                    &node_info ) != nullptr );

                std::strncpy( new_node_info.arch,
                    NS(ComputeNodeInfo_get_arch)( &node_info ), arch_str_len );
            }

            if( platform_str_len > size_t{ 0 } )
            {
                SIXTRL_ASSERT( ::NS(ComputeNodeInfo_get_platform)(
                    &new_node_info ) != nullptr );

                SIXTRL_ASSERT( ::NS(ComputeNodeInfo_get_platform)(
                    &node_info ) != nullptr );

                std::strncpy( new_node_info.platform,
                    ::NS(ComputeNodeInfo_get_platform)( &node_info ),
                    platform_str_len );
            }

            if( name_str_len > size_t{ 0 } )
            {
                SIXTRL_ASSERT( ::NS(ComputeNodeInfo_get_name)(
                    &new_node_info ) != nullptr );

                SIXTRL_ASSERT( ::NS(ComputeNodeInfo_get_name)(
                    &node_info ) != nullptr );

                std::strncpy( new_node_info.name,
                    ::NS(ComputeNodeInfo_get_name)( &node_info ),
                    name_str_len );
            }

            if( descr_str_len > size_t{ 0 } )
            {
                SIXTRL_ASSERT( ::NS(ComputeNodeInfo_get_description)(
                    &new_node_info ) != nullptr );

                SIXTRL_ASSERT( ::NS(ComputeNodeInfo_get_description)(
                    &node_info ) != nullptr );

                std::strncpy( new_node_info.description,
                    ::NS(ComputeNodeInfo_get_description)( &node_info ),
                    descr_str_len );
            }
        }

        return new_node_index;
    }


    void ContextOnNodesBase::doClearOnNodesBaseImpl() SIXTRL_NOEXCEPT
    {
        this->m_selected_node_id_str.clear();
        this->m_selected_node_index = int64_t{ -1 };
        this->doClearAvailableNodes();

        return;
    }

    ContextOnNodesBase::size_type
    ContextOnNodesBase::doGetDefaultNodeIndexOnNodesBaseImpl(
        ) const SIXTRL_NOEXCEPT
    {
        return ContextOnNodesBase::size_type{ 0 };
    }

    bool ContextOnNodesBase::doSelectNodeOnNodesBaseImpl(
        ContextOnNodesBase::size_type const node_index ) SIXTRL_NOEXCEPT
    {
        bool success = false;

        if( ( !this->hasSelectedNode() ) &&
            ( node_index < this->numAvailableNodes() ) )
        {
            SIXTRL_ASSERT( this->m_selected_node_index < int64_t{ 0 } );
            this->m_selected_node_index = static_cast< int64_t >( node_index );

            std::fill( this->m_selected_node_id_str.begin(),
                       this->m_selected_node_id_str.end(), char{ '\0' } );

            if( ::NS(ComputeNodeId_to_string)(
                &this->m_available_nodes_id[ node_index ],
                this->m_selected_node_id_str.data(),
                this->m_selected_node_id_str.size() ) == 0 )
            {
                success = true;
            }
        }

        return success;
    }

    /* --------------------------------------------------------------------- */

    std::ostream& operator<<( std::ostream& SIXTRL_RESTRICT os,
        ContextOnNodesBase const& SIXTRL_RESTRICT_REF context )
    {
        context.printNodesInfo( os );
        return os;
    }
}

/* end: sixtracklib/common/context/context_base_with_nodes.cpp */
