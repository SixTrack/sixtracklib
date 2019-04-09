#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/context/context_base_with_nodes.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
    #if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
        #include <algorithm>
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <string>
        #include <iostream>
        #include <iomanip>
        #include <vector>
    #endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */
#endif /* defined( __cplusplus ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/context/compute_arch.h"
    #include "sixtracklib/common/context/context_base.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    ContextOnNodesBase::ContextOnNodesBase(
        const char *const SIXTRL_RESTRICT config_str,
        ContextOnNodesBase::type_id_t const type_id ) :
        ContextBase( config_str, type_id ),
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

    ContextOnNodesBase::node_info_t const*
    ContextOnNodesBase::availableNodesInfoBegin() const SIXTRL_NOEXCEPT
    {
        return this->m_available_nodes_id.data();
    }

    ContextOnNodesBase::node_info_t const*
    ContextOnNodesBase::availableNodesInfoEnd() const SIXTRL_NOEXCEPT
    {
        using node_info_ptr_t = ContextNodeBase::node_info_t const*;
        node_info_ptr_t end_ptr = this->availableNodesInfoBegin();

        if( end_ptr != nullptr )
        {
            std::advance( end_ptr, this->numAvailableNodes() );
        }

        return end_ptr;
    }

    ContextOnNodesBase::node_info_t const*
    ContextOnNodesBase::defaultNodeInfo() const SIXTRL_NOEXCEPT
    {
        using node_info_ptr_t = ContextOnNodesBase::node_info_t const*;
        using size_t = ContextOnNodesBase::size_type;

        node_info_ptr_t   ptr_node_info = nullptr;
        size_t const default_node_index = this->doGetDefaultNodeIndex();

        if( default_node_index < this->numAvailableNodes() )
        {
            ptr_node_info = this->availableNodesInfoBegin();

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
        ContextOnNodesBase::node_id_t const node_id ) const SIXTRL_NOEXCEPT
    {
        ContextOnNodesBase::platform_id_t const platform_idx =
            ::NS(ComputeNodeId_get_platform_id)( &node_id );

        ContextOnNodesBase::device_id_t const device_idx =
            ::NS(ComputeNodeId_get_device_id)( &node_id );

        return ( this->findAvailableNodesIndex( platform_idx, device_idx ) <
                 this->numAvailableNodes() );
    }

    bool ContextOnNodesBase::isNodeAvailable(
        ContextOnNodesBase::platform_id_t const platform_idx,
        ContextOnNodesBase::device_id_t const device_idx
        ) const SIXTRL_NOEXCEPT
    {
        return ( this->findAvailableNodesIndex( platform_idx, device_idx ) <
                 this->numAvailableNodes() );
    }

    bool ContextOnNodesBase::isNodeAvailable(
        char const* node_id_str ) const SIXTRL_NOEXCEPT
    {
        return ( this->findAvailableNodesIndex( node_id_str ) <
                 this->numAvailableNodes() );
    }

    bool ContextOnNodesBase::isNodeAvailable(
        std::string const& node_id_str ) const SIXTRL_NOEXCEPT
    {
        return ( this->findAvailableNodesIndex( node_id_str.c_str() ) <
                 this->numAvailableNodes() );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ContextOnNodesBase::isDefaultNode(
        char const* node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->isDefaultNode(
            this->findAvailableNodesIndex( node_id_str ) );
    }

    bool ContextOnNodesBase::isDefaultNode(
        std::string const& SIXTRL_RESTRICT_REF node_id_str ) SIXTRL_NOEXCEPT
    {
        return this->isDefaultNode(
            this->findAvailableNodesIndex( node_id_str.c_str() ) );
    }

    bool ContextOnNodesBase::isDefaultNode(
        ContextOnNodesBase::node_id_t const node_id ) const SIXTRL_NOEXCEPT
    {
        return ( this->isDefaultNode(
            ::NS(ComputeNodeId_get_platform_id)( &node_id ),
            ::NS(ComputeNodeId_get_device_id)( &node_id ) );
    }

    bool ContextOnNodesBase::isDefaultNode(
        ContextOnNodesBase::platform_id_t const platform_idx,
        ContextOnNodesBase::device_id_t const device_idx ) const SIXTRL_NOEXCEPT
    {
        return this->isDefaultNode(
            this->findAvailableNodesIndex( platform_idx, device_idx ) );
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
        return this->ptrAvailableNodesId( this->findAvailableNodesIndex(
            platform_index, device_index ) );
    }

    ContextOnNodesBase::node_id_t const*
    ContextOnNodesBase::ptrAvailableNodesId(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesId( this->findAvailableNodesIndex(
            node_id_str ) );
    }

    ContextOnNodesBase::node_id_t const*
    ContextOnNodesBase::ptrAvailableNodesId( std::string const&
        SIXTRL_RESTRICT_REF node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesId( this->findAvailableNodesIndex(
            node_id_str.c_str() ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ContextOnNodesBase::node_info_t const*
    ContextOnNodesBase::ptrAvailableNodesInfo(
        ContextOnNodesBase::size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ( index < this->numAvailableNodes() )
            ? &this->m_available_nodes_info[ index ] : nullptr;
    }

    ContextOnNodesBase::node_info_t const*
    ContextOnNodesBase::ptrAvailableNodesInfo(
            ContextOnNodesBase::platform_id_t const platform_idx,
            ContextOnNodesBase::device_id_t const device_idx
        ) const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesInfo( this->findAvailableNodesIndex(
            platform_idx, device_idx ) );
    }

    ContextOnNodesBase::node_info_t const*
    ContextOnNodesBase::ptrAvailableNodesInfo(
        ContextOnNodesBase::node_id_t const node_id ) const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesInfo(
            ::NS(ComputeNodeId_get_platform_id)( &node_id ),
            ::NS(ComputeNodeId_get_device_id)( &node_id ) );
    }

    ContextOnNodesBase::node_info_t const*
    ContextOnNodesBase::ptrAvailableNodesInfo(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesInfo(
            this->findAvailableNodesIndex( node_id_str ) );
    }

    ContextOnNodesBase::node_info_t const*
    ContextOnNodesBase::ptrAvailableNodesInfo( std::string const&
        SIXTRL_RESTRICT_REF node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->ptrAvailableNodesInfo( node_id_str.c_str() );
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

    ContextOnNodesBase::node_info_t const*
    ContextOnNodesBase::ptrSelectedNodeInfo() const SIXTRL_NOEXCEPT
    {
        return ( this->hasSelectedNode() )
            ? this->ptrAvailableNodesInfo( this->m_selected_node_index )
            : nullptr;
    }

    std::string const&
    ContextOnNodesBase::selectedNodeIdStr() const SIXTRL_NOEXCEPT
    {
        return this->m_selected_node_id_str;
    }

    char const*
    ContextOnNodesBase::ptrSelectedNodeIdStr() const SIXTRL_NOEXCEPT
    {
        return this->m_selected_node_id_str.c_str();
    }

    bool ContextOnNodesBase::selectedNodeIdStr(
        char* SIXTRL_RESTRICT node_id_str, ContextOnNodesBase::size_type const
            max_str_length ) const SIXTRL_NOEXCEPT
    {
        bool success = false;

        if( ( this->hasSelectedNode() ) &&
            ( node_id_str != nullptr ) && ( max_str_length > size_type{ 0 } ) )
        {
            SIXTRL_ASSERT( 0u < std::strlen( this->m_selected_node_id_str.data() );
            std::strncpy( node_id_str, this->m_selected_node_id_str.data(),
                          max_str_length );

            success = true;
        }

        return success;
    }

    bool ContextOnNodesBase::selectNode(
        ContextOnNodesBase::node_id_t const node_id )
    {
        return this->selectNode(
            ::NS(ComputeNodeId_get_platform_id)( &node_id ),
            ::NS(ComputeNodeId_get_device_id)( &node_id ) );
    }

    bool ContextOnNodesBase::selectNode(
        ContextOnNodesBase::platform_id_t const platform_idx,
        ContextOnNodesBase::device_id_t const device_idx )
    {
        return this->selectNode( this->findAvailableNodesIndex(
            platform_idx, device_idx ) );
    }

    bool ContextOnNodesBase::selectNode( char const* node_id_str )
    {
        return this->selectNode( this->findAvailableNodesIndex(
            node_id_str ) );
    }

    bool ContextOnNodesBase::selectNode( std::string const& node_id_str )
    {
        return this->selectNode( this->findAvailableNodesIndex(
            node_id_str.c_str() ) );
    }

    bool ContextOnNodesBase::selectNode(
        ContextOnNodesBase::size_type const index )
    {
        return this->doSelectNode( index );
    }

    void ContextOnNodesBase::printNodesInfo()
    {
        this->printNodesInfo( std::cout );
    }

    void ContextOnNodesBase::printNodesInfo(
            std::ofstream& SIXTRL_RESTRICT_REF os )
    {
        using ptr_t = ContextOnNodesBase::node_info_t const*;
        ptr_t info_it  = this->availableNodesInfoBegin();
        ptr_t info_end = this->availableNodesInfoEnd();

        if( info_it != nullptr )
        {
            for( ; info_it != info_end ; ++info_it )
            {
                this->doPrintNodesInfo( os, *info_it );
            }
        }
    }

    void ContextOnNodesBase::printNodesInfo(
        ContextOnNodesBase::size_type const index ) const
    {
        this->printNodesInfo( std::cout, index );
    }

    void ContextOnNodesBase::printNodesInfo(
        std::ostream& SIXTRL_RESTRICT_REF os,
        ContextOnNodesBase::size_type const index ) const
    {
        ContextOnNodesBase::node_info_t const* ptr_info =
            this->ptrAvailableNodesInfo( index );

        if( ptr_info != nullptr )
        {
            this->doPrintNodesInfo( os, *ptr_info );
        }
    }

    void ContextOnNodesBase::printNodesInfo(
        ContextOnNodesBase::node_id_t const node_id ) const
    {
        this->printNodesInfo( std::cout,
            ::NS(ComputeNodeId_get_platform_id)( &node_id ),
            ::NS(ComputeNodeId_get_device_id)( &node_id ) );
    }

    void ContextOnNodesBase::printNodesInfo(
        std::ostream& SIXTRL_RESTRICT_REF os,
        ContextOnNodesBase::node_id_t const node_id ) const
    {
        this->printNodesInfo( os,
            ::NS(ComputeNodeId_get_platform_id)( &node_id ),
            ::NS(ComputeNodeId_get_device_id)( &node_id ) );
    }

    void ContextOnNodesBase::printNodesInfo(
        char const* SIXTRL_RESTRICT node_id_str ) const
    {
        this->printNodesInfo( std::cout, node_id_str );
    }

    void ContextOnNodesBase::printNodesInfo(
        std::ostream& SIXTRL_RESTRICT_REF os,
        char const* SIXTRL_RESTRICT node_id_str ) const
    {
        this->printNodesInfo( os,
            this->findAvailableNodesIndex( node_id_str ) );
    }

    void ContextOnNodesBase::printNodesInfo(
        std::string const& SIXTRL_RESTRICT_REF node_id_str ) const
    {
        this->printNodesInfo( std::cout, node_id_str.c_str() );
    }

    void ContextOnNodesBase::printNodesInfo(
        std::ostream& SIXTRL_RESTRICT_REF os,
        std::string const& SIXTRL_RESTRICT_REF node_id_str ) const
    {
        this->printNodesInfo( os, node_id_str.c_str() );
    }

    void ContextOnNodesBase::printSelectedNodesInfo() const
    {
        this->printSelectedNodesInfo( std::cout );
    }

    void ContextOnNodesBase::printSelectedNodesInfo( std::ostream& os ) const
    {
        if( this->hasSelectedNode() )
        {
            this->printNodesInfo( os, this->m_selected_node_index );
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

    void ContextOnNodesBase::doPrintNodesInfo(
        std::ostream& SIXTRL_RESTRICT_REF os,
        ContextOnNodesBase::node_info_t const& SIXTRL_RESTRICT_REF node_info )
    {
        ContextOnNodesBase::node_id_t const node_id =
            ::NS(ComputeNodeInfo_get_id)( &node_info );

        os << "Device ID   : "
            << ::NS(ComputeNodeId_get_platform_id)( &node_id )
            << "."
            << ::NS(ComputeNodeId_get_device_id)( &node_id );

        if( !this->isDefaultNode( node_id ) )
        {
            os << "\r\n"
        }
        else
        {
            os << " [DEFAULT]\r\n";
        }

        if( nullptr != ::NS(ComputeNodeInfo_get_arch)( &node_info ) )
        {
            os << "Architecture: "
               << ::NS(ComputeNodeInfo_get_arch)( &node_info ) << "\r\n";
        }

        if( nullptr != ::NS(ComputeNodeInfo_get_platform)( &node_info ) )
        {
            os << "Platform    : "
               << ::NS(ComputeNodeInfo_get_platform)( &node_info ) << "\r\n";
        }

        if( nullptr != ::NS(ComputeNodeInfo_get_name)( &node_info ) )
        {
            os << "Name        : "
               << ::NS(ComputeNodeInfo_get_name)( &node_info ) << "\r\n"
        }

        if( nullptr != ::NS(ComputeNodeInfo_get_description)( &node_info ) )
        {
            os << "Description : "
               << ::NS(ComputeNodeInfo_get_description)( &node_info )
               << "\r\n";
        }

        return;
    }

    ContextOnNodesBase::size_type
    ContextOnNodesBase::doFindAvailableNodesIndex(
        ContextOnNodesBase::platform_id_t const platform_idx,
        ContextOnNodesBase::device_id_t const device_idx ) const SIXTRL_NOEXCEPT
    {
        using size_t        = ContextOnNodesBase::size_type;
        using platform_id_t = ContextOnNodesBase::platform_id_t;
        using device_id_t   = ContextOnNodesBase::device_id_t;

        size_t index = this->numAvailableNodes();

        if( ( platform_idx >= platform_id_t{ 0 } ) &&
            ( device_idx   >= device_id_t{ 0 } ) )
        {
            index = size_t{ 0 };

            for( auto const& cmp_node_id : this->m_available_nodes_id )
            {
                platform_id_t const cmp_platform_idx =
                    ::NS(ComputeNodeId_get_platform_id)( &cmp_node_id );

                device_id_t const cmp_device_idx =
                    ::NS(ComputeNodeId_get_device_id)( &cmp_node_id );

                if( ( platform_idx == cmp_platform_idx ) &&
                    ( device_idx   == cmp_device_idx   ) )
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

        if( ( index > size_t{ 0 } ) && ( 0 == NS(ComputeNodeId_from_string)(
                &node_id, node_id_str ) ) )
        {
            index = this->doFindAvailableNodesIndex(
                ::NS(ComputeNodeId_get_platform_id)( &node_id ),
                ::NS(ComputeNodeId_get_device_id)( &node_id ) );
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
        ContextOnNodesBase::node_info_t const& SIXTRL_RESTRICT_REF node_info )
    {
        using _this_t     = ContextOnNodesBase;
        using node_id_t   = _this_t::node_id_t;
        using node_info_t = _this_t::node_info_t;
        using size_t      = _this_t::size_type;

        node_id_t new_node_id;
        node_info_t new_node_info;

        size_t const arch_str_len =
            ( ::NS(ComputeNodeInfo_get_arch)( &node_info ) != nullptr )
                ? std::strlen( ::NS(ComputeNodeInfo_get_arch)( &node_info )
                : size_t{ 0 };

        size_t const platform_str_len =
            ( ::NS(ComputeNodeInfo_get_platform)( &node_info ) != nullptr )
                ? std::strlen( ::NS(ComputeNodeInfo_get_platform)( &node_info )
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
        this->m_available_nodes_info.push_back( node_info_t{} );

        node_id_t& new_node_id = this->m_available_nodes_id.back();
        node_info_t& new_node_info = this->m_available_nodes_info.back();

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

                std::strncpy( new_node_info.name, ::NS(ComputeNodeInfo_get_name)(
                    &node_info ), name_str_len );
            }

            if( descr_str_len > size_t{ 0 } )
            {
                SIXTRL_ASSERT( ::NS(ComputeNodeInfo_get_description)(
                    &new_node_info ) != nullptr );

                SIXTRL_ASSERT( ::NS(ComputeNodeInfo_get_description)(
                    &node_info ) != nullptr );

                std::strncpy( new_node_info.description,
                              ::NS(ComputeNodeInfo_get_description)( &node_info ),
                              desrc_str_len );
            }
        }

        return new_node_index;
    }


    void ContextOnNodesBase::doClearOnNodesBaseImpl() SIXTRL_NOEXECPT
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

            if( NS(ComputeNodeId_to_string)(
                &this->m_available_nodes_info[ node_index ],
                this->m_selected_node_id_str.data(),
                this->m_selected_node_id_str.size() ) == 0 )
            {
                success = true;
            }
        }

        return success;
    }

    /* --------------------------------------------------------------------- */

    std::ostream& operator<<( std::ostream& SIXTRL_RESTRICT ostream,
        ContextOnNodesBase const& SIXTRL_RESTRICT_REF context )
    {
        context.printNodesInfo( ofstream );
        return ofstream;
    }
}

/* end: sixtracklib/common/context/context_base_with_nodes.cpp */
