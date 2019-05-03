#include "sixtracklib/common/control/node_id.h"

#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <sstream>
#include <ostream>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"

namespace SIXTRL_CXX_NAMESPACE
{
    NodeId::NodeId(
        NodeId::platform_id_t const platform_id,
        NodeId::device_id_t const device_id,
        NodeId::index_t const node_index ) SIXTRL_NOEXCEPT :
        m_platform_id( platform_id ),
        m_device_id( device_id ),
        m_node_index( node_index )
    {

    }

    NodeId::NodeId( std::string const& SIXTRL_RESTRICT_REF id_str ) :
        m_platform_id( NodeId::ILLEGAL_PLATFORM_ID ),
        m_device_id( NodeId::ILLEGAL_DEVICE_ID ),
        m_node_index( NodeId::UNDEFINED_INDEX )
    {
        this->fromString( id_str );
    }

    NodeId::NodeId( const char *const SIXTRL_RESTRICT id_str ) :
        m_platform_id( NodeId::ILLEGAL_PLATFORM_ID ),
        m_device_id( NodeId::ILLEGAL_DEVICE_ID ),
        m_node_index( NodeId::UNDEFINED_INDEX)
    {
        this->fromString( id_str );
    }

    bool NodeId::valid() const SIXTRL_NOEXCEPT
    {
        return ( ( this->m_platform_id != NodeId::ILLEGAL_PLATFORM_ID ) &&
                 ( this->m_device_id != NodeId::ILLEGAL_DEVICE_ID ) );
    }

    NodeId::platform_id_t NodeId::platformId() const SIXTRL_NOEXCEPT
    {
        return this->m_platform_id;
    }

    NodeId::device_id_t NodeId::deviceId() const SIXTRL_NOEXCEPT
    {
        return this->m_device_id;
    }

    bool NodeId::hasIndex() const SIXTRL_NOEXCEPT
    {
        return ( this->m_node_index != NodeId::UNDEFINED_INDEX);
    }

    NodeId::index_t NodeId::index() const SIXTRL_NOEXCEPT
    {
        return this->m_node_index;
    }

    void NodeId::setPlatformId(
        NodeId::platform_id_t const id ) SIXTRL_NOEXCEPT
    {
        this->m_platform_id = id;
    }

    void NodeId::setDeviceId(
        NodeId::device_id_t const id ) SIXTRL_NOEXCEPT
    {
        this->m_device_id = id;
    }

    void NodeId::setIndex( NodeId::index_t const index ) SIXTRL_NOEXCEPT
    {
        this->m_node_index = index;
    }

    std::string NodeId::toString() const
    {
        std::ostringstream a2str;

        if( this->valid() )
        {
            a2str << *this;
        }

        return a2str.str();
    }

    bool NodeId::toString( char* SIXTRL_RESTRICT node_id_str,
        NodeId::size_type const node_id_str_capacity ) const SIXTRL_NOEXCEPT
    {
        bool success = false;

        if( ( node_id_str != nullptr ) &&
            ( node_id_str_capacity > NodeId::size_type{ 0 } ) )
        {
            std::memset( node_id_str, static_cast< int >( '\0' ),
                         node_id_str_capacity );
        }

        if( this->valid() )
        {
            std::ostringstream a2str;
            a2str << *this;

            std::string const str( a2str.str() );

            if( str.size() < node_id_str_capacity )
            {
                std::copy( str.begin(), str.end(), node_id_str );
                success = true;
            }
        }

        return success;
    }

    bool NodeId::fromString(
        std::string const& SIXTRL_RESTRICT_REF id_str ) SIXTRL_NOEXCEPT
    {
        return ( !id_str.empty() )
            ? this->fromString( id_str.c_str() ) : false;
    }

    bool NodeId::fromString(
        const char *const SIXTRL_RESTRICT id_str ) SIXTRL_NOEXCEPT
    {
        bool success = false;

        if( ( id_str != nullptr ) &&
            ( std::strlen( id_str ) > NodeId::size_type{ 0 } ) )
        {
            long long int temp_platform_id = NodeId::ILLEGAL_PLATFORM_ID;
            long long int temp_device_id   = NodeId::ILLEGAL_DEVICE_ID;

            int const ret = std::sscanf( id_str, "%lld.%lld",
                &temp_platform_id, &temp_device_id );

            if( ( ret == int{ 2 }  ) &&
                ( temp_platform_id != NodeId::ILLEGAL_PLATFORM_ID ) &&
                ( temp_device_id   != NodeId::ILLEGAL_DEVICE_ID ) )
            {
                this->setPlatformId( temp_platform_id );
                this->setDeviceId( temp_device_id );
                success = true;
            }
        }

        return success;
    }

    bool NodeId::operator<(
        NodeId const& SIXTRL_RESTRICT_REF rhs ) const SIXTRL_NOEXCEPT
    {
        return ( ( this->m_platform_id < rhs.m_platform_id ) ||
                 ( ( this->m_platform_id == rhs.m_platform_id ) &&
                   ( this->m_device_id < rhs.m_device_id ) ) );
    }

    void NodeId::clear() SIXTRL_NOEXCEPT
    {
        this->m_platform_id = NodeId::ILLEGAL_PLATFORM_ID;
        this->m_device_id   = NodeId::ILLEGAL_DEVICE_ID;
        this->m_node_index  = NodeId::UNDEFINED_INDEX;
    }

    void NodeId::reset(
        NodeId::platform_id_t const platform_id,
        NodeId::device_id_t const device_id,
        NodeId::index_t const node_index ) SIXTRL_NOEXCEPT
    {
        this->m_platform_id = platform_id;
        this->m_device_id = device_id;
        this->m_node_index = node_index;
    }

    std::ostream& operator<<( std::ostream& SIXTRL_RESTRICT_REF output,
        SIXTRL_CXX_NAMESPACE::NodeId const& SIXTRL_RESTRICT_REF node_id )
    {
        output << node_id.platformId() << "." << node_id.deviceId();
        return output;
    }

    int compareNodeIds(
        SIXTRL_CXX_NAMESPACE::NodeId const& SIXTRL_RESTRICT_REF lhs,
        SIXTRL_CXX_NAMESPACE::NodeId const& SIXTRL_RESTRICT_REF rhs )
    {
        return ( lhs < rhs ) ? -1 : ( rhs < lhs ) ? +1 : 0;
    }

    void printNodeId( ::FILE* SIXTRL_RESTRICT fp,
        SIXTRL_CXX_NAMESPACE::NodeId const& SIXTRL_RESTRICT_REF node_id )
    {
        if( fp != nullptr )
        {
            std::ostringstream a2str;
            a2str << node_id;
            std::string const str( a2str.str() );
            int const ret = fprintf( fp, "%s", str.c_str() );

            SIXTRL_ASSERT( ret >= 0 );
            ( void )ret;
        }

        return;
    }
}

/* end: sixtracklib/common/control/node_id.cpp */
