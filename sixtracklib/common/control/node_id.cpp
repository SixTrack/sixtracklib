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

namespace st = SIXTRL_CXX_NAMESPACE;

namespace SIXTRL_CXX_NAMESPACE
{
    using _this_t = st::NodeId;

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

    NodeId::status_t NodeId::toString( char* SIXTRL_RESTRICT node_id_str,
        NodeId::size_type const node_id_str_capacity ) const SIXTRL_NOEXCEPT
    {
        NodeId::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

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
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    NodeId::status_t NodeId::to_string( char* SIXTRL_RESTRICT node_id_str,
        _this_t::size_type const node_id_str_capacity,
        _this_t::arch_id_t const arch_id,
        _this_t::str_format_t const format ) const SIXTRL_NOEXCEPT
    {
        _this_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( node_id_str != nullptr ) &&
            ( node_id_str_capacity > _this_t::size_type{ 0 } ) &&
            ( format != st::NODE_ID_STR_FORMAT_ILLEGAL ) &&
            ( this->valid() ) )
        {
            int ret = int{ 0 };
            _this_t::size_type const nn =
                node_id_str_capacity - _this_t::size_type{ 1 };

            if( format == st::NODE_ID_STR_FORMAT_NOARCH )
            {
                ret = std::snprintf( node_id_str, nn, "%d.%d",
                    static_cast< int >( this->platformId() ),
                    static_cast< int >( this->deviceId() ) );
            }
            else if( ( arch_id != st::ARCHITECTURE_ILLEGAL ) &&
                     ( arch_id != st::ARCHITECTURE_NONE ) )
            {
                if( format == st::NODE_ID_STR_FORMAT_ARCHID )
                {
                    ret = std::snprintf( node_id_str, nn, "%u:%d.%d",
                        static_cast< unsigned >( arch_id ),
                        static_cast< int >( this->platformId() ),
                        static_cast< int >( this->deviceId() ) );

                }
                else if( format == st::NODE_ID_STR_FORMAT_ARCHSTR )
                {
                    char TEMP_ARCH_NAME[ 32 ] =
                    {
                        '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
                        '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
                        '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
                        '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
                    };

                    if( arch_id == st::ARCHITECTURE_CPU )
                    {
                        strncpy( TEMP_ARCH_NAME,
                                 SIXTRL_ARCHITECTURE_CPU_STR, 31 );
                    }
                    else if( arch_id == st::ARCHITECTURE_OPENCL )
                    {
                        strncpy( TEMP_ARCH_NAME,
                                 SIXTRL_ARCHITECTURE_OPENCL_STR, 31 );
                    }
                    else if( arch_id == st::ARCHITECTURE_CUDA )
                    {
                        strncpy( TEMP_ARCH_NAME,
                                 SIXTRL_ARCHITECTURE_CUDA_STR, 31 );
                    }

                    if( std::strlen( TEMP_ARCH_NAME ) > 0u )
                    {
                        ret = std::snprintf( node_id_str, nn, "%s:%d.%d",
                            TEMP_ARCH_NAME,
                            static_cast< int >( this->platformId() ),
                            static_cast< int >( this->deviceId() ) );
                    }
                }
            }

            if( ( ret > 0 ) && ( ret <= static_cast< int >( nn ) ) )
            {
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    NodeId::status_t NodeId::fromString(
        std::string const& SIXTRL_RESTRICT_REF id_str ) SIXTRL_NOEXCEPT
    {
        return ( !id_str.empty() ) ? this->fromString( id_str.c_str() )
            : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    NodeId::status_t NodeId::fromString(
        const char *const SIXTRL_RESTRICT id_str ) SIXTRL_NOEXCEPT
    {
        NodeId::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

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
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
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
        st::NodeId const& SIXTRL_RESTRICT_REF node_id )
    {
        output << node_id.platformId() << "." << node_id.deviceId();
        return output;
    }

    int compareNodeIds( st::NodeId const& SIXTRL_RESTRICT_REF lhs,
                        st::NodeId const& SIXTRL_RESTRICT_REF rhs )
    {
        return ( lhs < rhs ) ? -1 : ( rhs < lhs ) ? +1 : 0;
    }

    void printNodeId( ::FILE* SIXTRL_RESTRICT fp,
        st::NodeId const& SIXTRL_RESTRICT_REF node_id )
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
