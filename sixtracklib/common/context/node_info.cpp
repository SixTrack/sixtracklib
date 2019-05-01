#include "sixtracklib/common/context/node_info.hpp"

#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ostream>
#include <sstream>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/context/definitions.h"
#include "sixtracklib/common/context/node_id.hpp"
#include "sixtracklib/common/context/arch_info.hpp"

namespace SIXTRL_CXX_NAMESPACE
{
    NodeInfoBase::NodeInfoBase(
        NodeInfoBase::arch_id_t const arch_id,
        const char *const SIXTRL_RESTRICT arch_str,
        NodeInfoBase::platform_id_t const platform_id,
        NodeInfoBase::device_id_t const device_id,
        const char *const SIXTRL_RESTRICT platform_name,
        const char *const SIXTRL_RESTRICT device_name,
        const char *const SIXTRL_RESTRICT description ) :
        SIXTRL_CXX_NAMESPACE::ArchInfo( arch_id, arch_str ), m_platform_name(),
        m_device_name(), m_description(), m_node_id( platform_id, device_id )
    {
        this->doSetPlatformName( platform_name );
        this->doSetDeviceName( device_name );
        this->doSetDescription( description );
    }

    NodeInfoBase::NodeInfoBase(
        NodeInfoBase::arch_info_t const* SIXTRL_RESTRICT ptr_arch_info,
        NodeInfoBase::node_id_t const* SIXTRL_RESTRICT ptr_node_id,
        const char *const SIXTRL_RESTRICT platform_name,
        const char *const SIXTRL_RESTRICT device_name,
        const char *const SIXTRL_RESTRICT description ) :
        SIXTRL_CXX_NAMESPACE::ArchInfo(),
        m_platform_name(), m_device_name(), m_description(), m_node_id()
    {
        if( ptr_arch_info != nullptr )
        {
            ArchInfo::operator=( *ptr_arch_info );
        }

        if( ptr_node_id != nullptr )
        {
            this>doSetNodeId( *ptr_node_id );
        }

        this->doSetPlatformName( platform_name );
        this->doSetDeviceName( device_name );
        this->doSetDescription( description );
    }

    NodeInfoBase::NodeInfoBase(
        NodeInfoBase::arch_info_t const& SIXTRL_RESTRICT_REF arch_info,
        NodeInfoBase::node_id_t const& SIXTRL_RESTRICT_REF node_id,
        std::string const& SIXTRL_RESTRICT_REF platform_name,
        std::string const& SIXTRL_RESTRICT_REF device_name,
        std::string const& SIXTRL_RESTRICT_REF description ) :
        SIXTRL_CXX_NAMESPACE::ArchInfo( arch_info ), m_platform_name(),
        m_device_name(), m_description(), m_node_id( node_id )
    {
        this->doSetPlatformName( platform_name.c_str() );
        this->doSetDeviceName( device_name.c_str() );
        this->doSetDescription( description.c_str() );
    }

    NodeInfoBase::NodeInfoBase(
        NodeInfoBase::arch_id_t const arch_id,
        std::string const& SIXTRL_RESTRICT_REF arch_str,
        NodeInfoBase::platform_id_t const platform_id,
        NodeInfoBase::device_id_t const device_id,
        std::string const& SIXTRL_RESTRICT_REF platform_name,
        std::string const& SIXTRL_RESTRICT_REF device_name,
        std::string const& SIXTRL_RESTRICT_REF description ):
        SIXTRL_CXX_NAMESPACE::ArchInfo( arch_id, arch_str ), m_platform_name(),
        m_device_name(), m_description(), m_node_id( platform_id, device_id )
    {
        this->doSetPlatformName( platform_name.c_str() );
        this->doSetDeviceName( device_name.c_str() );
        this->doSetDescription( description.c_str() );
    }

    /* ----------------------------------------------------------------- */

    NodeInfoBase::node_id_t const&
    NodeInfoBase::nodeId() const SIXTRL_NOEXCEPT
    {
        return this->m_node_id;
    }

    NodeInfoBase::node_id_t const*
    NodeInfoBase::ptrNodeId() const SIXTRL_NOEXCEPT
    {
        return &this->m_node_id;
    }

    NodeInfoBase::node_platform_id_t
    NodeInfoBase::platformId() const SIXTRL_NOEXCEPT
    {
        return this->m_node_id.platformId();
    }

    NodeInfoBase::node_device_id_t
    NodeInfoBase::deviceId() const SIXTRL_NOEXCEPT
    {
        return this->m_node_id.deviceId();
    }

    bool NodeInfoBase::isDefaultNode() const SIXTRL_NOEXCEPT
    {
        return this->m_is_default_node;
    }

    /* ----------------------------------------------------------------- */

    bool NodeInfoBase::hasPlatformName() const SIXTRL_NOEXCEPT
    {
        return !this->m_platform_name.empty();
    }

    std::string const& NodeInfoBase::platformName() const SIXTRL_NOEXCEPT
    {
        return this->m_platform_name;
    }

    char const* NodeInfoBase::ptrPlatformNameStr() const SIXTRL_NOEXCEPT
    {
        return this->m_platform_name.c_str();
    }

    bool NodeInfoBase::hasDeviceName() const SIXTRL_NOEXCEPT
    {
        return !this->m_device_name.empty();
    }

    std::string const& NodeInfoBase::deviceName() const SIXTRL_NOEXCEPT
    {
        return this->m_device_name;
    }

    char const* NodeInfoBase::ptrDeviceNameStr() const SIXTRL_NOEXCEPT
    {
        return this->m_device_name.c_str();
    }

    bool NodeInfoBase::hasDescription() const SIXTRL_NOEXCEPT
    {
        return !this->m_description.empty();
    }

    std::string const& NodeInfoBase::description() const SIXTRL_NOEXCEPT
    {
        return this->m_description;
    }

    char const* NodeInfoBase::ptrDescriptionStr() const SIXTRL_NOEXCEPT
    {
        return this->m_description.c_str();
    }

    /* ----------------------------------------------------------------- */

    void NodeInfoBase::reset( NodeInfoBase::arch_id_t const arch_id,
        const char *const SIXTRL_RESTRICT arch_str,
        NodeInfoBase::platform_id_t const platform_id,
        NodeInfoBase::device_id_t const device_id,
        const char *const SIXTRL_RESTRICT platform_name,
        const char *const SIXTRL_RESTRICT device_name,
        const char *const SIXTRL_RESTRICT description )
    {
        this->reset( arch_id, arch_str );
        this->m_node_id.reset( platform_id, device_id );

        this->doSetPlatformName( platform_name );
        this->doSetDeviceName( device_name );
        this->doSetDescription( description );
    }

    void NodeInfoBase::reset( NodeInfoBase::arch_id_t const arch_id,
        std::string const& SIXTRL_RESTRICT_REF arch_str,
        NodeInfoBase::platform_id_t const platform_id,
        NodeInfoBase::device_id_t const device_id,
        std::string const& SIXTRL_RESTRICT_REF platform_name,
        std::string const& SIXTRL_RESTRICT_REF device_name,
        std::string const& SIXTRL_RESTRICT_REF description )
    {
        this->reset( arch_id, arch_str );
        this->m_node_id.reset( platform_id, device_id );

        this->doSetPlatformName( platform_name.c_str() );
        this->doSetDeviceName( device_name.c_str() );
        this->doSetDescription( description.c_str() );
    }

    /* ----------------------------------------------------------------- */

    std::ostream& operator<<( std::ostream& SIXTRL_RESTRICT_REF output,
        NodeInfoBase const& SIXTRL_RESTRICT_REF node_info )
    {
        this->doPrintToOutputStream( output );
        return output;
    }

    void NodeInfoBase::print( ::FILE* SIXTRL_RESTRICT output ) const
    {
        if( output != nullptr )
        {
            std::ostringstream a2str;
            this->doPrintToOutputStream( a2str );
            std:string const str( a2str.str() );
            const int ret = std::fprintf( output, "%s", str.c_str() );
            SIXTRL_ASSERT( ret >= 0 );
            ( void )ret;
        }

        return;
    }

    void NodeInfoBase::printOut() const
    {
        this->print( ::stdout );
    }

    /* ----------------------------------------------------------------- */

    void NodeInfoBase::markAsDefault(
        bool const is_default ) SIXTRL_NOEXCEPT
    {
        this->m_is_default_node = is_default;
    }

    /* ----------------------------------------------------------------- */

    void NodeInfoBase::doPrintToOutputStream(
        std::ostream& SIXTRL_RESTRICT_REF output )
    {
        output << "Node         : " << this->m_node_id;

        if( this->isDefaultNode() )
        {
            output << " [DEFAULT]";
        }

        output << "\r\n";
               << "Architecture : id = " << output->archId();

        if( this->hasArchStr() )
        {
            output << " " << this->archStr();
        }

        output << "\r\n";

        if( this->hasPlatformName() )
        {
            output << "Platform     : " << this->platformName() << "\r\n";
        }

        if( this->hasDeviceName() )
        {
            output << "Device       : " << this->deviceName() << "\r\n";
        }

        if( this->hasDescription() )
        {
            output << "Description : " << this->description() << "\r\n";
        }

        return;
    }

    void NodeInfoBase::doSetNodeId(
        NodeInfoBase::node_id_t const& SIXTRL_RESTRICT_REF node_id )
    {
        this->m_node_id = node_id;
    }

    void NodeInfoBase::doSetPlatformName(
        const char *const SIXTRL_RESTRICT platform_name )
    {
        if( ( platform_name != nullptr ) &&
            ( std::strlen( platform_name ) > NodeInfoBase::size_type{ 0 } ) )
        {
            this->m_platform_name = platform_name;
        }
        else
        {
            this->m_platform_name.clear();
        }
    }

    void NodeInfoBase::doSetDeviceName(
        const char *const SIXTRL_RESTRICT device_name )
    {
        if( ( device_name != nullptr ) &&
            ( std::strlen( device_name ) > NodeInfoBase::size_type{ 0 } ) )
        {
            this->m_device_name = device_name;
        }
        else
        {
            this->m_device_name.clear();
        }
    }

    void NodeInfoBase::doSetDescription(
        const char *const SIXTRL_RESTRICT description )
    {
        if( ( description != nullptr ) &&
            ( std::strlen( description ) > NodeInfoBase::size_type{ 0 } ) )
        {
            this->m_description = description;
        }
        else
        {
            this->m_description.clear();
        }
    }
}

/* end: sixtracklib/common/context/node_info.cpp */
