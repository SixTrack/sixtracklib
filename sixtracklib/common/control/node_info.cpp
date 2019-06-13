#include "sixtracklib/common/control/node_info.hpp"

#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <ostream>
#include <sstream>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/node_id.hpp"
#include "sixtracklib/common/control/arch_info.hpp"

namespace st = SIXTRL_CXX_NAMESPACE;

namespace SIXTRL_CXX_NAMESPACE
{
    NodeInfoBase::NodeInfoBase(
        NodeInfoBase::arch_id_t const arch_id,
        const char *const SIXTRL_RESTRICT arch_str,
        NodeInfoBase::platform_id_t const platform_id,
        NodeInfoBase::device_id_t const device_id,
        const char *const SIXTRL_RESTRICT platform_name,
        const char *const SIXTRL_RESTRICT device_name,
        const char *const SIXTRL_RESTRICT description,
        bool const is_default_node, bool const is_selected_node ) :
        st::ArchInfo( arch_id, arch_str ), m_platform_name(),
        m_device_name(), m_description(), m_node_id( platform_id, device_id ),
        m_is_default_node( is_default_node ),
        m_is_selected_node( is_selected_node )
    {
        this->setPlatformName( platform_name );
        this->setDeviceName( device_name );
        this->setDescription( description );
    }

    NodeInfoBase::NodeInfoBase(
        NodeInfoBase::arch_info_t const* SIXTRL_RESTRICT ptr_arch_info,
        NodeInfoBase::node_id_t const* SIXTRL_RESTRICT ptr_node_id,
        const char *const SIXTRL_RESTRICT platform_name,
        const char *const SIXTRL_RESTRICT device_name,
        const char *const SIXTRL_RESTRICT description,
        bool const is_default_node, bool const is_selected_node ) :
        st::ArchInfo( ),
        m_platform_name(), m_device_name(), m_description(), m_node_id(),
        m_is_default_node( is_default_node ),
        m_is_selected_node( is_selected_node )
    {
        if( ptr_arch_info != nullptr )
        {
            ArchInfo::operator=( *ptr_arch_info );
        }

        if( ptr_node_id != nullptr )
        {
            this->nodeId() = *ptr_node_id;
        }

        this->setPlatformName( platform_name );
        this->setDeviceName( device_name );
        this->setDescription( description );
    }

    NodeInfoBase::NodeInfoBase(
        NodeInfoBase::arch_info_t const& SIXTRL_RESTRICT_REF arch_info,
        NodeInfoBase::node_id_t const& SIXTRL_RESTRICT_REF node_id,
        std::string const& SIXTRL_RESTRICT_REF platform_name,
        std::string const& SIXTRL_RESTRICT_REF device_name,
        std::string const& SIXTRL_RESTRICT_REF description,
        bool const is_default_node, bool const is_selected_node ) :
        st::ArchInfo( arch_info ), m_platform_name(),
        m_device_name(), m_description(), m_node_id( node_id ),
        m_is_default_node( is_default_node ),
        m_is_selected_node( is_selected_node )
    {
        this->setPlatformName( platform_name );
        this->setDeviceName( device_name );
        this->setDescription( description );
    }

    NodeInfoBase::NodeInfoBase(
        NodeInfoBase::arch_id_t const arch_id,
        std::string const& SIXTRL_RESTRICT_REF arch_str,
        NodeInfoBase::platform_id_t const platform_id,
        NodeInfoBase::device_id_t const device_id,
        std::string const& SIXTRL_RESTRICT_REF platform_name,
        std::string const& SIXTRL_RESTRICT_REF device_name,
        std::string const& SIXTRL_RESTRICT_REF description,
        bool const is_default_node, bool const is_selected_node ):
        st::ArchInfo( arch_id, arch_str ), m_platform_name(),
        m_device_name(), m_description(), m_node_id( platform_id, device_id ),
        m_is_default_node( is_default_node ),
        m_is_selected_node( is_selected_node )
    {
        this->setPlatformName( platform_name );
        this->setDeviceName( device_name );
        this->setDescription( description );
    }

    /* --------------------------------------------------------------------- */

    NodeInfoBase::node_id_t const&
    NodeInfoBase::nodeId() const SIXTRL_NOEXCEPT
    {
        return this->m_node_id;
    }

    NodeInfoBase::node_id_t& NodeInfoBase::nodeId() SIXTRL_NOEXCEPT
    {
        return this->m_node_id;
    }

    NodeInfoBase::node_id_t const*
    NodeInfoBase::ptrNodeId() const SIXTRL_NOEXCEPT
    {
        return &this->m_node_id;
    }

    NodeInfoBase::node_id_t* NodeInfoBase::ptrNodeId() SIXTRL_NOEXCEPT
    {
        return &this->m_node_id;
    }

    NodeInfoBase::platform_id_t NodeInfoBase::platformId() const SIXTRL_NOEXCEPT
    {
        return this->m_node_id.platformId();
    }

    void NodeInfoBase::setPlatformId(
        NodeInfoBase::platform_id_t const platform_id ) SIXTRL_NOEXCEPT
    {
        this->m_node_id.setPlatformId( platform_id );
    }

    NodeInfoBase::device_id_t
    NodeInfoBase::deviceId() const SIXTRL_NOEXCEPT
    {
        return this->m_node_id.deviceId();
    }

    void NodeInfoBase::setDeviceId(
        NodeInfoBase::device_id_t const device_id ) SIXTRL_NOEXCEPT
    {
        this->m_node_id.setDeviceId( device_id );
    }

    bool NodeInfoBase::hasNodeIndex() const SIXTRL_NOEXCEPT
    {
        return this->m_node_id.hasIndex();
    }

    NodeInfoBase::node_index_t NodeInfoBase::nodeIndex() const SIXTRL_NOEXCEPT
    {
        return this->m_node_id.index();
    }

    void NodeInfoBase::setNodeIndex(
        NodeInfoBase::node_index_t const node_index ) SIXTRL_NOEXCEPT
    {
        this->m_node_id.setIndex( node_index );
    }

    bool NodeInfoBase::isDefaultNode() const SIXTRL_NOEXCEPT
    {
        return this->m_is_default_node;
    }

    void NodeInfoBase::setIsDefaultNode(
        bool const is_default_node ) SIXTRL_NOEXCEPT
    {
        this->m_is_default_node = is_default_node;
    }

    bool NodeInfoBase::isSelectedNode() const SIXTRL_NOEXCEPT
    {
        return this->m_is_selected_node;
    }

    void NodeInfoBase::setIsSelectedNode(
        bool const is_selected_node ) SIXTRL_NOEXCEPT
    {
        this->m_is_selected_node = is_selected_node;
    }

    /* --------------------------------------------------------------------- */

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

    void NodeInfoBase::setPlatformName(
        std::string const& SIXTRL_RESTRICT_REF platform_name )
    {
        this->m_platform_name = platform_name;
    }

    void NodeInfoBase::setPlatformName(
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

    void NodeInfoBase::setDeviceName(
        std::string const& SIXTRL_RESTRICT_REF device_name )
    {
        this->m_device_name = device_name;
    }

    void NodeInfoBase::setDeviceName(
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

    void NodeInfoBase::setDescription(
        std::string const& SIXTRL_RESTRICT_REF description )
    {
        this->m_description = description;
    }

    void NodeInfoBase::setDescription(
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

    /* --------------------------------------------------------------------- */

    std::ostream& operator<<( std::ostream& SIXTRL_RESTRICT_REF output,
        NodeInfoBase const& SIXTRL_RESTRICT_REF node_info )
    {
        node_info.doPrintToOutputStream( output );
        return output;
    }

    void NodeInfoBase::print( ::FILE* SIXTRL_RESTRICT output ) const
    {
        if( output != nullptr )
        {
            std::ostringstream a2str;
            this->doPrintToOutputStream( a2str );
            std::string const str( a2str.str() );
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
    
    
    NodeInfoBase::size_type 
    NodeInfoBase::requiredOutStringLength() const
    {
        std::ostringstream a2str;
        this->doPrintToOutputStream( a2str );
        return a2str.str().size(); /* TODO: Find a more efficient method! */
    }
        
    std::string NodeInfoBase::toString() const
    {
        std::ostringstream a2str;
        this->doPrintToOutputStream( a2str );
        
        return a2str.str();
    }
    
    NodeInfoBase::status_t NodeInfoBase::toString(
        NodeInfoBase::size_type const out_str_capacity, 
        char* SIXTRL_RESTRICT out_str ) const
    {
        using size_t = NodeInfoBase::size_type;
        
        NodeInfoBase::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        
        if( ( out_str_capacity > size_t{ 0 } ) && ( out_str != nullptr ) )
        {
            std::memset( out_str, ( int )'\0', out_str_capacity );
            
            std::ostringstream a2str;
            this->doPrintToOutputStream( a2str );
            std::string const temp_str = a2str.str();
            
            if( !temp_str.empty() )
            {
                std::strncpy( out_str, temp_str.c_str(), 
                              out_str_capacity - size_t{ 1 } );
            
                if( out_str_capacity >= temp_str.size() )
                {
                    status = st::ARCH_STATUS_SUCCESS;
                }
            }            
        }
        
        return status;
    }

    /* --------------------------------------------------------------------- */

    void NodeInfoBase::doPrintToOutputStream(
        std::ostream& SIXTRL_RESTRICT_REF output ) const
    {
        output << "Node         : " << this->m_node_id;

        if( this->isSelectedNode() )
        {
            output << " [SELECTED]";
        }

        if( this->isDefaultNode() )
        {
            output << " [DEFAULT]";
        }

        output << "\r\nArchitecture : id = " << this->archId();

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
}

/* end: sixtracklib/common/control/node_info.cpp */
