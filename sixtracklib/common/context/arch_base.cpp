#include "sixtracklib/common/context/arch_base.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <regex>
#include <string>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/context/definitions.h"
#include "sixtracklib/common/context/arch_info.hpp"

namespace SIXTRL_CXX_NAMESPACE
{
    ArchBase::ArchBase( ArchBase::arch_id_t const arch_id,
        const char *const SIXTRL_RESTRICT arch_str,
        const char *const SIXTRL_RESTRICT config_str ) :
        SIXTRL_CXX_NAMESPACE::ArchInfo( arch_id, arch_str ),
        m_config_str()
    {
        this->doUpdateStoredConfigStr( config_str );
    }

    ArchBase::ArchBase( ArchBase::arch_id_t const arch_id,
        std::string const& SIXTRL_RESTRICT arch_str,
        std::string const& SIXTRL_RESTRICT config_str ) :
        SIXTRL_CXX_NAMESPACE::ArchInfo( arch_str,
            ( !arch_str.empty() ) ? arch_str.c_str() : nullptr ),
        m_config_str( config_str )
    {

    }

    bool ArchBase::hasConfigStr() const SIXTRL_NOEXCEPT
    {
        return !this->m_config_str.empty();
    }

    std::string const& ArchBase::configStr() const SIXTRL_NOEXCEPT
    {
        return this->m_config_str;
    }

    char const* ArchBase::ptrConfigStr() const SIXTRL_NOEXCEPT
    {
        return this->m_config_str.c_str();
    }

    bool ArchBase::doParseConfigStr(
        const char *const SIXTRL_RESTRICT config_str )
    {
        return this->doParseConfigStrArchBase( config_str );
    }

    void ArchBase::doUpdateStoredConfigStr(
        const char *const SIXTRL_RESTRICT config_str )
    {
        if( ( config_str != nullptr ) &&
            ( std::strlen( config_str ) > ArchBase::size_type{ 0 } ) )
        {
            this->m_config_str = config_str;
        }
        else
        {
            this->m_config_str.clear();
        }
    }

    bool ArchBase::doParseConfigStrArchBase(
        const char *const SIXTRL_RESTRICT config_str ) SIXTRL_NOEXCEPT
    {
        ( void )config_str;

        return true;
    }
}

/* end: sixtracklib/common/context/arch_base.cpp */
