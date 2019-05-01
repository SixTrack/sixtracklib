#include "sixtracklib/common/context/arch_info.hpp"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/context/definitions.h"

namespace SIXTRL_CXX_NAMESPACE
{
    ArchInfo::ArchInfo( ArchInfo::arch_id_t const arch_id,
        const char *const SIXTRL_RESTRICT arch_str ) SIXTRL_NOEXCEPT :
        m_arch_str(), m_arch_id( arch_id )
    {
        this->doSetArchStr( arch_str );
    }

    ArchInfo::arch_id_t ArchInfo::archId() const SIXTRL_NOEXCEPT
    {
        return this->m_arch_id;
    }

    bool ArchInfo::hasArchStr() const SIXTRL_NOEXCEPT
    {
        return ( !this->m_arch_str.empty() );
    }

    std::string const& ArchInfo::archStr() const SIXTRL_NOEXCEPT
    {
        return this->m_arch_str;
    }

    char const* ArchInfo::ptrArchStr() const SIXTRL_NOEXCEPT
    {
        return this->m_arch_str.c_str();
    }

    bool ArchInfo::isCompatibleWith(
        ArchInfo const& SIXTRL_RESTRICT other ) const SIXTRL_NOEXCEPT
    {
        return ( this->m_arch_id == other.m_arch_id );
    }

    bool ArchInfo::operator==(
         ArchInfo const& SIXTRL_RESTRICT rhs ) const SIXTRL_NOEXCEPT
     {
        return ( this->m_arch_id == rhs.m_arch_id );
     }

    void ArchInfo::reset( ArchInfo::arch_id_t const arch_id,
        const char *const SIXTRL_RESTRICT arch_str )
    {
        this->m_arch_id = arch_id;

        if( ( arch_str != nullptr ) &&
            ( std::strlen( arch_str ) > ArchInfo::size_type{ 0 } ) )
        {
            this->m_arch_str = arch_str;
        }
        else
        {
            this->m_arch_str.clear();
        }

        return;
    }

    void ArchInfo::doSetArchId(
        ArchInfo::arch_id_t const arch_id ) SIXTRL_NOEXCEPT
    {
        this->m_arch_id = arch_id;
    }

    void ArchInfo::doSetArchStr(
        const char *const SIXTRL_RESTRICT arch_str )
    {
        if( ( arch_str != nullptr ) &&
            ( std::strlen( arch_str ) > std::size_t{ 0 } ) )
        {
            this->m_arch_str = std::string{ arch_str };
        }
        else
        {
            this->m_arch_str.clear();
        }
    }
}

/* end: sixtracklib/common/context/arch_info.cpp */
