#include "sixtracklib/common/control/arch_base.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <regex>
#include <string>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/debug_register.h"
#include "sixtracklib/common/control/arch_info.hpp"

namespace st = SIXTRL_CXX_NAMESPACE;

namespace SIXTRL_CXX_NAMESPACE
{
    ArchBase::ArchBase( ArchBase::arch_id_t const arch_id,
        const char *const SIXTRL_RESTRICT arch_str,
        const char *const SIXTRL_RESTRICT config_str ) :
        st::ArchInfo( arch_id, arch_str ),
        m_config_str()
    {
        this->doUpdateStoredConfigStr( config_str );
    }

    ArchBase::ArchBase( ArchBase::arch_id_t const arch_id,
        std::string const& SIXTRL_RESTRICT arch_str,
        std::string const& SIXTRL_RESTRICT config_str ) :
        st::ArchInfo( arch_id, ( !arch_str.empty() ) ?
            arch_str.c_str() : nullptr ),
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

    /* ********************************************************************* */

    bool ArchBaseDebug::isInDebugMode() const SIXTRL_NOEXCEPT
    {
        return this->m_debug_mode;
    }

    bool ArchBaseDebug::enableDebugMode() const SIXTRL_NOEXCEPT
    {
        bool success = false;

        if( !this->isInDebugMode() )
        {
            success = this->doSwitchDebugMode( true );
        }
        else
        {
            success = true;
        }

        return success;
    }

    bool ArchBaseDebug::disableDebugMode() const SIXTRL_NOEXCEPT
    {
        bool success = false;

        if( this->isInDebugMode() )
        {
            success = this->doSwitchDebugModeFlag( false );
        }
        else
        {
            success = true;
        }

        return success;
    }

    bool ArchBaseDebug::doSwitchDebugMode( bool const is_in_debug_mode )
    {
        bool const old_debug_state = this->isInDebugMode();
        this->doSetIsInDebugModeFlag( is_in_debug_mode );

        return ( ( is_in_debug_mode == this->isInDebugMode() ) &&
                 ( is_in_debug_mode != old_debug_state ) );
    }

    void ArchBaseDebug::doSetDebugModeFlag(
        bool const debug_mode ) SIXTRL_NOEXCEPT
    {
        this->m_debug_mode = debug_mode;
    }

    /* --------------------------------------------------------------------- */

    ArchBaseDebug::debug_register_t
    ArchBaseDebug::debugRegister() const
    {
        ArchBaseDebug::status_t status = this->doFetchDebugRegister(
            this->doGetPtrLocalDebugRegister() );

        return ( status == st::ARCH_STATUS_SUCCESS )
            ? *this->doGetPtrLocalDebugRegister()
            : st::SIXTRL_ARCH_DEBUGGING_GENERAL_FAILURE;
    }

    ArchBaseDebug::status_t ArchBaseDebug::setDebugRegister(
        ArchBaseDebug::debug_register_t const debug_register )
    {
        return this->doSetDebugRegister( debug_register );
    }

    ArchBaseDebug::status_t ArchBaseDebug::prepareDebugRegisterForUse()
    {
        ArchBaseDebug::status_t const status = this->doSetDebugRegister(
            st::ARCH_DEBUGGING_REGISTER_EMPTY );

        SIXTRL_ASSERT( ( status != st::ARCH_STATUS_SUCCESS ) ||
            ( ( this->doGetPtrLocalDebugRegister() != nullptr ) &&
              ( *this->doGetPtrLocalDebugRegister() ==
                st::ARCH_DEBUGGING_REGISTER_EMPTY ) ) );

        return status;
    }

    ArchBaseDebug::status_t ArchBaseDebug::evaluateDebugRegisterAfterUse()
    {
        ArchBaseDebug::status_t status = this->doFetchDebugRegister(
            this->doGetPtrLocalDebugRegister() );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            if( ::NS(DebugReg_has_status_flags_set)(
                    *this->doGetPtrLocalDebugRegister() ) )
            {
                status = ::NS(DebugReg_get_stored_arch_status)(
                    *this->doGetPtrLocalDebugRegister() );
            }
            else if( ::NS(DebugReg_has_any_flags_set)(
                *this->doGetPtrLocalDebugRegister() ) )
            {
                status = st::ARCH_STATUS_GENERAL_FAILURE;
            }
        }

        return status;
    }

    /* --------------------------------------------------------------------- */

    ArchBaseDebug::status_t ArchBaseDebug::doSetDebugRegister(
        ArchBaseDebug::debug_register_t const debug_register )
    {
        ArchBaseDebug::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( this->doGetPtrLocalDebugRegister() != nullptr )
        {
            *this->doGetPtrLocalDebugRegister() = debug_register;
            status = st::ARCH_STATUS_SUCCESS;
        }

        return status;
    }

    ArchBaseDebug::status_t ArchBaseDebug::doFetchDebugRegister(
        ArchBaseDebug::debug_register_t* SIXTRL_RESTRICT ptr_debug_register )
    {
        ArchBaseDebug::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ptr_debug_register != nullptr )
        {
            if( ptr_debug_register != this->doGetPtrLocalDebugRegister() )
            {
                *ptr_debug_register = *this->doGetPtrLocalDebugRegister();
            }

            status = st::ARCH_STATUS_SUCCESS;
        }

        return status;
    }

    ArchBaseDebug::debug_register_t const*
    ArchBaseDebug::doGetPtrLocalDebugRegister() const SIXTRL_NOEXCEPT
    {
        return &this->m_local_debug_register;
    }

    ArchBaseDebug::debug_register_t*
    ArchBaseDebug::doGetPtrLocalDebugFlag() SIXTRL_NOEXCEPT
    {
        return &this->m_local_debug_register;
    }
}

/* end: sixtracklib/common/control/arch_base.cpp */
