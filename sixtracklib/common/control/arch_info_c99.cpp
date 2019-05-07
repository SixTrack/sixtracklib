#include "sixtracklib/common/control/arch_info.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/arch_info.hpp"

namespace st = SIXTRL_CXX_NAMESPACE;

SIXTRL_ARGPTR_DEC ::NS(ArchInfo)* NS(ArchInfo_preset)(
    SIXTRL_ARGPTR_DEC ::NS(ArchInfo)* SIXTRL_RESTRICT arch_info )
{
    if( arch_info != nullptr )
    {
        arch_info->reset();
    }

    return arch_info;
}

SIXTRL_ARGPTR_DEC NS(ArchInfo)* NS(ArchInfo_create)()
{
    return new st::ArchInfo;
}

SIXTRL_ARGPTR_DEC NS(ArchInfo)* NS(ArchInfo_new)(
    NS(arch_id_t) const arch_id, char const* SIXTRL_RESTRICT arch_str )
{
    return new st::ArchInfo( arch_id, arch_str );
}

void NS(ArchInfo_delete)(
    SIXTRL_ARGPTR_DEC NS(ArchInfo)* SIXTRL_RESTRICT arch_info )
{
    delete arch_info;
}

::NS(arch_id_t) NS(ArchInfo_get_arch_id)(
    SIXTRL_ARGPTR_DEC const ::NS(ArchInfo) *const SIXTRL_RESTRICT arch_info )
{
    return ( arch_info != nullptr )
        ? arch_info->archId() : st::ArchInfo::ILLEGAL_ARCH;
}

bool NS(ArchInfo_has_arch_string)(
    SIXTRL_ARGPTR_DEC const ::NS(ArchInfo) *const SIXTRL_RESTRICT arch_info )
{
    return ( ( arch_info != nullptr ) && ( arch_info->hasArchStr() ) );
}

char const* NS(ArchInfo_get_arch_string)(
    SIXTRL_ARGPTR_DEC const ::NS(ArchInfo) *const SIXTRL_RESTRICT arch_info )
{
    return ( arch_info != nullptr ) ? arch_info->ptrArchStr() : nullptr;
}

bool NS(ArchInfo_is_compatible_with)(
    SIXTRL_ARGPTR_DEC const ::NS(ArchInfo) *const SIXTRL_RESTRICT arch_info,
    SIXTRL_ARGPTR_DEC const ::NS(ArchInfo) *const SIXTRL_RESTRICT other )
{
    return ( ( arch_info != nullptr ) && ( other != nullptr ) &&
             ( arch_info->isArchCompatibleWith( *other ) ) );
}

bool NS(ArchInfo_is_compatible_with_arch_id)(
    SIXTRL_ARGPTR_DEC const ::NS(ArchInfo) *const SIXTRL_RESTRICT arch_info,
    ::NS(arch_id_t) const other_arch_id )
{
    return ( ( arch_info != nullptr ) &&
             ( arch_info->isArchCompatibleWith( other_arch_id ) ) );
}

bool NS(ArchInfo_is_identical_to)(
    SIXTRL_ARGPTR_DEC const ::NS(ArchInfo) *const SIXTRL_RESTRICT arch_info,
    SIXTRL_ARGPTR_DEC const ::NS(ArchInfo) *const SIXTRL_RESTRICT other )
{
    return ( ( arch_info != nullptr ) && ( other != nullptr ) &&
             ( arch_info->isArchIdenticalTo( *other ) ) );
}

bool NS(ArchInfo_is_identical_to_arch_id)(
    SIXTRL_ARGPTR_DEC const ::NS(ArchInfo) *const SIXTRL_RESTRICT arch_info,
    ::NS(arch_id_t) const other_arch_id )
{
    return ( ( arch_info != nullptr ) &&
             ( arch_info->isArchIdenticalTo( other_arch_id ) ) );
}

void NS(ArchInfo_reset)(
    SIXTRL_ARGPTR_DEC ::NS(ArchInfo)* SIXTRL_RESTRICT arch_info,
    ::NS(arch_id_t) const arch_id, char const* SIXTRL_RESTRICT arch_str )
{
    if( arch_info != nullptr )
    {
        arch_info->reset( arch_id, arch_str );
    }
}

void NS(ArchInfo_reset_to_initial_values)(
    SIXTRL_ARGPTR_DEC ::NS(ArchInfo)* SIXTRL_RESTRICT arch_info )
{
    if( arch_info != nullptr )
    {
        arch_info->reset();
    }
}

/* ------------------------------------------------------------------------ */


NS(controller_status_t) NS(ArchInfo_sanitize_arch_str_inplace)(
    char* SIXTRL_RESTRICT arch_str,
    ::NS(buffer_size_t) const arch_str_capacity )
{
     NS(controller_status_t) success = ::NS(CONTROLLER_STATUS_GENERAL_FAILURE);

    if( ( arch_str != nullptr ) &&
        ( std::strlen( arch_str ) < arch_str_capacity ) )
    {
        std::string const str = st::ArchInfo_sanitize_arch_str( arch_str );

        if( ( !str.empty() ) && ( str.size() < arch_str_capacity ) )
        {
            std::memset( arch_str, ( int )'\0', arch_str_capacity );
            std::strncpy( arch_str, str.c_str(), str.size() );
            success = ::NS(CONTROLLER_STATUS_SUCCESS);
        }
    }

    return success;
}

NS(controller_status_t) NS(ArchInfo_sanitize_arch_str)(
    const char *const SIXTRL_RESTRICT arch_str,
    char* SIXTRL_RESTRICT sanarch_str,
    ::NS(buffer_size_t) const sanarch_str_capacity )
{
    using buf_size_t = ::NS(buffer_size_t);

    NS(controller_status_t) success = ::NS(CONTROLLER_STATUS_GENERAL_FAILURE);

    if( ( arch_str != nullptr ) && ( sanarch_str != nullptr ) &&
        ( sanarch_str_capacity > buf_size_t{ 1 } ) )
    {
        std::string const str = st::ArchInfo_sanitize_arch_str( arch_str );

        if( ( !str.empty() ) && ( str.size() <= sanarch_str_capacity ) )
        {
            std::memset( sanarch_str, ( int )'\0', sanarch_str_capacity );
            std::strncpy( sanarch_str, str.c_str(), str.size() );
            success = ::NS(CONTROLLER_STATUS_SUCCESS);
        }
    }

    return success;
}

/* end: sixtracklib/common/control/arch_info_c99.cpp */
