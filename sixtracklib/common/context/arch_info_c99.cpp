#include "sixtracklib/common/context/arch_info.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/context/definitions.h"
#include "sixtracklib/common/context/arch_info.hpp"

::NS(arch_id_t) NS(ArchInfo_get_arch_id)(
    SIXTRL_ARGPTR_DEC const ::NS(ArchInfo) *const SIXTRL_RESTRICT arch_info )
{
    return ( arch_info != nullptr )
        ? arch_info->archId() : SIXTRL_CXX_NAMESPACE::CONTEXT_TYPE_INVALID;
}

bool NS(ArchInfo_has_arch_str)(
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
             ( arch_info->isCompatibleWith( *other ) ) );
}

bool NS(ArchInfo_is_compatible_with_arch_id)(
    SIXTRL_ARGPTR_DEC const ::NS(ArchInfo) *const SIXTRL_RESTRICT arch_info,
    ::NS(arch_id_t) const other_arch_id )
{
    return ( ( arch_info != nullptr ) &&
             ( arch_info->isCompatibleWith( other_arch_id ) ) );
}

bool NS(ArchInfo_is_identical_to)(
    SIXTRL_ARGPTR_DEC const ::NS(ArchInfo) *const SIXTRL_RESTRICT arch_info,
    SIXTRL_ARGPTR_DEC const ::NS(ArchInfo) *const SIXTRL_RESTRICT other )
{
    return ( ( arch_info != nullptr ) && ( other != nullptr ) &&
             ( arch_info->isIdenticalTo( *other ) ) );
}

bool NS(ArchInfo_is_identical_to_arch_id)(
    SIXTRL_ARGPTR_DEC const ::NS(ArchInfo) *const SIXTRL_RESTRICT arch_info,
    ::NS(arch_id_t) const other_arch_id )
{
    return ( ( arch_info != nullptr ) &&
             ( arch_info->isIdenticalTo( other_arch_id ) ) );
}

void NS(ArchInfo_reset)(
    SIXTRL_ARGPTR_DEC NS(ArchInfo)* SIXTRL_RESTRICT arch_info,
    NS(arch_id_t) const arch_id, const char *const SIXTRL_RESTRICT arch_str )
{
    if( arch_info != nullptr )
    {
        arch_info->reset( arch_id, arch_str );
    }
}

/* end: sixtracklib/common/context/arch_info_c99.cpp */
