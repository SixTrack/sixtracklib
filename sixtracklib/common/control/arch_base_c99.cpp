#include "sixtracklib/common/control/arch_base.h"

#include <cstddef>
#include <cstdlib>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/arch_base.hpp"

bool NS(ArchBase_has_config_str)(
    SIXTRL_ARGPTR_DEC const NS(ArchBase) *const SIXTRL_RESTRICT arch_base )
{
    return ( ( arch_base != nullptr ) && ( arch_base->hasConfigStr() ) );
}

char const* NS(ArchBase_get_config_string)(
    SIXTRL_ARGPTR_DEC const NS(ArchBase) *const SIXTRL_RESTRICT arch_base )
{
    return ( arch_base != nullptr ) ? arch_base->ptrConfigStr() : nullptr;
}

/* end: sixtracklib/common/control/arch_base_c99.cpp */
