#include "sixtracklib/common/context/context_abs_base.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include "sixtracklib/common/generated/namespace.h"

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_HOST_FN ContextBase::ContextBase(
        ContextBase::type_id_t const type_id ) SIXTRL_NOEXCEPT
        : m_type_id( type_id )
    {

    }

    SIXTRL_HOST_FN ContextBase::type_id_t
    ContextBase::type() const SIXTRL_NOEXCEPT
    {
        return this->m_type_id;
    }
}

SIXTRL_HOST_FN NS(context_type_t) NS(ContextBase_get_type)(
    const NS(ContextBase) *const SIXTRL_RESTRICT context )
{
    return ( context != nullptr )
        ? static_cast< NS(context_type_t) >( context->type() )
        : NS(CONTEXT_TYPE_INVALID);
}

/* end: sixtracklib/common/context/context_abs_base.cpp */
