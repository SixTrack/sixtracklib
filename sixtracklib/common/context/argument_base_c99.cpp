#include "sixtracklib/common/context/argument_base.h"

#if !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cstddef>
#include <cstdlib>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/context/definitions.h"
#include "sixtracklib/common/context/argument_base.hpp"
#include "sixtracklib/common/context/context_base.hpp"


::NS(context_type_id_t) NS(Argument_get_type)(
    const ::NS(ArgumentBase) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->type() : ::NS(CONTEXT_TYPE_INVALID);
}

char const* NS(Argument_get_ptr_type_strt)(
    const ::NS(ArgumentBase) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->ptrTypeStr() : nullptr;
}

::NS(context_status_t) NS(Argument_send_again)(
    ::NS(ArgumentBase)* SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->send() : ::NS(context_status_t){ -1 };
}

::NS(context_status_t) NS(Argument_send_buffer)(
    ::NS(ArgumentBase)* SIXTRL_RESTRICT arg,
    const ::NS(Buffer) *const SIXTRL_RESTRICT_REF buffer )
{
    return ( arg != nullptr )
        ? arg->send( buffer ) : ::NS(context_status_t){ -1 };
}

::NS(context_status_t) NS(Argument_send_memory)(
    ::NS(ArgumentBase)* SIXTRL_RESTRICT arg,
    void const* SIXTRL_RESTRICT arg_begin,
    ::NS(context_size_t) const arg_size )
{
    return ( arg != nullptr )
        ? arg->send( arg_begin, arg_size ) : ::NS(context_status_t){ -1 };
}

::NS(context_status_t) NS(Argument_receive_again)(
    ::NS(ArgumentBase)* SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->receive() : ::NS(context_status_t){ -1 };
}

::NS(context_status_t) NS(Argument_receive_buffer)(
    ::NS(ArgumentBase)* SIXTRL_RESTRICT arg,
    ::NS(Buffer)* SIXTRL_RESTRICT buf )
{
    return ( arg != nullptr )
        ? arg->receive( buf ) : ::NS(context_status_t){ -1 };
}

::NS(context_status_t) NS(Argument_receive_memory)(
    ::NS(ArgumentBase)* SIXTRL_RESTRICT arg, void* SIXTRL_RESTRICT arg_begin,
    ::NS(context_size_t) const arg_capacity )
{
    return ( arg != nullptr )
        ? arg->receive( arg_begin, arg_capacity )
        : ::NS(context_status_t){ -1 };
}

bool NS(Argument_uses_cobjects_buffer)(
    const ::NS(ArgumentBase) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->usesCObjectsBuffer() : false;
}

::NS(Buffer) const* NS(Argument_get_const_cobjects_buffer)(
    const ::NS(ArgumentBase) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->ptrCObjectsBuffer() : nullptr;
}

::NS(Buffer)* NS(Argument_get_cobjects_buffer)(
    ::NS(ArgumentBase)* SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->ptrCObjectsBuffer() : nullptr;
}


bool NS(Argument_is_using_raw_argument)(
    const ::NS(ArgumentBase) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->usesRawArgument() : false;
}

void const* NS(Argument_get_const_ptr_raw_argument)(
    const ::NS(ArgumentBase) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->ptrRawArgument() : nullptr;
}

void* NS(Argument_get_ptr_raw_argument)(
    ::NS(ArgumentBase)* SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->ptrRawArgument() : nullptr;
}

::NS(context_size_t) NS(Argument_get_size)(
    const ::NS(ArgumentBase) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->size() : ::NS(context_size_t){ 0 };
}

::NS(context_size_t) NS(Argument_get_capacity)(
    const ::NS(ArgumentBase) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->capacity() : ::NS(context_size_t){ 0 };
}

bool NS(Argument_has_argument_buffer)(
    const ::NS(ArgumentBase) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->hasArgumentBuffer() : false;
}

bool NS(Argument_requires_argument_buffer)(
    const ::NS(ArgumentBase) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->requiresArgumentBuffer() : false;
}

::NS(ContextBase) const* NS(Argument_get_ptr_base_context)(
    ::NS(Argument)* SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->ptrBaseContext() : nullptr;
}

::NS(ContextBase) const* NS(Argument_get_const_ptr_base_context)(
    const ::NS(Argument) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->ptrBaseContext() : nullptr;
}

#endif /* !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ ) */

/* end: sixtracklib/common/context/argument_base_c99.cpp */