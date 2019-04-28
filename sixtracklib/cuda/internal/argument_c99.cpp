#include "sixtracklib/cuda/argument.h"

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cstddef>
#include <cstdlib>

#include "sixtracklib/common/context/definitions.h"
#include "sixtracklib/cuda/argument.hpp"
#include "sixtracklib/cuda/context.hpp"

namespace st = SIXTRL_CXX_NAMESPACE;

::NS(CudaArgument)* NS(CudaArgument_new)(
    ::NS(CudaContext)* SIXTRL_RESTRICT ctx )
{
    return new SIXTRL_CXX_NAMESPACE::CudaArgument( ctx );
}

::NS(CudaArgument)* NS(CudaArgument_new_from_buffer)(
    ::NS(Buffer)* SIXTRL_RESTRICT buffer,
    ::NS(CudaContext)* SIXTRL_RESTRICT ctx )
{
    return new SIXTRL_CXX_NAMESPACE::CudaArgument( buffer, ctx );
}

::NS(CudaArgument)* NS(CudaArgument_new_from_size)(
    ::NS(context_size_t) const capacity,
    ::NS(CudaContext)* SIXTRL_RESTRICT ctx )
{
    return new SIXTRL_CXX_NAMESPACE::CudaArgument( capacity, ctx );
}

::NS(CudaArgument)* NS(CudaArgument_new_from_memory)(
    void const* SIXTRL_RESTRICT raw_arg_begin,
    ::NS(context_size_t) const raw_arg_length,
    ::NS(CudaContext)* SIXTRL_RESTRICT ctx )
{
    return new SIXTRL_CXX_NAMESPACE::CudaArgument(
        raw_arg_begin, raw_arg_length, ctx );
}

void NS(CudaArgument_delete)( ::NS(CudaArgument)* SIXTRL_RESTRICT argument )
{
    delete argument;
}

::NS(context_status_t) NS(CudaArgument_send_buffer)(
    ::NS(CudaArgument)* SIXTRL_RESTRICT arg,
    const ::NS(Buffer) *const SIXTRL_RESTRICT source_buffer )
{
    return ( arg != nullptr )
        ? arg->send( source_buffer ) : st::CONTEXT_STATUS_GENERAL_FAILURE;
}

::NS(context_status_t) NS(CudaArgument_send_memory)(
    ::NS(CudaArgument)* SIXTRL_RESTRICT arg,
    const void *const SIXTRL_RESTRICT source_arg_begin,
    ::NS(context_size_t) const source_arg_length )
{
    return ( arg != nullptr )
        ? arg->send( source_arg_begin, source_arg_length )
        : st::CONTEXT_STATUS_GENERAL_FAILURE;
}

::NS(context_status_t) NS(CudaArgument_receive_buffer)(
    ::NS(CudaArgument)* SIXTRL_RESTRICT arg,
    ::NS(Buffer)* SIXTRL_RESTRICT destination_buffer )
{
    return ( arg != nullptr )
        ? arg->receive( destination_buffer )
        : st::CONTEXT_STATUS_GENERAL_FAILURE;
}

::NS(context_status_t) NS(CudaArgument_receive_memory)(
    ::NS(CudaArgument)* SIXTRL_RESTRICT arg,
    void* SIXTRL_RESTRICT destination_buffer,
    ::NS(context_size_t) const destination_capacity )
{
    return ( arg != nullptr )
        ? arg->receive( destination_buffer, arg->size() )
        : st::CONTEXT_STATUS_GENERAL_FAILURE;
}


bool NS(CudaArgument_uses_cobjects_buffer)(
    const ::NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr ) ? argument->usesCObjectsBuffer() : false;
}

::NS(Buffer)* NS(CudaArgument_get_cobjects_buffer)(
    const ::NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr ) ? argument->ptrCObjectsBuffer() : nullptr;
}

bool NS(CudaArgument_uses_raw_argument)(
    const ::NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr ) ? argument->usesRawArgument() : false;
}

bool NS(CudaArgument_get_ptr_raw_argument)(
    const ::NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr ) ? argument->ptrRawArgument() : nullptr;
}

::NS(context_size_t) NS(CudaArgument_get_size)(
    const ::NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr )
        ? argument->size() : ::NS(context_size_t){ 0 };
}

::NS(context_size_t) NS(CudaArgument_get_capacity)(
    const ::NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr )
        ? argument->capacity() : ::NS(context_size_t){ 0 };
}

bool NS(CudaArgument_has_argument_buffer)(
    const ::NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr ) ? argument->hasArgumentBuffer() : false;
}

bool NS(CudaArgument_requires_argument_buffer)(
    const ::NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr )
        ? argument->requiresArgumentBuffer() : false;
}

::NS(context_type_id_t) NS(CudaArgument_get_type_id)(
    const ::NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr ) ? argument->type()
        : SIXTRL_CXX_NAMESPACE::CONTEXT_TYPE_INVALID;
}

char const* NS(CudaArgument_get_type_id_str)(
    const ::NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr ) ? argument->ptrTypeStr() : nullptr;
}

#endif /* C++, Host */

/* end: sixtracklib/cuda/internal/argument_c99.cpp */
