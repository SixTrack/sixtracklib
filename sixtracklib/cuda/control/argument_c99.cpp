#include "sixtracklib/cuda/argument.h"

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cstddef>
#include <cstdlib>

#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/cuda/argument.hpp"
#include "sixtracklib/cuda/controller.hpp"

namespace st = SIXTRL_CXX_NAMESPACE;

::NS(CudaArgument)* NS(CudaArgument_new)(
    ::NS(CudaController)* SIXTRL_RESTRICT ctrl )
{
    return new SIXTRL_CXX_NAMESPACE::CudaArgument( ctrl );
}

::NS(CudaArgument)* NS(CudaArgument_new_from_buffer)(
    ::NS(Buffer)* SIXTRL_RESTRICT buffer,
    ::NS(CudaController)* SIXTRL_RESTRICT ctrl )
{
    return new SIXTRL_CXX_NAMESPACE::CudaArgument( buffer, ctrl );
}

::NS(CudaArgument)* NS(CudaArgument_new_from_size)(
    ::NS(ctrl_size_t) const capacity,
    ::NS(CudaController)* SIXTRL_RESTRICT ctrl )
{
    return new SIXTRL_CXX_NAMESPACE::CudaArgument( capacity, ctrl );
}

::NS(CudaArgument)* NS(CudaArgument_new_from_raw_argument)(
    void const* SIXTRL_RESTRICT raw_arg_begin,
    ::NS(ctrl_size_t) const raw_arg_length,
    ::NS(CudaController)* SIXTRL_RESTRICT ctrl )
{
    return new SIXTRL_CXX_NAMESPACE::CudaArgument(
        raw_arg_begin, raw_arg_length, ctrl );
}

void NS(CudaArgument_delete)( ::NS(CudaArgument)* SIXTRL_RESTRICT argument )
{
    delete argument;
}

::NS(ctrl_status_t) NS(CudaArgument_send_buffer)(
    ::NS(CudaArgument)* SIXTRL_RESTRICT arg,
    const ::NS(Buffer) *const SIXTRL_RESTRICT source_buffer )
{
    return ( arg != nullptr )
        ? arg->send( source_buffer ) : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(ctrl_status_t) NS(CudaArgument_send_memory)(
    ::NS(CudaArgument)* SIXTRL_RESTRICT arg,
    const void *const SIXTRL_RESTRICT source_arg_begin,
    ::NS(ctrl_size_t) const source_arg_length )
{
    return ( arg != nullptr )
        ? arg->send( source_arg_begin, source_arg_length )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(ctrl_status_t) NS(CudaArgument_receive_buffer)(
    ::NS(CudaArgument)* SIXTRL_RESTRICT arg,
    ::NS(Buffer)* SIXTRL_RESTRICT destination_buffer )
{
    return ( arg != nullptr )
        ? arg->receive( destination_buffer )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(ctrl_status_t) NS(CudaArgument_receive_memory)(
    ::NS(CudaArgument)* SIXTRL_RESTRICT arg,
    void* SIXTRL_RESTRICT destination_buffer,
    ::NS(ctrl_size_t) const destination_capacity )
{
    return ( arg != nullptr )
        ? arg->receive( destination_buffer, arg->size() )
        : st::ARCH_STATUS_GENERAL_FAILURE;
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

::NS(ctrl_size_t) NS(CudaArgument_get_size)(
    const ::NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr ) ? argument->size() : ::NS(ctrl_size_t){ 0 };
}

::NS(ctrl_size_t) NS(CudaArgument_get_capacity)(
    const ::NS(CudaArgument) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->capacity() : ::NS(ctrl_size_t){ 0 };
}

bool NS(CudaArgument_has_argument_buffer)(
    const ::NS(CudaArgument) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->hasArgumentBuffer() : false;
}

bool NS(CudaArgument_requires_argument_buffer)(
    const ::NS(CudaArgument) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->requiresArgumentBuffer() : false;
}

::NS(arch_id_t) NS(CudaArgument_get_arch_id)(
    const ::NS(CudaArgument) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->archId() : ::NS(ARCHITECTURE_ILLEGAL);
}

char const* NS(CudaArgument_get_arch_string)(
    const ::NS(CudaArgument) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->ptrArchStr() : nullptr;
}

#endif /* C++, Host */

/* end: sixtracklib/cuda/control/argument_c99.cpp */
