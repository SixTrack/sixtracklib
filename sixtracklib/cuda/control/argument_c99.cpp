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
    return new st::CudaArgument( ctrl );
}

::NS(CudaArgument)* NS(CudaArgument_new_from_buffer)(
    ::NS(Buffer)* SIXTRL_RESTRICT buffer,
    ::NS(CudaController)* SIXTRL_RESTRICT ctrl )
{
    return new st::CudaArgument( buffer, ctrl );
}

::NS(CudaArgument)* NS(CudaArgument_new_from_size)(
    ::NS(ctrl_size_t) const capacity,
    ::NS(CudaController)* SIXTRL_RESTRICT ctrl )
{
    return new st::CudaArgument( capacity, ctrl );
}

::NS(CudaArgument)* NS(CudaArgument_new_from_raw_argument)(
    void const* SIXTRL_RESTRICT raw_arg_begin,
    ::NS(ctrl_size_t) const raw_arg_length,
    ::NS(CudaController)* SIXTRL_RESTRICT ctrl )
{
    return new st::CudaArgument( raw_arg_begin, raw_arg_length, ctrl );
}

/* ------------------------------------------------------------------------- */


bool NS(Argument_has_cuda_arg_buffer)(
    const ::NS(ArgumentBase) *const SIXTRL_RESTRICT arg )
{
    ::NS(CudaArgument) const* cuda_arg = st::asCudaArgument( arg );
    return ( ( cuda_arg != nullptr ) && ( cuda_arg->hasCudaArgBuffer() ) );
}

::NS(cuda_arg_buffer_t) NS(CudaArgument_get_cuda_arg_buffer)(
    ::NS(CudaArgument)* SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->cudaArgBuffer() : nullptr;
}

::NS(cuda_const_arg_buffer_t) NS(CudaArgument_get_const_cuda_arg_buffer)(
    const ::NS(CudaArgument) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->cudaArgBuffer() : nullptr;
}


SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
    ::NS(CudaArgument)* SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr )
        ? arg->cudaArgBufferAsCObjectsDataBegin() : nullptr;
}

SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(CudaArgument_get_cuda_arg_buffer_as_const_cobject_buffer_begin)(
    const ::NS(CudaArgument) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr )
        ? arg->cudaArgBufferAsCObjectsDataBegin() : nullptr;
}

SIXTRL_BUFFER_DATAPTR_DEC ::NS(arch_debugging_t)*
NS(CudaArgument_get_cuda_arg_buffer_as_debugging_register_begin)(
    ::NS(CudaArgument)* SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr )
        ? arg->cudaArgBufferAsPtrDebugRegister() : nullptr;
}

SIXTRL_BUFFER_DATAPTR_DEC ::NS(arch_debugging_t) const*
NS(CudaArgument_get_cuda_arg_buffer_as_const_debugging_register_begin)(
    const ::NS(CudaArgument) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr )
        ? arg->cudaArgBufferAsPtrDebugRegister() : nullptr;
}

SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC ::NS(ElemByElemConfig)*
NS(CudaArgument_get_cuda_arg_buffer_as_elem_by_elem_config_begin)(
    ::NS(CudaArgument)* SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr )
        ? arg->cudaArgBufferAsElemByElemByElemConfig() : nullptr;
}

SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC ::NS(ElemByElemConfig) const*
NS(CudaArgument_get_cuda_arg_buffer_as_const_elem_by_elem_config_begin)(
    const ::NS(CudaArgument) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr )
        ? arg->cudaArgBufferAsElemByElemByElemConfig() : nullptr;
}

#endif /* C++, Host */

/* end: sixtracklib/cuda/control/argument_c99.cpp */
