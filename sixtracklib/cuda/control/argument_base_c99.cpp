#include "sixtracklib/cuda/control/argument_base.h"


#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cstddef>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"
#include "sixtracklib/cuda/control/argument_base.hpp"

bool NS(CudaArgument_has_cuda_arg_buffer)(
    const ::NS(CudaArgumentBase) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->hasCudaArgBuffer() : false;
}

::NS(cuda_arg_buffer_t) NS(CudaArgument_get_cuda_arg_buffer)(
    ::NS(CudaArgumentBase)* SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->cudaArgBuffer() : nullptr;
}

::NS(cuda_const_arg_buffer_t) NS(CudaArgument_get_const_cuda_arg_buffer)(
    const ::NS(CudaArgumentBase) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->cudaArgBuffer() : nullptr;
}


SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
    NS(CudaArgumentBase)* SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr )
        ? arg->cudaArgBufferAsCObjectsDataBegin() : nullptr;
}

SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(CudaArgument_get_cuda_arg_buffer_as_const_cobject_buffer_begin)(
    const NS(CudaArgumentBase) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr )
        ? arg->cudaArgBufferAsCObjectsDataBegin() : nullptr;
}

SIXTRL_BUFFER_DATAPTR_DEC NS(arch_debugging_t)*
NS(CudaArgument_get_cuda_arg_buffer_as_debugging_register_begin)(
    NS(CudaArgumentBase)* SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr )
        ? arg->cudaArgBufferAsPtrDebugRegister() : nullptr;
}

SIXTRL_BUFFER_DATAPTR_DEC NS(arch_debugging_t) const*
NS(CudaArgument_get_cuda_arg_buffer_as_const_debugging_register_begin)(
    const NS(CudaArgumentBase) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr )
        ? arg->cudaArgBufferAsPtrDebugRegister() : nullptr;
}

SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
NS(CudaArgument_get_cuda_arg_buffer_as_elem_by_elem_config_begin)(
    NS(CudaArgumentBase)* SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr )
        ? arg->cudaArgBufferAsElemByElemByElemConfig() : nullptr;
}

SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig) const*
NS(CudaArgument_get_cuda_arg_buffer_as_const_elem_by_elem_config_begin)(
    const NS(CudaArgumentBase) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr )
        ? arg->cudaArgBufferAsElemByElemByElemConfig() : nullptr;
}


#endif /* C++, Host */

/* end: sixtracklib/cuda/control/argument_base_c99.cpp */
