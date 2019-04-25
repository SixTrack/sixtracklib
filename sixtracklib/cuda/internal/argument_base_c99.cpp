#include "sixtracklib/cuda/internal/argument_base.h"


#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cstddef>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/cuda/internal/argument_base.hpp"

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

#endif /* C++, Host */

/* end: sixtracklib/cuda/internal/argument_base_c99.cpp */
