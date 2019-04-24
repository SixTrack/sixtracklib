#include "sixtracklib/cuda/internal/context_base.h"

#if !defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cstddef>
#include <cstdlib>
#include <memory>

#include "sixtracklib/common/definitions.h"

void NS(CudaContext_delete)( NS(CudaContextBase)* SIXTRL_RESTRICT context )
{
    delete context;
    return;
}

#endif /* C++, Host */

/* end: sixtracklib/cuda/internal/context_base_c99.cpp */