#include "sixtracklib/cuda/internal/controller_base.h"

#if !defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cstddef>
#include <cstdlib>
#include <memory>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/cuda/definitions.h"
#include "sixtracklib/cuda/internal/controller_base.hpp"

void NS(CudaControllerBase_delete)(
    NS(CudaControllerBase)* SIXTRL_RESTRICT ctrl )
{
    delete ctrl;
    return;
}

#endif /* C++, Host */

/* end: sixtracklib/cuda/internal/controller_base_c99.cpp */
