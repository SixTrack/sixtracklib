#include "sixtracklib/cuda/controller.h"

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cstddef>
#include <cstdlib>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/cuda/definitions.h"
#include "sixtracklib/cuda/controller.hpp"
#include "sixtracklib/cuda/argument.hpp"

NS(CudaController)* NS(CudaController_create)( void )
{
    return new SIXTRL_CXX_NAMESPACE::CudaController( "" );
}

void NS(CudaController_delete)( NS(CudaController)* SIXTRL_RESTRICT ctrl )
{
    delete ctrl;
    return;
}

#endif /* C++, Host */

/* end: sixtracklib/cuda/internal/controller_c99.cpp */
