#include "sixtracklib/cuda/context.h"

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cstddef>
#include <cstdlib>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/cuda/definitions.h"
#include "sixtracklib/cuda/context.hpp"
#include "sixtracklib/cuda/argument.hpp"

NS(CudaContext)* NS(CudaContext_create)( void )
{
    return new SIXTRL_CXX_NAMESPACE::CudaContext( "" );
}

void NS(CudaContext_delete)( NS(CudaContext)* SIXTRL_RESTRICT context )
{
    delete context;
    return;
}

#endif /* C++, Host */

/* end: sixtracklib/cuda/internal/context_c99.cpp */
