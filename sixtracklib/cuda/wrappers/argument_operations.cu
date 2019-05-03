#include "sixtracklib/cuda/wrappers/argument_operations.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/cuda/definitions.h"

NS(cuda_arg_buffer_t) NS(CudaArgument_alloc_arg_buffer)(
    NS(controller_size_t) const capacity )
{
    NS(cuda_arg_buffer_t) arg_buffer = SIXTRL_NULLPTR;

    if( capacity > ( NS(controller_size_t) )0u )
    {
        cudaError_t const ret = cudaMalloc( ( void** )&arg_buffer, capacity );

        if( ret != cudaSuccess )
        {
            if( arg_buffer != SIXTRL_NULLPTR )
            {
                cudaFree( arg_buffer );
                arg_buffer = SIXTRL_NULLPTR;
            }
        }
    }

    return arg_buffer;
}

void NS(CudaArgument_free_arg_buffer)(
    NS(cuda_arg_buffer_t) SIXTRL_RESTRICT arg_buffer )
{
    if( arg_buffer != SIXTRL_NULLPTR )
    {
        cudaFree( arg_buffer );
        arg_buffer = SIXTRL_NULLPTR;
    }

    return;
}

/* end: sixtracklib/cuda/wrappers/argument_operations.cu */
