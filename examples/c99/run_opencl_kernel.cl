#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/namespace.h"
    #include "sixtracklib/common/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__kernel__ void NS(Add_vectors_kernel)(
    __global__ double const* __restrict__ a,
    __global__ double const* __restrict__ b,
    __global__ double* __restrict__ result,
    long int const vector_size )
{
    long int const global_id = get_global_id( 0 );
    long int const stride    = get_global_size( 0 );
    long int ii = global_id;

    for( ; ii < vector_size ; ii += stride )
    {
        result[ ii ] = a[ ii ] + b[ ii ];
    }

    return;
}

__kernel__ void NS(Subtract_vectors_kernel)(
    __global__ double const* __restrict__ a,
    __global__ double const* __restrict__ b,
    __global__ double* __restrict__ result,
    long int const vector_size )
{
    long int const global_id = get_global_id( 0 );
    long int const stride    = get_global_size( 0 );
    long int ii = global_id;

    for( ; ii < vector_size ; ii += stride )
    {
        result[ ii ] = a[ ii ] - b[ ii ];
    }

    return;
}

/* end: examples/c99/run_opencl_kernel.cl */
