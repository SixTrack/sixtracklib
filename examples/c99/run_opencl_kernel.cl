#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/generated/namespace.h"
    #include "sixtracklib/common/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__kernel void NS(Add_vectors_kernel)(
    __global double const* SIXTRL_RESTRICT a,
    __global double const* SIXTRL_RESTRICT b,
    __global double* SIXTRL_RESTRICT result, long int const vector_size )
{
    long int const global_id = get_global_id( 0 );
    long int const stride  = get_global_size( 0 );
    long int ii = global_id;

    for( ; ii < vector_size ; ii += stride )
    {
        result[ ii ] = a[ ii ] + b[ ii ];
    }

    return;
}

__kernel void NS(Subtract_vectors_kernel)(
    __global double const* SIXTRL_RESTRICT a,
    __global double const* SIXTRL_RESTRICT b,
    __global double* SIXTRL_RESTRICT result, long int const vector_size )
{
    long int const global_id = get_global_id( 0 );
    long int const stride  = get_global_size( 0 );

    long int ii = global_id;

    for( ; ii < vector_size ; ii += stride )
    {
        result[ ii ] = a[ ii ] - b[ ii ];
    }

    return;
}

/* end: examples/c99/run_opencl_kernel.cl */
