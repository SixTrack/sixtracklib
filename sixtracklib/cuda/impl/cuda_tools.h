#ifndef SIXTRACKLIB_CUDA_IMPL_CUDA_TOOLS_H__
#define SIXTRACKLIB_CUDA_IMPL_CUDA_TOOLS_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdio.h>
    #include <stdlib.h>

    #include <cuda_runtime_api.h>
    #include <cuda.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#if !defined( _GPUCODE )

SIXTRL_HOST_FN SIXTRL_STATIC unsigned long NS(Cuda_get_1d_thread_id)(
    dim3 const thread_idx, dim3 const block_idx,
    dim3 const grid_dim,   dim3 const block_dim );

SIXTRL_HOST_FN SIXTRL_STATIC unsigned long NS(Cuda_get_1d_thread_stride)(
    dim3 const grid_dim, dim3 const block_dim );

SIXTRL_HOST_FN SIXTRL_STATIC unsigned long NS(Cuda_get_num_blocks)(
    dim3 const grid_dim, dim3 const block_dim );

SIXTRL_HOST_FN SIXTRL_STATIC unsigned long NS(Cuda_get_num_threads_per_block)(
    dim3 const grid_dim, dim3 const block_dim );

SIXTRL_HOST_FN SIXTRL_STATIC unsigned long NS(Cuda_get_total_num_threads)(
    dim3 const grid_dim, dim3 const block_dim );

#endif /* !defined( _GPUCODE ) */

#if defined( _GPUCODE )

SIXTRL_DEVICE_FN SIXTRL_STATIC unsigned long
    NS(Cuda_get_1d_thread_id_in_kernel)( void );

SIXTRL_DEVICE_FN SIXTRL_STATIC unsigned long
    NS(Cuda_get_1d_thread_stride_in_kernel)( void );

SIXTRL_DEVICE_FN SIXTRL_STATIC unsigned long
    NS(Cuda_get_num_blocks_in_kernel)( void );

SIXTRL_DEVICE_FN SIXTRL_STATIC unsigned long
    NS(Cuda_get_num_threads_per_block_in_kernel)( void );

SIXTRL_DEVICE_FN SIXTRL_STATIC unsigned long
    NS(Cuda_get_total_num_threads_in_kernel)( void );

#endif /* #if defined( _GPUCODE ) */


#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====             Implementation of inline functions                ===== */
/* ========================================================================= */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#if !defined( _GPUCODE )

SIXTRL_INLINE unsigned long NS(Cuda_get_1d_thread_id)(
    dim3 const thread_idx, dim3 const block_idx,
    dim3 const grid_dim,   dim3 const block_dim )
{
    unsigned long const num_threads_per_block =
        NS(Cuda_get_num_threads_per_block)( grid_dim, block_dim );

    unsigned long thread_id = thread_idx.x;

    unsigned long temp = block_dim.x;
    thread_id += thread_idx.y * temp;

    temp *= block_dim.y;
    thread_id += temp * thread_idx.z;

    temp  = num_threads_per_block;
    thread_id += block_idx.x * temp;

    temp *= grid_dim.x;
    thread_id += block_idx.y * temp;

    temp *= grid_dim.y;
    return thread_id + block_idx.z * temp;
}

SIXTRL_INLINE unsigned long NS(Cuda_get_1d_thread_stride)(
    dim3 const grid_dim, dim3 const block_dim )
{
    SIXTRL_ASSERT(
        ( grid_dim.x  > 0 ) && ( grid_dim.y  > 0 ) && ( grid_dim.z  > 0 ) &&
        ( block_dim.y > 0 ) && ( block_dim.y > 0 ) && ( block_dim.z > 0 ) );

    return ( grid_dim.x  * grid_dim.y  * grid_dim.z  ) *
           ( block_dim.x * block_dim.y * block_dim.z );
}

SIXTRL_INLINE unsigned long NS(Cuda_get_num_blocks)(
    dim3 const grid_dim, dim3 const block_dim )
{
    ( void )block_dim;

    SIXTRL_ASSERT( ( grid_dim.x > 0 ) && ( grid_dim.y > 0 ) &&
                   ( grid_dim.z > 0 ) );

    return ( grid_dim.x * grid_dim.y * grid_dim.z );
}

SIXTRL_INLINE unsigned long NS(Cuda_get_num_threads_per_block)(
    dim3 const grid_dim, dim3 const block_dim )
{
    ( void )grid_dim;

    SIXTRL_ASSERT( ( block_dim.y > 0 ) && ( block_dim.y > 0 ) &&
                   ( block_dim.z > 0 ) );

    return ( block_dim.x * block_dim.y * block_dim.z );
}

SIXTRL_INLINE unsigned long NS(Cuda_get_total_num_threads)(
    dim3 const grid_dim, dim3 const block_dim )
{
    SIXTRL_ASSERT(
        ( grid_dim.x  > 0 ) && ( grid_dim.y  > 0 ) && ( grid_dim.z  > 0 ) &&
        ( block_dim.y > 0 ) && ( block_dim.y > 0 ) && ( block_dim.z > 0 ) );

    return ( grid_dim.x  * grid_dim.y  * grid_dim.z  ) *
           ( block_dim.x * block_dim.y * block_dim.z );
}

#endif /* !defined( _GPUCODE ) */

#if defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_DEVICE_FN unsigned long NS(Cuda_get_1d_thread_id_in_kernel)()
{
    unsigned long const num_threads_per_block =
        NS(Cuda_get_num_threads_per_block_in_kernel)();

    unsigned long thread_id = threadIdx.x;

    unsigned long temp = blockDim.x;
    thread_id += threadIdx.y * temp;

    temp *= blockDim.y;
    thread_id += temp * threadIdx.z;

    temp  = num_threads_per_block;
    thread_id += blockIdx.x * temp;

    temp *= gridDim.x;
    thread_id += blockIdx.y * temp;

    temp *= gridDim.y;
    return thread_id + blockIdx.z * temp;
}

SIXTRL_INLINE SIXTRL_DEVICE_FN unsigned long NS(Cuda_get_1d_thread_stride_in_kernel)()
{
    SIXTRL_ASSERT(
        ( gridDim.x  > 0 ) && ( gridDim.y  > 0 ) && ( gridDim.z  > 0 ) &&
        ( blockDim.y > 0 ) && ( blockDim.y > 0 ) && ( blockDim.z > 0 ) );

    return ( gridDim.x  * gridDim.y  * gridDim.z  ) *
           ( blockDim.x * blockDim.y * blockDim.z );
}

SIXTRL_INLINE SIXTRL_DEVICE_FN unsigned long NS(Cuda_get_num_blocks_in_kernel)()
{
    SIXTRL_ASSERT( ( gridDim.x > 0 ) && ( gridDim.y > 0 ) && ( gridDim.z > 0 ) );
    return ( gridDim.x * gridDim.y * gridDim.z );
}

SIXTRL_INLINE SIXTRL_DEVICE_FN unsigned long
    NS(Cuda_get_num_threads_per_block_in_kernel)()
{
    SIXTRL_ASSERT( ( blockDim.y > 0 ) && ( blockDim.y > 0 ) && ( blockDim.z > 0 ) );
    return ( blockDim.x * blockDim.y * blockDim.z );
}

SIXTRL_INLINE SIXTRL_DEVICE_FN unsigned long
    NS(Cuda_get_total_num_threads_in_kernel)()
{
    return NS(Cuda_get_num_blocks_in_kernel)() *
           NS(Cuda_get_num_threads_per_block_in_kernel)();
}

#endif /* defined( _GPUCODE ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_CUDA_IMPL_CUDA_TOOLS_H__ */

/* end: sixtracklib/cuda/impl/cuda_tools.h */
