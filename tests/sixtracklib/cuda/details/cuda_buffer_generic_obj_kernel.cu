#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/impl/cuda_buffer_generic_obj_kernel.cuh"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cuda_runtime.h>
    #include <cuda.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/managed_buffer_minimal.h"
    #include "sixtracklib/common/impl/managed_buffer_remap.h"
    #include "sixtracklib/testlib/generic_buffer_obj.h"
    #include "sixtracklib/cuda/impl/cuda_tools.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

extern __global__ void Remap_original_buffer_kernel_cuda(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT orig_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT copy_begin,
    SIXTRL_BUFFER_DATAPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag );

extern __global__ void NS(Copy_original_buffer_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT orig_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*       SIXTRL_RESTRICT copy_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag );

extern __host__ int NS(Run_test_buffer_generic_obj_kernel_on_cuda)(
    dim3 const grid_dim, dim3 const block_dim,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT orig_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT copy_buffer );


__global__ void NS(Remap_original_buffer_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT orig_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT copy_begin,
    SIXTRL_BUFFER_DATAPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const thread_id =
        NS(Cuda_get_1d_thread_id_in_kernel)();

    buf_size_t const total_num_threads =
        NS(Cuda_get_total_num_threads_in_kernel)();

    buf_size_t const tid_to_remap_orig_buffer = ( buf_size_t )0u;

    buf_size_t const tid_to_remap_copy_buffer =
        ( total_num_threads > ( buf_size_t )1u )
            ? ( tid_to_remap_orig_buffer + ( buf_size_t )1u )
            : ( tid_to_remap_orig_buffer );

    if( thread_id <= tid_to_remap_copy_buffer )
    {
        int32_t success_flag = ( int32_t )0;
        buf_size_t const slot_size = ( buf_size_t )8u;

        if( thread_id == tid_to_remap_orig_buffer )
        {
            if( ( success_flag == 0 ) && ( orig_begin != SIXTRL_NULLPTR ) )
            {
                if( NS(ManagedBuffer_needs_remapping)( orig_begin, slot_size ) )
                {
                    if( NS(ManagedBuffer_remap)( orig_begin, slot_size ) != 0 )
                    {
                        success_flag |= -1;
                    }
                }
            }
            else if( orig_begin != SIXTRL_NULLPTR )
            {
                success_flag |= -2;
            }
        }

        if( thread_id == tid_to_remap_copy_buffer )
        {
            if( ( success_flag == 0 ) && ( copy_begin != SIXTRL_NULLPTR ) )
            {
                if( NS(ManagedBuffer_needs_remapping)( copy_begin, slot_size ) )
                {
                    if( NS(ManagedBuffer_remap)( copy_begin, slot_size ) != 0 )
                    {
                        success_flag |= -4;
                    }
                }
            }
            else if( copy_begin != SIXTRL_NULLPTR )
            {
                success_flag |= -8;
            }
        }

        if( ( success_flag != 0 ) && ( ptr_success_flag != SIXTRL_NULLPTR ) )
        {
            #if defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ > 120 )
            atomicOr( ptr_success_flag, success_flag );
            #else /* defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ > 120 ) */
            *ptr_success_flag |= success_flag;
            #endif /* defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ > 120 ) */
        }
    }

    return;
}

__global__ void NS(Copy_original_buffer_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT orig_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*       SIXTRL_RESTRICT copy_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t work_item_id =
        NS(Cuda_get_1d_thread_id_in_kernel)();

    buf_size_t const total_num_threads =
        NS(Cuda_get_total_num_threads_in_kernel)();

    buf_size_t const stride =
        NS(Cuda_get_1d_thread_stride_in_kernel)();

    buf_size_t const slot_size = ( buf_size_t )8u;

    int32_t success_flag = -1;

    if( ( !NS(ManagedBuffer_needs_remapping)( orig_buffer, slot_size ) ) &&
        ( !NS(ManagedBuffer_needs_remapping)( copy_buffer, slot_size ) ) )
    {
        typedef NS(Object)      object_t;
        typedef NS(GenericObj)  gen_obj_t;

        typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC object_t const*    in_index_ptr_t;
        typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC object_t*          out_index_ptr_t;

        typedef SIXTRL_BUFFER_OBJ_DATAPTR_DEC gen_obj_t const* in_obj_ptr_t;
        typedef SIXTRL_BUFFER_OBJ_DATAPTR_DEC gen_obj_t*       out_obj_ptr_t;

        buf_size_t const num_obj = NS(ManagedBuffer_get_num_objects)(
            orig_buffer, slot_size );

        success_flag = 0;

        while( ( work_item_id < num_obj ) && ( success_flag == 0 ) )
        {
            in_index_ptr_t ptr_in_info = ( in_index_ptr_t )( uintptr_t
                )NS(ManagedBuffer_get_const_objects_index_begin)(
                    orig_buffer, slot_size );

            out_index_ptr_t ptr_out_info = ( out_index_ptr_t )( uintptr_t
               )NS(ManagedBuffer_get_objects_index_begin)(
                   copy_buffer, slot_size );

            in_obj_ptr_t  in_obj  = SIXTRL_NULLPTR;
            out_obj_ptr_t out_obj = SIXTRL_NULLPTR;

            success_flag = ( ( ptr_in_info  != SIXTRL_NULLPTR ) &&
                             ( ptr_out_info != SIXTRL_NULLPTR ) ) ? 0 : -2;

            ptr_in_info   = ptr_in_info  + work_item_id;
            ptr_out_info  = ptr_out_info + work_item_id;
            work_item_id += stride;

            in_obj  = ( in_obj_ptr_t  )( uintptr_t
                )NS(Object_get_const_begin_ptr)( ptr_in_info );

            out_obj = ( out_obj_ptr_t )( uintptr_t
                )NS(Object_get_begin_ptr)( ptr_out_info );

            if( ( out_obj != SIXTRL_NULLPTR ) && ( in_obj != SIXTRL_NULLPTR ) &&
                ( out_obj != in_obj ) &&
                ( out_obj->type_id == in_obj->type_id ) &&
                ( out_obj->num_d   == in_obj->num_d   ) &&
                ( out_obj->num_e   == in_obj->num_e   ) &&
                ( out_obj->d != in_obj->d ) &&
                ( out_obj->d != SIXTRL_NULLPTR ) &&
                ( in_obj->d  != SIXTRL_NULLPTR ) &&
                ( out_obj->e != SIXTRL_NULLPTR ) &&
                ( in_obj->e  != SIXTRL_NULLPTR ) )
            {
                out_obj->a = in_obj->a;
                out_obj->b = in_obj->b;

                SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
                     &out_obj->c[ 0 ], &in_obj->c[ 0 ], ( size_t )4u );

                SIXTRACKLIB_COPY_VALUES( SIXTRL_UINT8_T,
                     out_obj->d, in_obj->d, in_obj->num_d );

                SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
                     out_obj->e, in_obj->e, in_obj->num_e );
            }
            else
            {
               success_flag |= -4;
            }
        }
    }

    if( ( success_flag != 0 ) && ( ptr_success_flag != SIXTRL_NULLPTR ) )
    {
        #if defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ > 120 )
        atomicOr( ptr_success_flag, success_flag );
        #else /* defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ > 120 ) */
        *ptr_success_flag |= success_flag;
        #endif /* defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ > 120 ) */
    }

    return;
}

__host__ int NS(Run_test_buffer_generic_obj_kernel_on_cuda)(
    dim3 const grid_dim, dim3 const block_dim,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT orig_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT copy_buffer_begin )
{
    typedef NS(buffer_size_t) buf_size_t;

    int success = -16;

    buf_size_t const slot_size = ( buf_size_t )8u;

    size_t const orig_buffer_size = NS(ManagedBuffer_get_buffer_length)(
        orig_buffer_begin, slot_size );

    size_t const copy_buffer_size = NS(ManagedBuffer_get_buffer_length)(
        copy_buffer_begin, slot_size );

    if( ( orig_buffer_begin != SIXTRL_NULLPTR ) &&
        ( copy_buffer_begin != SIXTRL_NULLPTR ) &&
        ( orig_buffer_size > slot_size ) &&
        ( orig_buffer_size == copy_buffer_size ) )
    {
        int32_t success_flag = 0;

        unsigned char* cuda_orig_begin = nullptr;
        unsigned char* cuda_copy_begin = nullptr;
        int32_t* cuda_success_flag     = nullptr;

        SIXTRL_ASSERT( orig_buffer_size == copy_buffer_size );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)( orig_buffer_begin,
            slot_size ) == NS(ManagedBuffer_get_num_of_objects)(
                copy_buffer_begin, slot_size ) );

        SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
            orig_buffer_begin, slot_size ) );

        SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
            copy_buffer_begin, slot_size ) );

        if( cudaSuccess == cudaMalloc(
            ( void** )&cuda_orig_begin, orig_buffer_size ) )
        {
            success = 0;
        }

        if( ( success == 0 ) && ( cudaSuccess != cudaMalloc(
                ( void** )&cuda_copy_begin, copy_buffer_size ) ) )
        {
            success |= -32;
        }

        if( ( success == 0 ) && ( cudaSuccess != cudaMalloc(
                ( void** )&cuda_success_flag, sizeof( success_flag ) ) ) )
        {
            success |= -64;
        }

        SIXTRL_ASSERT( ( success != 0 ) ||
            ( cuda_orig_begin != SIXTRL_NULLPTR ) &&
            ( cuda_copy_begin != SIXTRL_NULLPTR ) &&
            ( cuda_success_flag != SIXTRL_NULLPTr ) );

        if( ( success == 0 ) &&
            ( cudaSuccess != cudaMemcpy( cuda_orig_begin, orig_buffer_begin,
                orig_buffer_size, cudaMemcpyHostToDevice ) ) )
        {
            success |= -128;
        }

        if( ( success == 0 ) &&
            ( cudaSuccess != cudaMemcpy( cuda_copy_begin, copy_buffer_begin,
                copy_buffer_size, cudaMemcpyHostToDevice ) ) )
        {
            success |= -256;
        }

        if( ( success == 0 ) &&
            ( cudaSuccess != cudaMemcpy( cuda_success_flag, &success_flag,
                sizeof( success_flag ), cudaMemcpyHostToDevice ) ) )
        {
            success |= -512;
        }


        if( success == 0 )
        {
            NS(Remap_original_buffer_kernel_cuda)<<< grid_dim, block_dim >>>(
                cuda_orig_begin, cuda_copy_begin, cuda_success_flag );

            if( cudaSuccess != cudaDeviceSynchronize() )
            {
                success |= -1024;
            }
        }

        if( ( success == 0 ) &&
            ( cudaSuccess != cudaMemcpy( &success_flag, cuda_success_flag,
                sizeof( success_flag ), cudaMemcpyDeviceToHost ) ) )
        {
            success |= -2048;
        }

        if( success == 0 )
        {
            success |= ( int )success_flag;
        }

        if( success == 0 )
        {
            NS(Copy_original_buffer_kernel_cuda)<<< grid_dim, block_dim >>>(
                cuda_orig_begin, cuda_copy_begin, cuda_success_flag );

            if( cudaSuccess != cudaDeviceSynchronize() )
            {
                success |= -4096;
            }
        }

        if( ( success == 0 ) &&
            ( cudaSuccess != cudaMemcpy( &success_flag, cuda_success_flag,
                sizeof( success_flag ), cudaMemcpyDeviceToHost ) ) )
        {
            success |= -8192;
        }

        if( success == 0 )
        {
            success |= ( int )success_flag;
        }

        if( ( success == 0 ) &&
            ( cudaSuccess != cudaMemcpy( copy_buffer_begin, cuda_copy_begin,
                copy_buffer_size, cudaMemcpyDeviceToHost ) ) )
        {
            success |= -16384;
        }

        if( ( success == 0 ) &&
            ( 0 != NS(ManagedBuffer_remap)( copy_buffer_begin, slot_size ) ) )
        {
            success |= -32768;
        }

        if( ( ( cuda_orig_begin != SIXTRL_NULLPTR ) &&
              ( cudaSuccess != cudaFree( cuda_orig_begin ) ) ) ||
            ( ( cuda_copy_begin != SIXTRL_NULLPTR ) &&
              ( cudaSuccess != cudaFree( cuda_copy_begin ) ) ) ||
            ( ( cuda_success_flag != SIXTRL_NULLPTR ) &&
              ( cudaSuccess != cudaFree( cuda_success_flag ) ) ) )
        {
            success |= -65536;
        }
    }

    return success;
}

/* end: tests/sixtracklib/cuda/details/cuda_buffer_generic_obj_kernel.cu */
