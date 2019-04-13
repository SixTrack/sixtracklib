#ifndef SIXTRACKLIB_CUDA_INTERNAL_ARGUMENT_BASE_H__
#define SIXTRACKLIB_CUDA_INTERNAL_ARGUMENT_BASE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <limits.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

// #if !defined( _GPUCODE ) && defined( __cplusplus )
// extern "C" {
// #endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */
//
// struct NS(Buffer);
//
// #if !defined( _GPUCODE ) && defined( __cplusplus )
// }
// #endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <iterator>
        #include <memory>
        #include <string>
        #include <vector>
    #endif /* defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    class CudaContextBase;

    class CudaArgumentBase
    {
        public:

        using buffer_t                    = SIXTRL_CXX_NAMESPACE::Buffer;
        using c_buffer_t                  = ::NS(Buffer);
        using size_type                   = buffer_t::size_type;
        using ptr_context_t               = CudaContextBase*;
        using ptr_const_context_t         = CudaContextBase const*;
        using ptr_cuda_arg_buffer_t       = unsigned char*;
        using ptr_const_cuda_arg_buffer_t = unsigned char const*;

        SIXTRL_HOST_FN bool write( buffer_t& SIXTRL_RESTRICT_REF buffer );
        SIXTRL_HOST_FN bool write( c_buffer_t* SIXTRL_RESTRICT ptr_c_buffer );

        SIXTRL_HOST_FN bool write( void const* SIXTRL_RESTRICT arg_buffer_begin,
                                   size_type const arg_size );

        SIXTRL_HOST_FN bool read( buffer_t& SIXTRL_RESTRICT_REF buffer );
        SIXTRL_HOST_FN bool read( c_buffer_t* SIXTRL_RESTRICT ptr_c_buffer );
        SIXTRL_HOST_FN bool read( void* SIXTRL_RESTRICT arg_buffer_begin,
                                  size_type const arg_size );

        SIXTRL_HOST_FN bool usesCObjectsBuffer()  const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN buffer_t* ptrCObjectsBuffer() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN buffer_t& cobjectsBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool usesCObjectsCBuffer() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN c_buffer_t* ptrCObjectsCBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool usesRawArgumentBuffer() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN void* ptrRawArgumentBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN ptr_cuda_arg_buffer_t
        cudaArgumentBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN ptr_const_cuda_arg_buffer_t
        cudaArgumentBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type size() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN size_type capacity() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN virtual ~CudaArgumentBase() SIXTRL_NOEXCEPT;

        protected:

        explicit CudaArgumentBase( ptr_context_t SIXTRL_RESTRICT
            ptr_context = nullptr ) SIXTRL_NOEXCEPT;

        explicit CudaArgumentBase(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            ptr_context_t SIXTRL_RESTRICT ptr_context = nullptr );

        explicit CudaArgumentBase(
            c_buffer_t* SITRL_RESTRICT ptr_c_buffer,
            ptr_context_t SIXTRL_RESTRICT ptr_context = nullptr );

        explicit CudaArgumentBase( size_type const arg_size,
            ptr_context_t SIXTRL_RESTRICT ptr_context = nullptr );

        explicit CudaArgumentBase(
            void const* SIXTRL_RESTRICT arg_buffer_begin,
            size_type const arg_size,
            ptr_context_t SIXTRL_RESTRICT ptr_context = nullptr );

        CudaArgumentBase( CudaArgumentBase const& other ) = delete;
        CudaArgumentBase( CudaArgumentBase&& other ) = delete;

        CudaArgumentBase& operator=( CudaArgumentBase const& other ) = delete;
        CudaArgumentBase& operator=( CudaArgumentBase&& other ) = delete;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN virtual int doReserveCudaArgumentBuffer(
            size_type const required_buffer_size );

        SIXTRL_HOST_FN virtual int doTransferBufferToDevice(
            void const* SIXTRL_RESTRICT source_buffer_begin,
            size_type const buffer_size );

        SIXTRL_HOST_FN virtual int doTransferBufferFromDevice(
            void* SIXTRL_RESTRICT dest_buffer_begin,
            size_type const buffer_size );

        SIXTRL_HOST_FN virtual int doRemapCObjectBufferAtDevice();

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void doResetArgBuffer(
            unsigned char* new_arg_buffer, size_type const capacity );

        SIXTRL_HOST_FN void doSetPtrContext(
            ptr_context_t SIXTRL_RESTRICT ptr_context ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetPtrBuffer(
            buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetPtrCBuffer(
            c_buffer_t* SIXTRL_RESTRICT ptr_c_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetRawArgumentBuffer(
            void* SIXTRL_RESTRICT raw_arg_begin ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetArgSize(
            size_type const arg_size ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN ptr_const_context_t
        doGetPtrBaseContext() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN ptr_context_t doGetPtrBaseContext() SIXTRL_NOEXCEPT;

        private:

        SIXTRL_HOST_FN int doReserveCudaArgumentBufferBaseImpl(
            size_type const required_buffer_size );

        SIXTRL_HOST_FN int doTransferBufferToDeviceBaseImpl(
            void const* SIXTRL_RESTRICT source_buffer_begin,
            size_type const buffer_size );

        SIXTRL_HOST_FN int doTransferBufferFromDeviceBaseImpl(
            void* SIXTRL_RESTRICT dest_buffer_begin,
            size_type const buffer_size );

        SIXTRL_HOST_FN int doRemapCObjectBufferAtDeviceBaseImpl();

        unsigned char*          m_arg_buffer;

        mutable void*           m_ptr_raw_arg_buffer;
        mutable buffer_t*       m_ptr_cobj_buffer;
        mutable c_buffer_t*     m_ptr_c_cobj_buffer;
        ptr_context_t           m_ptr_context;

        size_type               m_arg_size;
        size_type               m_arg_capacity;
    };
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

typedef SIXTRL_CXX_NAMESPACE::CudaArgumentBase NS(CudaArgumentBase);

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#else /* !defined( __cplusplus ) */

typedef void NS(CudaArgumentBase);

#endif /* defined( __cplusplus ) */


#endif /* SIXTRACKLIB_CUDA_INTERNAL_ARGUMENT_BASE_H__ */
/* end: sixtracklib/cuda/internal/argument_base.h */
