#ifndef SIXTRACKLIB_CUDA_INTERNAL_ARGUMENT_BASE_H__
#define SIXTRACKLIB_CUDA_INTERNAL_ARGUMENT_BASE_H__

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

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"

    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* defined( __cplusplus ) */

    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/context/argument_base.h"

#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    class CudaContextBase;

    class CudaArgumentBase : public SIXTRL_CXX_NAMESPACE::ArgumentBase
    {
        private:

        using _base_argument_t    = SIXTRL_CXX_NAMESPACE::ArgumentBase;

        public:

        using buffer_t            = _base_argument_t::buffer_t;
        using c_buffer_t          = _base_argument_t::c_buffer_t;
        using size_type           = _base_argument_t::::size_type;

        SIXTRL_HOST_FN virtual ~CudaArgumentBase() SIXTRL_NOEXCEPT;

        protected:

        using ptr_cuda_base_context_t       = CudaContextBase*;
        using ptr_cuda_base_const_context_t = CudaContextBase const*;

        using ptr_cuda_arg_buffer_t         = unsigned char*;
        using ptr_const_cuda_arg_buffer_t   = unsigned char const*;

        SIXTRL_HOST_FN explicit CudaArgumentBase( ptr_context_t SIXTRL_RESTRICT
            ptr_context = nullptr ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN explicit CudaArgumentBase(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            ptr_cuda_base_context_t SIXTRL_RESTRICT ptr_context = nullptr );

        SIXTRL_HOST_FN explicit CudaArgumentBase(
            c_buffer_t* SITRL_RESTRICT ptr_c_buffer,
            ptr_cuda_base_context_t SIXTRL_RESTRICT ptr_context = nullptr );

        SIXTRL_HOST_FN explicit CudaArgumentBase( size_type const arg_size,
            ptr_cuda_base_context_t SIXTRL_RESTRICT ptr_context = nullptr );

        SIXTRL_HOST_FN explicit CudaArgumentBase(
            void const* SIXTRL_RESTRICT arg_buffer_begin,
            size_type const arg_size,
            ptr_cuda_base_context_t SIXTRL_RESTRICT ptr_context = nullptr );

        CudaArgumentBase( CudaArgumentBase const& other ) = delete;
        CudaArgumentBase( CudaArgumentBase&& other ) = delete;

        CudaArgumentBase& operator=( CudaArgumentBase const& other ) = delete;
        CudaArgumentBase& operator=( CudaArgumentBase&& other ) = delete;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN ptr_cuda_arg_buffer_t
        doGetCudaArgumentBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN ptr_const_cuda_arg_buffer_t
        doGetCudaArgumentBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doDeleteCudaArgumentBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doResetCudaArgumentBuffer(
            ptr_cuda_arg_buffer_t SIXTRL_RESTRICT new_arg_buffer,
            size_type const capacity );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN virtual int doReserveArgumentBuffer(
            size_type const required_buffer_size ) override;

        SIXTRL_HOST_FN virtual int doTransferBufferToDevice(
            void const* SIXTRL_RESTRICT source_buffer_begin,
            size_type const buffer_size ) override;

        SIXTRL_HOST_FN virtual int doTransferBufferFromDevice(
            void* SIXTRL_RESTRICT dest_buffer_begin,
            size_type const buffer_size ) override;

        SIXTRL_HOST_FN virtual int doRemapCObjectBufferAtDevice() override;

        private:

        SIXTRL_HOST_FN int doInitWriteBufferCudaBaseImpl(
            c_buffer_t* SIXTRL_RESTRICT ptr_c_buffer );

        SIXTRL_HOST_FN int doReserveArgumentBufferCudaBaseImpl(
            size_type const required_buffer_size );

        SIXTRL_HOST_FN int doTransferBufferToDeviceCudaBaseImpl(
            void const* SIXTRL_RESTRICT source_buffer_begin,
            size_type const buffer_size );

        SIXTRL_HOST_FN int doTransferBufferFromDeviceCudaBaseImpl(
            void* SIXTRL_RESTRICT dest_buffer_begin,
            size_type const buffer_size );

        SIXTRL_HOST_FN int doRemapCObjectBufferAtDeviceCudaBaseImpl();

        unsigned char* m_cuda_arg_buffer;
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
