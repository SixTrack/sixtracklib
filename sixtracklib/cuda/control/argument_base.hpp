#ifndef SIXTRACKLIB_CUDA_CONTROL_ARGUMENT_BASE_HPP__
#define SIXTRACKLIB_CUDA_CONTROL_ARGUMENT_BASE_HPP__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include <cstddef>
        #include <cstdlib>
        #include <memory>
        #include <string>
    #endif /* C++, Host */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/common/control/argument_base.hpp"
    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* defined( __cplusplus ) */
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
namespace SIXTRL_CXX_NAMESPACE
{
    class CudaController;

    class CudaArgumentBase : public SIXTRL_CXX_NAMESPACE::ArgumentBase
    {
        public:

        using cuda_controller_t = SIXTRL_CXX_NAMESPACE::CudaControllerBase;
        using cuda_arg_buffer_t = ::NS(cuda_arg_buffer_t);
        using cuda_const_arg_buffer_t = ::NS(cuda_const_arg_buffer_t);

        SIXTRL_HOST_FN virtual ~CudaArgumentBase() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasCudaArgBuffer() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cuda_arg_buffer_t cudaArgBuffer() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cuda_const_arg_buffer_t
            cudaArgBuffer() const SIXTRL_NOEXCEPT;

        protected:

        using ptr_cuda_controller_t = cuda_controller_t*;
        using ptr_cuda_const_controller_t = cuda_controller_t const*;

        SIXTRL_HOST_FN explicit CudaArgumentBase(
            cuda_controller_t* SIXTRL_RESTRICT ptr_controller = nullptr );

        SIXTRL_HOST_FN explicit CudaArgumentBase(
            size_type const arg_buffer_capacity,
            cuda_controller_t* SIXTRL_RESTRICT ptr_controller = nullptr );

        SIXTRL_HOST_FN CudaArgumentBase(
            CudaArgumentBase const& other ) = delete;

        SIXTRL_HOST_FN CudaArgumentBase( CudaArgumentBase&& other ) = delete;

        SIXTRL_HOST_FN CudaArgumentBase& operator=(
            CudaArgumentBase const& other ) = delete;

        SIXTRL_HOST_FN CudaArgumentBase& operator=(
            CudaArgumentBase&& other ) = delete;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void doDeleteCudaArgumentBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doResetCudaArgumentBuffer(
            cuda_arg_buffer_t SIXTRL_RESTRICT new_arg_buffer,
            size_type const capacity );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN virtual bool doReserveArgumentBuffer(
            size_type const required_buffer_size ) override;

        SIXTRL_HOST_FN static cuda_arg_buffer_t CudaAllocArgBuffer(
            size_type const capacity );

        SIXTRL_HOST_FN static void CudaFreeArgBuffer(
            cuda_arg_buffer_t SIXTRL_RESTRICT arg_buffer );

        private:

        SIXTRL_HOST_FN bool doReserveArgumentBufferCudaBaseImpl(
            size_type const required_buffer_size );

        cuda_arg_buffer_t m_arg_buffer;
    };
}

extern "C" {

typedef SIXTRL_CXX_NAMESPACE::CudaArgumentBase NS(CudaArgumentBase);

}

#else /* C++, Host */

typedef void  NS(CudaArgumentBase);

#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_CONTROL_ARGUMENT_BASE_HPP__ */
/* end: sixtracklib/cuda/control/argument_base.h */
