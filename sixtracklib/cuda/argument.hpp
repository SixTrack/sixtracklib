#ifndef SIXTRACKLIB_CUDA_ARGUMENT_HPP__
#define SIXTRACKLIB_CUDA_ARGUMENT_HPP__

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstddef>
    #include <cstdlib>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#endif /* C++, Host */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/control/argument_base.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer.hpp"
    #include "sixtracklib/cuda/control/argument_base.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    class CudaController;

    class CudaArgument : public SIXTRL_CXX_NAMESPACE::CudaArgumentBase
    {
        private:

        using _base_arg_t = SIXTRL_CXX_NAMESPACE::CudaArgumentBase;

        public:

        using arch_id_t  = _base_arg_t::arch_id_t;
        using status_t   = _base_arg_t::status_t;
        using buffer_t   = _base_arg_t::buffer_t;
        using c_buffer_t = _base_arg_t::c_buffer_t;
        using size_type  = _base_arg_t::size_type;

        using ptr_base_controller_t = _base_arg_t::ptr_base_controller_t;
        using ptr_const_base_controller_t =
            _base_arg_t::ptr_const_base_controller_t;

        using ptr_cuda_controller_t = SIXTRL_CXX_NAMESPACE::CudaController*;
        using ptr_const_cuda_controller_t =
            SIXTRL_CXX_NAMESPACE::CudaController const*;

        SIXTRL_HOST_FN explicit CudaArgument(
            CudaController* SIXTRL_RESTRICT ctx = nullptr );

        SIXTRL_HOST_FN explicit CudaArgument(
            buffer_t const& SIXTRL_RESTRICT_REF buffer,
            CudaController* SIXTRL_RESTRICT ctx = nullptr );

        SIXTRL_HOST_FN explicit CudaArgument(
            const c_buffer_t *const SIXTRL_RESTRICT ptr_c_buffer,
            CudaController* SIXTRL_RESTRICT ctx = nullptr );

        SIXTRL_HOST_FN explicit CudaArgument(
            size_type const arg_buffer_capacity,
            CudaController* SIXTRL_RESTRICT ctx = nullptr );

        SIXTRL_HOST_FN CudaArgument(
            const void *const SIXTRL_RESTRICT raw_arg_begin,
            size_type const raw_arg_length,
            CudaController* SIXTRL_RESTRICT ctx = nullptr );

        SIXTRL_HOST_FN CudaArgument( CudaArgument const& other ) = delete;
        SIXTRL_HOST_FN CudaArgument( CudaArgument&& other ) = delete;

        SIXTRL_HOST_FN CudaArgument&
        operator=( CudaArgument const& rhs ) = delete;
        SIXTRL_HOST_FN CudaArgument& operator=( CudaArgument&& rhs ) = delete;

        SIXTRL_HOST_FN virtual ~CudaArgument() = default;

        SIXTRL_HOST_FN ptr_cuda_controller_t cudaController() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN ptr_const_cuda_controller_t
        cudaController() const SIXTRL_NOEXCEPT;
    };
}
#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++ */

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

typedef SIXTRL_CXX_NAMESPACE::CudaArgument NS(CudaArgument);

#else /* C++, Host */

typedef void NS(CudaArgument);

#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++ */

#endif /* SIXTRACKLIB_CUDA_ARGUMENT_H__ */

/* end: sixtracklib/cuda/argument.hpp */
