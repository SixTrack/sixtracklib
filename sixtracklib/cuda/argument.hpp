#ifndef SIXTRACKLIB_CUDA_ARGUMENT_HPP__
#define SIXTRACKLIB_CUDA_ARGUMENT_HPP__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include <cstddef>
        #include <cstdlib>
    #endif /* C++, Host */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/context/definitions.h"
    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* defined( __cplusplus ) */
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/internal/argument_base.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

namespace SIXTRL_CXX_NAMESPACE
{
    class CudaContext;

    class CudaArgument : public SIXTRL_CXX_NAMESPACE::CudaArgumentBase
    {
        private:

        using _base_arg_t = SIXTRL_CXX_NAMESPACE::CudaArgumentBase;

        public:

        using type_id_t                = _base_arg_t::type_id_t;
        using status_t                 = _base_arg_t::status_t;
        using buffer_t                 = _base_arg_t::buffer_t;
        using c_buffer_t               = _base_arg_t::c_buffer_t;
        using size_type                = _base_arg_t::size_type;
        using ptr_base_context_t       = _base_arg_t::ptr_base_context_t;
        using ptr_const_base_context_t = _base_arg_t::ptr_const_base_context_t;

        using ptr_cuda_context_t       = SIXTRL_CXX_NAMESPACE::CudaContext*;
        using ptr_const_cuda_context_t =
            SIXTRL_CXX_NAMESPACE::CudaContext const*;

        SIXTRL_HOST_FN explicit CudaArgument(
            CudaContext* SIXTRL_RESTRICT ctx = nullptr );

        SIXTRL_HOST_FN explicit CudaArgument(
            buffer_t const& SIXTRL_RESTRICT_REF buffer,
            CudaContext* SIXTRL_RESTRICT ctx = nullptr );

        SIXTRL_HOST_FN explicit CudaArgument(
            const c_buffer_t *const SIXTRL_RESTRICT ptr_c_buffer,
            CudaContext* SIXTRL_RESTRICT ctx = nullptr );

        SIXTRL_HOST_FN explicit CudaArgument(
            size_type const arg_buffer_capacity,
            CudaContext* SIXTRL_RESTRICT ctx = nullptr );

        SIXTRL_HOST_FN CudaArgument(
            const void *const SIXTRL_RESTRICT raw_arg_begin,
            size_type const raw_arg_length,
            CudaContext* SIXTRL_RESTRICT ctx = nullptr );

        SIXTRL_HOST_FN CudaArgument( CudaArgument const& other ) = delete;
        SIXTRL_HOST_FN CudaArgument( CudaArgument&& other ) = delete;

        SIXTRL_HOST_FN CudaArgument&
        operator=( CudaArgument const& rhs ) = delete;
        SIXTRL_HOST_FN CudaArgument& operator=( CudaArgument&& rhs ) = delete;

        SIXTRL_HOST_FN virtual ~CudaArgument() = default;

        SIXTRL_HOST_FN ptr_cuda_context_t cudaContext() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN ptr_const_cuda_context_t
        cudaContext() const SIXTRL_NOEXCEPT;
    };
}

extern "C" { typedef SIXTRL_CXX_NAMESPACE::CudaArgument NS(CudaArgument); }

#else /* !defined( __cplusplus ) */

typedef void NS(CudaArgument);

#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_CUDA_ARGUMENT_H__ */

/* end: sixtracklib/cuda/argument.hpp */
