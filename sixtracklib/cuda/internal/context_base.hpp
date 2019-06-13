#ifndef SIXTRACKLIB_CUDA_INTERNAL_CONTEXT_BASE_HPP__
#define SIXTRACKLIB_CUDA_INTERNAL_CONTEXT_BASE_HPP__

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
    #include "sixtracklib/common/context/context_base.hpp"
    #include "sixtracklib/common/context/context_base_with_nodes.hpp"
    #include "sixtracklib/common/buffer.h"

    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* C++, Host */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

namespace SIXTRL_CXX_NAMESPACE
{
    class CudaContextBase : public SIXTRL_CXX_NAMESPACE::ContextOnNodesBase
    {
        private:

        using _base_context_t = SIXTRL_CXX_NAMESPACE::ContextOnNodesBase;

        public:

        using node_id_t      = _base_context_t::node_id_t;
        using node_info_t    = _base_context_t::node_info_t;
        using size_type      = _base_context_t::size_type;
        using platform_id_t  = _base_context_t::platform_id_t;
        using device_id_t    = _base_context_t::device_id_t;
        using type_id_t      = _base_context_t::type_id_t;
        using ptr_arg_base_t = _base_context_t::ptr_arg_base_t;
        using status_t       = _base_context_t::status_t;

        using buffer_t       = SIXTRL_CXX_NAMESPACE::Buffer;
        using c_buffer_t     = ::NS(Buffer);

        SIXTRL_HOST_FN virtual ~CudaContextBase() = default;

        protected:

        SIXTRL_HOST_FN explicit CudaContextBase(
            char const* config_str = nullptr );

        SIXTRL_HOST_FN CudaContextBase( CudaContextBase const& other ) = default;
        SIXTRL_HOST_FN CudaContextBase( CudaContextBase&& other ) = default;

        SIXTRL_HOST_FN CudaContextBase& operator=(
            CudaContextBase const& rhs ) = default;

        SIXTRL_HOST_FN CudaContextBase& operator=(
            CudaContextBase&& rhs) = default;

        SIXTRL_HOST_FN virtual status_t doSend(
            ptr_arg_base_t SIXTRL_RESTRICT destination,
            const void *const SIXTRL_RESTRICT source,
            size_type const source_length ) override;

        SIXTRL_HOST_FN virtual status_t doReceive(
            void* SIXTRL_RESTRICT destination,
            size_type const dest_capacity,
            ptr_arg_base_t SIXTRL_RESTRICT source ) override;

        SIXTRL_HOST_FN virtual status_t doRemapSentCObjectsBuffer(
            ptr_arg_base_t SIXTRL_RESTRICT arg,
            size_type const arg_size ) override;
    };
}

extern "C" {

typedef SIXTRL_CXX_NAMESPACE::CudaContextBase   NS(CudaContextBase);

}

#else /* C++, Host */

typedef void NS(CudaContextBase);

#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_INTERNAL_CONTEXT_BASE_HPP__ */

/* end: sixtracklib/cuda/internal/context_base.hpp */
