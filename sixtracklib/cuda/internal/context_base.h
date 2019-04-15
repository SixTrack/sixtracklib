#ifndef SIXTRACKLIB_CUDA_INTERNAL_CONTEXT_BASE_H__
#define SIXTRACKLIB_CUDA_INTERNAL_CONTEXT_BASE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <cstddef>
        #include <cstdlib>
        #include <iterator>
        #include <string>
        #include <map>
        #include <vector>
    #endif /* defined( __cplusplus ) */

    #include <stdint.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/context/definitions.h"
    #include "sixtracklib/common/context/context_base.h"
    #include "sixtracklib/common/context/context_base_with_nodes.h"
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

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

        using buffer_t       = SIXTRL_CXX_NAMESPACE::Buffer;
        using c_buffer_t     = ::NS(Buffer);
        using status_t       = int32_t;

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

typedef SIXTRL_CXX_NAMESPACE::CudaContextBase   NS(CudaContextBase);

#else /* !defined( __cplusplus ) */

typedef void NS(CudaContextBase);

#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_CUDA_INTERNAL_CONTEXT_BASE_H__ */

/* end: sixtracklib/cuda/internal/context_base.h */
