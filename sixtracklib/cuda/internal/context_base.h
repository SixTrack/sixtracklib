#ifndef SIXTRACKLIB_CUDA_INTERNAL_CONTEXT_BASE_H__
#define SIXTRACKLIB_CUDA_INTERNAL_CONTEXT_BASE_H__

#if defined( __CUDACC__ )

#if defined( __cplusplus )

    #if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <iterator>
        #include <string>
        #include <map>
        #include <vector>

        #include <CL/cl.hpp>
    #endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

    #if !defined( SIXTRL_NO_INCLUDES )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* !defined( SIXTRL_NO_INCLUDES ) */

#endif /* defined( __cplusplus ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
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

        using node_id_t         = _base_context_t::node_id_t;
        using node_info_t       = _base_context_t::node_info_t;
        using size_type         = _base_context_t::size_type;
        using platform_id_t     = _base_context_t::platform_id_t;
        using device_id_t       = _base_context_t::device_id_t;
        using type_id_t         = _base_context_t::type_id_t;

        using buffer_t          = SIXTRL_CXX_NAMESPACE::Buffer;
        using c_buffer_t        = ::NS(Buffer);
        using status_t          = int32_t;

        SIXTRL_HOST_FN virtual ~CudaContextBase();

        protected:

        ContextOnNodesBase( const char *const SIXTRL_RESTRICT config_str,
            type_id_t const type_id );

        SIXTRL_HOST_FN explicit CudaContextBase(
            char const* config_str = nullptr );

        SIXTRL_HOST_FN CudaContextBase( CudaContextBase const& other ) = default;
        SIXTRL_HOST_FN CudaContextBase( CudaContextBase&& other ) = default;

        SIXTRL_HOST_FN CudaContextBase& operator=(
            CudaContextBase const& rhs ) = default;

        SIXTRL_HOST_FN CudaContextBase& operator=(
            CudaContextBase&& rhs) = default;
    };
}

#else /* defined( __cplusplus ) */


#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_CUDA_INTERNAL_CONTEXT_BASE_H__ */

/* end: sixtracklib/cuda/internal/context_base.h */
