#ifndef SIXTRACKLIB_CUDA_CONTEXT_HPP__
#define SIXTRACKLIB_CUDA_CONTEXT_HPP__

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
    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/internal/context_base.hpp"
    #include "sixtracklib/cuda/internal/argument_base.hpp"
    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* C++, Host */
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
namespace SIXTRL_CXX_NAMESPACE
{
    class CudaContext : public SIXTRL_CXX_NAMESPACE::CudaContextBase
    {
        private:

        using _base_context_t = SIXTRL_CXX_NAMESPACE::CudaContextBase;

        public:

        using node_id_t      = _base_context_t::node_id_t;
        using node_info_t    = _base_context_t::node_info_t;
        using size_type      = _base_context_t::size_type;
        using platform_id_t  = _base_context_t::platform_id_t;
        using device_id_t    = _base_context_t::device_id_t;
        using type_id_t      = _base_context_t::type_id_t;
        using ptr_arg_base_t = _base_context_t::ptr_arg_base_t;

        using buffer_t       = _base_context_t::buffer_t;
        using c_buffer_t     = _base_context_t::c_buffer_t;
        using status_t       = _base_context_t::status_t;


        SIXTRL_HOST_FN explicit CudaContext(
            char const* config_str = nullptr );

        SIXTRL_HOST_FN CudaContext( CudaContext const& other) = default;
        SIXTRL_HOST_FN CudaContext( CudaContext&& other) = default;

        SIXTRL_HOST_FN CudaContext&
        operator=( CudaContext const& other) = default;

        SIXTRL_HOST_FN CudaContext& operator=( CudaContext&& other) = default;

        SIXTRL_HOST_FN virtual ~CudaContext() = default;

        protected:

        using success_flag_arg_t = SIXTRL_CXX_NAMESPACE::CudaArgumentBase;
        using cuda_arg_buffer_t  = ::NS(cuda_arg_buffer_t);
        using cuda_const_arg_buffer_t = ::NS(cuda_const_arg_buffer_t);

        SIXTRL_HOST_FN virtual success_flag_t
            doGetSuccessFlagValueFromArg() const override;

        SIXTRL_HOST_FN virtual void doSetSuccessFlagValueFromArg(
            success_flag_t const success_flag ) override;

        SIXTRL_HOST_FN success_flag_arg_t* doGetPtrToDerivedSuccessFlagArg();

        SIXTRL_HOST_FN success_flag_arg_t const*
        doGetPtrToDerivedSuccessFlagArg() const;
    };
}

extern "C" { typedef SIXTRL_CXX_NAMESPACE::CudaContext NS(CudaContext); }

#else /* C++, Host */

typedef void NS(CudaContext);

#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_CONTEXT_HPP__ */

/* end: sixtracklib/cuda/context.hpp */
