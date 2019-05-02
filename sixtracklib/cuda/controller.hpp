#ifndef SIXTRACKLIB_CUDA_CONTROLLER_HPP__
#define SIXTRACKLIB_CUDA_CONTROLLER_HPP__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include <cstddef>
        #include <cstdlib>
    #endif /* C++, Host */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/internal/controller_base.hpp"
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
    class CudaController : public SIXTRL_CXX_NAMESPACE::CudaControllerBase
    {
        private:

        using _base_controller_t = SIXTRL_CXX_NAMESPACE::CudaControllerBase;

        public:

        using node_id_t      = _base_controller_t::node_id_t;
        using node_info_t    = _base_controller_t::node_info_t;
        using size_type      = _base_controller_t::size_type;
        using platform_id_t  = _base_controller_t::platform_id_t;
        using device_id_t    = _base_controller_t::device_id_t;
        using type_id_t      = _base_controller_t::type_id_t;
        using ptr_arg_base_t = _base_controller_t::ptr_arg_base_t;

        using buffer_t       = _base_controller_t::buffer_t;
        using c_buffer_t     = _base_controller_t::c_buffer_t;
        using status_t       = _base_controller_t::status_t;


        SIXTRL_HOST_FN explicit CudaController(
            char const* config_str = nullptr );

        SIXTRL_HOST_FN CudaController( CudaController const& other) = default;
        SIXTRL_HOST_FN CudaController( CudaController&& other) = default;

        SIXTRL_HOST_FN CudaController&
        operator=( CudaController const& other) = default;

        SIXTRL_HOST_FN CudaController& operator=( CudaController&& other) = default;

        SIXTRL_HOST_FN virtual ~CudaController() = default;

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

extern "C" { typedef SIXTRL_CXX_NAMESPACE::CudaController NS(CudaController); }

#else /* C++, Host */

typedef void NS(CudaController);

#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_CONTROLLER_HPP__ */

/* end: sixtracklib/cuda/controller.hpp */
