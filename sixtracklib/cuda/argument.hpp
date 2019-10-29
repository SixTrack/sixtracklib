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
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/cuda/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/control/argument_base.hpp"
    #include "sixtracklib/common/buffer.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    class CudaController;

    class CudaArgument : public SIXTRL_CXX_NAMESPACE::ArgumentBase
    {
        private:

        using _base_arg_t = SIXTRL_CXX_NAMESPACE::ArgumentBase;

        public:

        using cuda_controller_t       = SIXTRL_CXX_NAMESPACE::CudaController;
        using cuda_arg_buffer_t       = ::NS(cuda_arg_buffer_t);
        using cuda_const_arg_buffer_t = ::NS(cuda_const_arg_buffer_t);
        using elem_by_elem_config_t   = ::NS(ElemByElemConfig);
        using debug_register_t        = ::NS(arch_debugging_t);

        using arch_id_t  = _base_arg_t::arch_id_t;
        using status_t   = _base_arg_t::status_t;
        using buffer_t   = _base_arg_t::buffer_t;
        using c_buffer_t = _base_arg_t::c_buffer_t;
        using size_type  = _base_arg_t::size_type;

        using ptr_base_controller_t =
            _base_arg_t::ptr_base_controller_t;

        using ptr_const_base_controller_t =
            _base_arg_t::ptr_const_base_controller_t;

        using ptr_cuda_controller_t =
            SIXTRL_CXX_NAMESPACE::CudaController*;

        using ptr_const_cuda_controller_t =
            SIXTRL_CXX_NAMESPACE::CudaController const*;

        /* ----------------------------------------------------------------- */

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

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN CudaArgument( CudaArgument const& other ) = delete;
        SIXTRL_HOST_FN CudaArgument( CudaArgument&& other ) = delete;

        SIXTRL_HOST_FN CudaArgument&
        operator=( CudaArgument const& rhs ) = delete;
        SIXTRL_HOST_FN CudaArgument& operator=( CudaArgument&& rhs ) = delete;

        SIXTRL_HOST_FN virtual ~CudaArgument();

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool hasCudaArgBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cuda_arg_buffer_t cudaArgBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cuda_const_arg_buffer_t
            cudaArgBuffer() const SIXTRL_NOEXCEPT;

        template< typename Ptr >
        SIXTRL_HOST_FN Ptr* cudaArgBufferAsPtr() SIXTRL_NOEXCEPT;

        template< typename Ptr >
        SIXTRL_HOST_FN Ptr cudaArgBufferAsPtr() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        cudaArgBufferAsCObjectsDataBegin() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        cudaArgBufferAsCObjectsDataBegin() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC debug_register_t*
        cudaArgBufferAsPtrDebugRegister() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC debug_register_t const*
        cudaArgBufferAsPtrDebugRegister() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        elem_by_elem_config_t*
        cudaArgBufferAsElemByElemByElemConfig() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        elem_by_elem_config_t const*
        cudaArgBufferAsElemByElemByElemConfig() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN ptr_cuda_controller_t
        cudaController() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN ptr_const_cuda_controller_t
        cudaController() const SIXTRL_NOEXCEPT;

        protected:

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

    SIXTRL_STATIC SIXTRL_HOST_FN CudaArgument const* asCudaArgument(
        SIXTRL_CXX_NAMESPACE::ArgumentBase const*
            SIXTRL_RESTRICT base_arg ) SIXTRL_NOEXCEPT;

    SIXTRL_STATIC SIXTRL_HOST_FN CudaArgument* asCudaArgument(
        SIXTRL_CXX_NAMESPACE::ArgumentBase*
            SIXTRL_RESTRICT base_arg ) SIXTRL_NOEXCEPT;
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

/* ************************************************************************* */
/* ****** Implementation and Definitions of inline template functions ****** */
/* ************************************************************************* */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_INLINE CudaArgument const* asCudaArgument(
        SIXTRL_CXX_NAMESPACE::ArgumentBase const*
            SIXTRL_RESTRICT base_arg ) SIXTRL_NOEXCEPT
    {
        CudaArgument const* ptr = nullptr;

        if( base_arg != nullptr )
        {
            ptr = base_arg->asDerivedArgument< CudaArgument >(
                SIXTRL_CXX_NAMESPACE::ARCHITECTURE_CUDA, true );
        }

        return ptr;
    }

    SIXTRL_INLINE CudaArgument* asCudaArgument(
        SIXTRL_CXX_NAMESPACE::ArgumentBase* SIXTRL_RESTRICT
            base_arg ) SIXTRL_NOEXCEPT
    {
        using base_arg_t = SIXTRL_CXX_NAMESPACE::ArgumentBase;
        using cuda_arg_t = SIXTRL_CXX_NAMESPACE::CudaArgument;

        base_arg_t const* cbase_arg_ptr = base_arg;
        return const_cast< cuda_arg_t* >( asCudaArgument( cbase_arg_ptr ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename Ptr >
    SIXTRL_INLINE Ptr* CudaArgument::cudaArgBufferAsPtr() SIXTRL_NOEXCEPT
    {
        return reinterpret_cast< Ptr >( this->cudaArgBuffer() );
    }

    template< typename Ptr >
    SIXTRL_INLINE Ptr CudaArgument::cudaArgBufferAsPtr() const SIXTRL_NOEXCEPT
    {
        return reinterpret_cast< Ptr >( this->cudaArgBuffer() );
    }
}

#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_ARGUMENT_H__ */

/* end: sixtracklib/cuda/argument.hpp */
