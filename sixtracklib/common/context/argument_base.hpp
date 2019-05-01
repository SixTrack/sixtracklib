#ifndef SIXTRACKLIB_COMMON_CONTEXT_ARGUMENT_BASE_HPP__
#define SIXTRACKLIB_COMMON_CONTEXT_ARGUMENT_BASE_HPP__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if  defined( __cplusplus ) && !defined( _GPUCODE ) && \
        !defined( __CUDA_ARCH__ )
        #include <cstddef>
        #include <cstdlib>
    #endif /* C++, host */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/context/definitions.h"

    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
        !defined( __CUDA_ARCH__ )
        #include "sixtracklib/common/buffer.hpp"
        #include "sixtracklib/common/arch_base.hpp"
    #endif /* C++, host */

    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( __CUDA_ARCH__ ) && !defined( _GPUCODE )

namespace SIXTRL_CXX_NAMESPACE
{
    class ContextBase;

    class ArgumentBase : public SIXTRL_CXX_NAMESPACE::ArchBase
    {
        private:
        using _arch_base_t  = SIXTRL_CXX_NAMESPACE::ArchBase;

        public:

        using arch_id_t   = _arch_base_t::arch_id_t;
        using status_t    = SIXTRL_CXX_NAMESPACE::context_status_t;
        using buffer_t    = SIXTRL_CXX_NAMESPACE::Buffer;
        using c_buffer_t  = buffer_t::c_api_t;
        using size_type   = buffer_t::size_type;

        using ptr_base_context_t       = ContextBase*;
        using ptr_const_base_context_t = ContextBase const*;

        SIXTRL_HOST_FN virtual ~ArgumentBase() = default;

        SIXTRL_HOST_FN NS(context_status_t) send();

        SIXTRL_HOST_FN NS(context_status_t) send(
            buffer_t const& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_HOST_FN NS(context_status_t) send(
            const c_buffer_t *const SIXTRL_RESTRICT ptr_c_buffer );

        SIXTRL_HOST_FN NS(context_status_t) send(
            void const* SIXTRL_RESTRICT arg_begin,
            size_type const arg_size );

        SIXTRL_HOST_FN NS(context_status_t) receive();

        SIXTRL_HOST_FN NS(context_status_t) receive(
            buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_HOST_FN NS(context_status_t) receive(
            c_buffer_t* SIXTRL_RESTRICT ptr_c_buffer );

        SIXTRL_HOST_FN NS(context_status_t) receive(
            void* SIXTRL_RESTRICT arg_begin,
            size_type const arg_capacity );

        SIXTRL_HOST_FN bool usesCObjectsCxxBuffer() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN buffer_t* ptrCObjectsCxxBuffer() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN buffer_t& cobjectsCxxBuffer() const;

        SIXTRL_HOST_FN bool usesCObjectsBuffer() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN c_buffer_t* ptrCObjectsBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool usesRawArgument() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN void* ptrRawArgument() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type size() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN size_type capacity() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasArgumentBuffer() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool requiresArgumentBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN ptr_base_context_t
        ptrBaseContext() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN ptr_const_base_context_t
        ptrBaseContext() const SIXTRL_NOEXCEPT;

        template< class Derived > SIXTRL_HOST_FN Derived const*
        asDerivedArgument( arch_id_t const required_arch_id,
            bool requires_exact_match = false ) const SIXTRL_NOEXCEPT;

        template< class Derived > SIXTRL_HOST_FN Derived* asDerivedArgument(
            arch_id_t const required_arch_id,
            bool requires_exact_match = false ) SIXTRL_NOEXCEPT;

        protected:

        SIXTRL_HOST_FN ArgumentBase(
            arch_id_t const type_id, const char *const arch_str,
            const char *const SIXTRL_RESTRICT config_str = nullptr,
            bool const needs_argument_buffer = true,
            ptr_base_context_t SIXTRL_RESTRICT context = nullptr );

        SIXTRL_HOST_FN ArgumentBase( ArgumentBase const& other ) = default;
        SIXTRL_HOST_FN ArgumentBase( ArgumentBase&& other ) = default;

        SIXTRL_HOST_FN ArgumentBase&
        operator=( ArgumentBase const& other ) = default;

        SIXTRL_HOST_FN ArgumentBase&
        operator=( ArgumentBase&& other ) = default;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN virtual bool doReserveArgumentBuffer(
            size_type const required_buffer_size );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void doSetArgSize(
            size_type const arg_size ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetArgCapacity(
            size_type const arg_capacity ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetPtrContext(
            ptr_base_context_t SIXTRL_RESTRICT ptr_context ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetBufferRef(
            buffer_t const& SIXTRL_RESTRICT_REF buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doResetPtrCxxBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetPtrCBuffer(
            const c_buffer_t *const SIXTRL_RESTRICT buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetPtrRawArgument(
            const void *const SIXTRL_RESTRICT raw_arg_begin ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetHasArgumentBufferFlag(
            bool const is_available ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetNeedsArgumentBufferFlag(
            bool const needs_argument_buffer ) SIXTRL_NOEXCEPT;

        private:

        mutable void*           m_ptr_raw_arg_begin;
        mutable buffer_t*       m_ptr_cobj_cxx_buffer;
        mutable c_buffer_t*     m_ptr_cobj_c99_buffer;

        ptr_base_context_t      m_ptr_base_context;

        size_type               m_arg_size;
        size_type               m_arg_capacity;

        bool                    m_needs_arg_buffer;
        bool                    m_has_arg_buffer;

    };
}

#endif /* C++, host */

#if defined( __cplusplus ) && !defined( __CUDA_ARCH__ ) && !defined( _GPUCODE )

extern "C" { typedef SIXTRL_CXX_NAMESPACE::ArgumentBase  NS(ArgumentBase); }

#else /* C++, host */

typedef void NS(ArgumentBase);

#endif /* C++, host */



#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <type_traits>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    template< class Derived > Derived const* ArgumentBase::asDerivedArgument(
        ArgumentBase::arch_id_t const required_arch_id,
        bool requires_exact_match ) const SIXTRL_NOEXCEPT
    {
        Derived const* ptr_derived = nullptr;

        static_assert( std::is_base_of< ArgumentBase, Derived >::value,
                       "asDerivedArgument< Derived > requires Dervied to be "
                       "derived from SIXTRL_CXX_NAMESPACE::ArgumentBase" );

        if( ( ( !requires_exact_match ) &&
              ( this->isCompatibleWith( required_arch_id ) ) ) ||
            ( this->isIdenticalTo( required_arch_id ) ) )
        {
            ptr_derived = static_cast< Derived const* >( this );
        }

        return ptr_derived;
    }

    template< class Derived > Derived* ArgumentBase::asDerivedArgument(
        ArgumentBase::arch_id_t const required_arch_id,
        bool requires_exact_match ) SIXTRL_NOEXCEPT
    {
        return const_cast< Derived* >( static_cast< ArgumentBase const& >(
            *this ).asDerivedArgument< Derived >(
                required_arch_id, requires_exact_match );
    }
}

#endif /* C++, Host */


#endif /* SIXTRACKLIB_COMMON_CONTEXT_ARGUMENT_BASE_HPP__ */
/* end: sixtracklib/common/context/argument_base.hpp */
