#ifndef SIXTRACKLIB_COMMON_CONTEXT_ARGUMENT_BASE_H__
#define SIXTRACKLIB_COMMON_CONTEXT_ARGUMENT_BASE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <cstddef>
        #include <cstdlib>
    #endif /* defined( __cplusplus ) */

    #include <stdint.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"

    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* defined( __cplusplus ) */

    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/context/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    class ContextBase;

    class ArgumentBase
    {
        public:

        using type_id_t   = SIXTRL_CXX_NAMESPACE::context_type_id_t;
        using status_t    = SIXTRL_CXX_NAMESPACE::context_status_t;
        using buffer_t    = SIXTRL_CXX_NAMESPACE::Buffer;
        using c_buffer_t  = ::NS(Buffer);
        using size_type   = buffer_t::size_type;

        using ptr_base_context_t       = ContextBase*;
        using ptr_const_base_context_t = ContextBase const*;

        SIXTRL_HOST_FN virtual ~ArgumentBase() = default;

        SIXTRL_HOST_FN bool send();

        SIXTRL_HOST_FN bool send(
            buffer_t const& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_HOST_FN bool send(
            const c_buffer_t *const SIXTRL_RESTRICT ptr_c_buffer );

        SIXTRL_HOST_FN bool send(
            void const* SIXTRL_RESTRICT arg_begin,
            size_type const arg_size );

        SIXTRL_HOST_FN bool receive();

        SIXTRL_HOST_FN bool receive(
            buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_HOST_FN bool receive(
            c_buffer_t* SIXTRL_RESTRICT ptr_c_buffer );

        SIXTRL_HOST_FN bool receive(
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

        SIXTRL_HOST_FN type_id_t type() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN std::string const& typeStr() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN char const* ptrTypeStr() const SIXTRL_NOEXCEPT;

        protected:

        SIXTRL_HOST_FN ArgumentBase(
            type_id_t const type_id, const char *const type_id_str,
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

        SIXTRL_HOST_FN void doSetTypeId(
            type_id_t const type_id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetTypeIdStr(
            const char *const SIXTRL_RESTRICT type_id_str ) SIXTRL_NOEXCEPT;

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

        std::string             m_type_id_str;
        mutable void*           m_ptr_raw_arg_begin;
        mutable buffer_t*       m_ptr_cobj_cxx_buffer;
        mutable c_buffer_t*     m_ptr_cobj_c99_buffer;

        ptr_base_context_t           m_ptr_base_context;

        size_type               m_arg_size;
        size_type               m_arg_capacity;
        type_id_t               m_type_id;

        bool                    m_needs_arg_buffer;
        bool                    m_has_arg_buffer;

    };
}

typedef SIXTRL_CXX_NAMESPACE::ArgumentBase  NS(ArgumentBase);

#else /* !defined( __cplusplus ) */

typedef void NS(ArgumentBase);

#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_CONTEXT_ARGUMENT_BASE_H__ */
/* end: sixtracklib/common/context/argument_base.h */
