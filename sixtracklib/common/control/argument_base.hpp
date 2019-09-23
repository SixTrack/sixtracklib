#ifndef SIXTRACKLIB_COMMON_CONTROL_ARGUMENT_BASE_HPP__
#define SIXTRACKLIB_COMMON_CONTROL_ARGUMENT_BASE_HPP__

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )
#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
        #include <cstddef>
        #include <cstdlib>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */
#endif /* C++, host */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer.hpp"
    #include "sixtracklib/common/control/arch_base.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    class ControllerBase;

    class ArgumentBase : public SIXTRL_CXX_NAMESPACE::ArchBase
    {
        private:
        using _arch_base_t  = SIXTRL_CXX_NAMESPACE::ArchBase;

        public:

        using status_t    = SIXTRL_CXX_NAMESPACE::arch_status_t;
        using buffer_t    = SIXTRL_CXX_NAMESPACE::Buffer;
        using c_buffer_t  = buffer_t::c_api_t;

        using perform_remap_flag_t =
            SIXTRL_CXX_NAMESPACE::ctrl_perform_remap_flag_t;

        using ptr_base_controller_t       = ControllerBase*;
        using ptr_const_base_controller_t = ControllerBase const*;

        static SIXTRL_CONSTEXPR_OR_CONST status_t
            STATUS_SUCCESS = SIXTRL_CXX_NAMESPACE::ARCH_STATUS_SUCCESS;

        static SIXTRL_CONSTEXPR_OR_CONST status_t STATUS_GENERAL_FAILURE =
                SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;

        SIXTRL_HOST_FN virtual ~ArgumentBase() = default;

        SIXTRL_HOST_FN status_t send(
            perform_remap_flag_t const perform_remap_flag =
                SIXTRL_CXX_NAMESPACE::CTRL_PERFORM_REMAP );

        SIXTRL_HOST_FN status_t send( buffer_t const& SIXTRL_RESTRICT_REF buf,
            perform_remap_flag_t const perform_remap_flag =
                SIXTRL_CXX_NAMESPACE::CTRL_PERFORM_REMAP );

        SIXTRL_HOST_FN status_t send(
            const c_buffer_t *const SIXTRL_RESTRICT ptr_c_buffer,
            perform_remap_flag_t const perform_remap_flag =
                SIXTRL_CXX_NAMESPACE::CTRL_PERFORM_REMAP );

        SIXTRL_HOST_FN status_t send( void const* SIXTRL_RESTRICT arg_begin,
            size_type const arg_size );


        SIXTRL_HOST_FN status_t receive(
            perform_remap_flag_t const perform_remap_flag =
                SIXTRL_CXX_NAMESPACE::CTRL_PERFORM_REMAP );

        SIXTRL_HOST_FN status_t receive(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            perform_remap_flag_t const perform_remap_flag =
                SIXTRL_CXX_NAMESPACE::CTRL_PERFORM_REMAP );

        SIXTRL_HOST_FN status_t receive(
            c_buffer_t* SIXTRL_RESTRICT ptr_c_buffer,
            perform_remap_flag_t const perform_remap_flag =
                SIXTRL_CXX_NAMESPACE::CTRL_PERFORM_REMAP );

        SIXTRL_HOST_FN status_t receive( void* SIXTRL_RESTRICT arg_begin,
            size_type const arg_capacity );

        SIXTRL_HOST_FN status_t updateRegion(
            size_type const offset, size_type const length,
            void const* SIXTRL_RESTRICT_REF new_value );

        SIXTRL_HOST_FN status_t updateRegions(
            size_type const num_regions_to_update,
            size_type const* SIXTRL_RESTRICT offsets,
            size_type const* SIXTRL_RESTRICT lengths,
            void const* SIXTRL_RESTRICT const* new_values );

        SIXTRL_HOST_FN status_t remap();

        SIXTRL_HOST_FN bool usesCObjectsCxxBuffer() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN buffer_t* ptrCObjectsCxxBuffer() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN buffer_t& cobjectsCxxBuffer() const;

        SIXTRL_HOST_FN bool usesCObjectsBuffer() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN c_buffer_t* ptrCObjectsBuffer() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN size_type cobjectsBufferSlotSize() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool usesRawArgument() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN void* ptrRawArgument() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type size() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN size_type capacity() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasArgumentBuffer() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool requiresArgumentBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN ptr_base_controller_t
        ptrControllerBase() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN ptr_const_base_controller_t
        ptrControllerBase() const SIXTRL_NOEXCEPT;

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
            ptr_base_controller_t SIXTRL_RESTRICT controller = nullptr );

        SIXTRL_HOST_FN ArgumentBase( ArgumentBase const& other ) = default;
        SIXTRL_HOST_FN ArgumentBase( ArgumentBase&& other ) = default;

        SIXTRL_HOST_FN ArgumentBase&
        operator=( ArgumentBase const& other ) = default;

        SIXTRL_HOST_FN ArgumentBase&
        operator=( ArgumentBase&& other ) = default;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN virtual bool doReserveArgumentBuffer(
            size_type const required_buffer_size );

        SIXTRL_HOST_FN virtual status_t doUpdateRegions(
            size_type const num_regions_to_update,
            size_type const* SIXTRL_RESTRICT offsets,
            size_type const* SIXTRL_RESTRICT lengths,
            void const* SIXTRL_RESTRICT const* SIXTRL_RESTRICT new_values );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void doSetArgSize(
            size_type const arg_size ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetArgCapacity(
            size_type const arg_capacity ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetPtrControllerBase(
            ptr_base_controller_t SIXTRL_RESTRICT ptr_ctrl ) SIXTRL_NOEXCEPT;

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

        ptr_base_controller_t   m_ptr_base_controller;

        size_type               m_arg_size;
        size_type               m_arg_capacity;

        bool                    m_needs_arg_buffer;
        bool                    m_has_arg_buffer;

    };
}

#endif /* C++, host */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++ */

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

typedef SIXTRL_CXX_NAMESPACE::ArgumentBase  NS(ArgumentBase);

#else /* C++, host */

typedef void NS(ArgumentBase);

#endif /* C++, host */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++ */

/* ************************************************************************* */
/* ********     Inline and template function implementation     ************ */
/* ************************************************************************* */

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <type_traits>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_INLINE ArgumentBase::status_t ArgumentBase::updateRegion(
        ArgumentBase::size_type const offset,
        ArgumentBase::size_type const size,
        void const* SIXTRL_RESTRICT new_value )
    {
        return this->doUpdateRegions( ArgumentBase::size_type{ 1 },
            &offset, &size, &new_value );
    }

    SIXTRL_INLINE ArgumentBase::status_t ArgumentBase::updateRegions(
        ArgumentBase::size_type const num_regions_to_update,
        ArgumentBase::size_type const* SIXTRL_RESTRICT offsets,
        ArgumentBase::size_type const* SIXTRL_RESTRICT sizes,
        void const* SIXTRL_RESTRICT const* SIXTRL_RESTRICT new_values )
    {
        return this->doUpdateRegions(
            num_regions_to_update, offsets, sizes, new_values );
    }

    template< class Derived > Derived const* ArgumentBase::asDerivedArgument(
        ArgumentBase::arch_id_t const required_arch_id,
        bool requires_exact_match ) const SIXTRL_NOEXCEPT
    {
        Derived const* ptr_derived = nullptr;

        static_assert( std::is_base_of< ArgumentBase, Derived >::value,
                       "asDerivedArgument< Derived > requires Dervied to be "
                       "derived from SIXTRL_CXX_NAMESPACE::ArgumentBase" );

        if( ( ( !requires_exact_match ) &&
              ( this->isArchCompatibleWith( required_arch_id ) ) ) ||
            ( this->isArchIdenticalTo( required_arch_id ) ) )
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
                required_arch_id, requires_exact_match ) );
    }
}

#endif /* C++, Host */


#endif /* SIXTRACKLIB_COMMON_CONTROL_ARGUMENT_BASE_HPP__ */
/* end: sixtracklib/common/control/argument_base.hpp */
