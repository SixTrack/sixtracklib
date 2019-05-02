#ifndef SIXTRACKLIB_COMMON_CONTROL_CONTROLLER_BASE_HPP__
#define SIXTRACKLIB_COMMON_CONTROL_CONTROLLER_BASE_HPP__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include <cstddef>
        #include <cstdlib>
        #include <string>
        #include <memory>
    #endif /* C++, host code */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"

    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include "sixtracklib/common/buffer.hpp"
        #include "sixtracklib/common/control/arch_base.hpp"
    #endif /* C++, host code */

    #include "sixtracklib/common/control/arch_base.h"
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( __CUDA_ARCH__ ) && !defined( _GPUCODE )

namespace SIXTRL_CXX_NAMESPACE
{
    class ArgumentBase;

    class ControllerBase : public SIXTRL_CXX_NAMESPACE::ArchBase
    {
        private:

        using _base_arch_obj_t = SIXTRL_CXX_NAMESPACE::ArchBase;

        public:

        using arch_id_t      = _base_arch_obj_t::arch_id_t;
        using status_t       = SIXTRL_CXX_NAMESPACE::controller_status_t;
        using success_flag_t = SIXTRL_CXX_NAMESPACE::controller_success_flag_t;
        using buffer_t       = SIXTRL_CXX_NAMESPACE::Buffer;
        using c_buffer_t     = buffer_t::c_api_t;
        using size_type      = buffer_t::size_type;

        using ptr_arg_base_t       = ArgumentBase*;
        using ptr_const_arg_base_t = ArgumentBase const*;

        static SIXTRL_CONSTEXPR_OR_CONST arch_id_t ILLEGAL_ARCH_ID =
            SIXTRL_CXX_NAMESPACE::ARCHITECTURE_ILLEGAL;

        static SIXTRL_CONSTEXPR_OR_CONST arch_id_t NO_ARCH_ID =
            SIXTRL_CXX_NAMESPACE::ARCHITECTURE_NONE;


        SIXTRL_HOST_FN bool usesNodes() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void clear();

        SIXTRL_HOST_FN bool readyForSend()    const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool readyForReceive() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool readyForRemap()   const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN status_t send( ptr_arg_base_t SIXTRL_RESTRICT dest,
            void const* SIXTRL_RESTRICT source, size_type const src_length );

        SIXTRL_HOST_FN status_t send( ptr_arg_base_t SIXTRL_RESTRICT dest,
            const c_buffer_t *const SIXTRL_RESTRICT source );

        SIXTRL_HOST_FN status_t send( ptr_arg_base_t SIXTRL_RESTRICT dest,
            buffer_t const& SIXTRL_RESTRICT_REF source );

        SIXTRL_HOST_FN status_t receive( void* SIXTRL_RESTRICT destination,
            size_type const destination_capacity,
            ptr_arg_base_t SIXTRL_RESTRICT source );

        SIXTRL_HOST_FN status_t receive( c_buffer_t* SIXTRL_RESTRICT dest,
            ptr_arg_base_t SIXTRL_RESTRICT source );

        SIXTRL_HOST_FN status_t receive( buffer_t& SIXTRL_RESTRICT_REF dest,
            ptr_arg_base_t SIXTRL_RESTRICT source );

        SIXTRL_HOST_FN status_t remapSentCObjectsBuffer(
            ptr_arg_base_t SIXTRL_RESTRICT arg,
            size_type const arg_size = size_type{ 0 } );

        SIXTRL_HOST_FN bool hasSuccessFlagArgument() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN ptr_arg_base_t ptrSuccessFlagArgument() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN ptr_const_arg_base_t
        ptrSuccessFlagArgument() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN success_flag_t lastSuccessFlagValue() const;
        SIXTRL_HOST_FN bool isInDebugMode() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN virtual ~ControllerBase() SIXTRL_NOEXCEPT;

        template< class Derived > Derived const* asDerivedController(
            arch_id_t const requ_arch_id,
            bool const exact_match_required = false ) const SIXTRL_NOEXCEPT;

        template< class Derived > Derived* asDerivedController(
            arch_id_t const requ_arch_id,
            bool const exact_match_required = false ) SIXTRL_NOEXCEPT;

        protected:

        using ptr_stored_base_argument_t = std::unique_ptr< ArgumentBase >;

        SIXTRL_HOST_FN explicit ControllerBase(
            arch_id_t const arch_id,
            const char *const SIXTRL_RESTRICT arch_str,
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN ControllerBase( ControllerBase const& other ) = default;
        SIXTRL_HOST_FN ControllerBase( ControllerBase&& other ) = default;

        SIXTRL_HOST_FN ControllerBase&
        operator=( ControllerBase const& rhs ) = default;

        SIXTRL_HOST_FN ControllerBase&
        operator=( ControllerBase&& rhs ) = default;

        SIXTRL_HOST_FN virtual void doClear();

        SIXTRL_HOST_FN virtual status_t doSend(
            ptr_arg_base_t SIXTRL_RESTRICT destination,
            const void *const SIXTRL_RESTRICT source,
            size_type const source_length );

        SIXTRL_HOST_FN virtual status_t doReceive(
            void* SIXTRL_RESTRICT destination, size_type const dest_capacity,
            ptr_arg_base_t SIXTRL_RESTRICT source );

        SIXTRL_HOST_FN virtual status_t doRemapSentCObjectsBuffer(
            ptr_arg_base_t SIXTRL_RESTRICT arg,
            size_type arg_size );

        SIXTRL_HOST_FN virtual success_flag_t
            doGetSuccessFlagValueFromArg() const;

        SIXTRL_HOST_FN virtual void doSetSuccessFlagValueFromArg(
            success_flag_t const success_flag );

        SIXTRL_HOST_FN void doSetUsesNodesFlag(
            bool const flag ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetReadyForSendFlag(
            bool const flag ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetReadyForReceiveFlag(
            bool const flag ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetReadyForRemapFlag(
            bool const flag ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetDebugModeFlag(
            bool const flag ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doUpdateStoredSuccessFlagArgument(
            ptr_stored_base_argument_t&& ptr_stored_arg ) SIXTRL_NOEXCEPT;

        private:

        ptr_stored_base_argument_t m_ptr_success_flag_arg;

        bool        m_uses_nodes;
        bool        m_ready_for_remap;
        bool        m_ready_for_send;
        bool        m_ready_for_receive;
        bool        m_debug_mode;
    };
}
#endif /* C++, host */

#if defined( __cplusplus ) && !defined( __CUDA_ARCH__ ) && !defined( _GPUCODE )

extern "C" { typedef SIXTRL_CXX_NAMESPACE::ControllerBase NS(ControllerBase); }

#else /* C++, host */

typedef void NS(ControllerBase);

#endif /* C++, host */


#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <type_traits>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    template< class Derived > Derived const*
    ControllerBase::asDerivedController(
        ControllerBase::arch_id_t const requ_arch_id,
        bool const exact_match_required ) const SIXTRL_NOEXCEPT
    {
        Derived const* ptr_derived = nullptr;

        using _this_t = SIXTRL_CXX_NAMESPACE::ControllerBase;

        static_assert( std::is_base_of< _this_t, Derived >::value,
                       "asDerivedController< Derived > requires Dervied to be "
                       "derived from SIXTRL_CXX_NAMESPACE::ControllerBase" );

        if( ( ( !exact_match_required ) &&
              ( this->isArchCompatibleWith( requ_arch_id ) ) ) ||
            ( this->isArchIdenticalTo( requ_arch_id ) ) )
        {
            ptr_derived = static_cast< Derived const* >( this );
        }

        return ptr_derived;
    }

    template< class Derived > Derived* ControllerBase::asDerivedController(
        ControllerBase::arch_id_t const requ_arch_id,
        bool const exact_match_required ) SIXTRL_NOEXCEPT
    {
        using _this_t = SIXTRL_CXX_NAMESPACE::ControllerBase;

        return const_cast< Derived* >( static_cast< _this_t const& >(
            *this ).asDerivedController< Derived >(
                requ_arch_id, exact_match_required ) );
    }
}

#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_CONTROL_CONTROLLER_BASE_HPP__ */

/* end: sixtracklib/common/control/controller_base.hpp */
