#ifndef SIXTRACKLIB_COMMON_CONTROL_ARCH_BASE_OBJECT_HPP__
#define SIXTRACKLIB_COMMON_CONTROL_ARCH_BASE_OBJECT_HPP__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include <cstddef>
        #include <cstdlib>
        #include <string>
    #endif /* C++, Host */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/control/arch_info.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    class ArchBase : public SIXTRL_CXX_NAMESPACE::ArchInfo
    {
        public:

        using arch_info_t = SIXTRL_CXX_NAMESPACE::ArchInfo;
        using arch_id_t   = arch_info_t::arch_id_t;
        using size_type   = arch_info_t::size_type;

        SIXTRL_HOST_FN explicit ArchBase( arch_id_t const arch_id,
            const char *const SIXTRL_RESTRICT arch_str = nullptr,
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN explicit ArchBase(
            arch_id_t const arch_id,
            std::string const& SIXTRL_RESTRICT arch_str,
            std::string const& SIXTRL_RESTRICT config_str = std::string{} );

        SIXTRL_HOST_FN ArchBase( ArchBase const& other ) = default;
        SIXTRL_HOST_FN ArchBase( ArchBase&& other ) = default;

        SIXTRL_HOST_FN ArchBase& operator=( ArchBase const& rhs ) = default;
        SIXTRL_HOST_FN ArchBase& operator=( ArchBase&& rhs ) = default;

        SIXTRL_HOST_FN virtual ~ArchBase() = default;

        template< class Derived > SIXTRL_HOST_FN Derived const* asDerived(
            arch_id_t const required_arch_id,
            bool requires_exact_match = false ) const SIXTRL_NOEXCEPT;

        template< class Derived > SIXTRL_HOST_FN Derived* asDerived(
            arch_id_t const required_arch_id,
            bool requires_exact_match = false ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasConfigStr() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN std::string const& configStr() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN char const* ptrConfigStr() const SIXTRL_NOEXCEPT;

        protected:

        SIXTRL_HOST_FN virtual bool doParseConfigStr(
            const char *const SIXTRL_RESTRICT config_str );

        SIXTRL_HOST_FN void doUpdateStoredConfigStr(
            const char *const SIXTRL_RESTRICT config_str );

        private:

        SIXTRL_HOST_FN bool doParseConfigStrArchBase(
            const char *const SIXTRL_RESTRICT config_str ) SIXTRL_NOEXCEPT;

        std::string m_config_str;
    };
}

typedef SIXTRL_CXX_NAMESPACE::ArchBase NS(ArchBase);

#else /* C++, Host */

typedef void NS(ArchBase);

#endif /* C++, Host */

/* ************************************************************************* */
/* **********  Implementation of template member functions      ************ */
/* ************************************************************************* */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <type_traits>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    template< class Derived > Derived const* ArchBase::asDerived(
        ArchBase::arch_id_t const required_arch_id,
        bool requires_exact_match ) const SIXTRL_NOEXCEPT
    {
        Derived const* ptr_derived = nullptr;

        static_assert( std::is_base_of< ArchBase, Derived >::value,
                       "asDerived< Derived > requires Dervied to be derived "
                       "from SIXTRL_CXX_NAMESPACE::ArchBase" );

        if( ( ( !requires_exact_match ) &&
              ( this->isArchCompatibleWith( required_arch_id ) ) ) ||
            ( this->isArchIdenticalTo( required_arch_id ) ) )
        {
            ptr_derived = static_cast< Derived const* >( this );
        }

        return ptr_derived;
    }

    template< class Derived > Derived* ArchBase::asDerived(
        ArchBase::arch_id_t const required_arch_id,
        bool requires_exact_match ) SIXTRL_NOEXCEPT
    {
        return const_cast< Derived* >( static_cast< ArchBase const& >(
            *this ).asDerived< Derived >(
                required_arch_id, requires_exact_match ) );
    }
}

#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_CONTROL_ARCH_BASE_OBJECT_HPP__ */

/* end: sixtracklib/common/control/arch_base.hpp */
