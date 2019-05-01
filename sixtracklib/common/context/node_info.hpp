#ifndef SIXTRACKLIB_COMMON_CONTEXT_NODE_INFO_BASE_HPP__
#define SIXTRACKLIB_COMMON_CONTEXT_NODE_INFO_BASE_HPP__

#if !defined( SIXTRKL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include <cstddef>
        #include <cstdlib>
        #include <cstring>
        #include <cstdio>
    #endif /* C++, Host */
#endif /* !defined( SIXTRKL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/context/definitions.h"
    #include "sixtracklib/common/context/node_id.hpp"
    #include "sixtracklib/common/context/arch_info.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

namespace SIXTRL_CXX_NAMESPACE
{
    class NodeInfoBase : public SIXTRL_CXX_NAMESPACE::ArchInfo
    {
        public:

        using node_id_t     = SIXTRL_CXX_NAMESPACE::NodeId;
        using platform_id_t = node_id_t::platform_id_t;
        using device_id_t   = node_id_t::device_id_t;
        using node_index_t  = node_id_t::index_t;
        using size_type     = node_id_t::size_type;

        using arch_info_t   = SIXTRL_CXX_NAMESPACE::ArchInfo;
        using arch_id_t     = arch_info_t::arch_id_t;

        SIXTRL_HOST_FN explicit NodeInfoBase(
            arch_id_t const arch_id,
            const char *const SIXTRL_RESTRICT arch_str,
            platform_id_t const platform_id =
                SIXTRL_CXX_NAMESPACE::NODE_ILLEGAL_PATFORM_ID,
            device_id_t const device_id =
                SIXTRL_CXX_NAMESPACE::NODE_ILLEGAL_DEVICE_ID,
            const char *const SIXTRL_RESTRICT platform_name = nullptr,
            const char *const SIXTRL_RESTRICT device_name = nullptr,
            const char *const SIXTRL_RESTRICT description = nullptr );

        SIXTRL_HOST_FN expicit NodeInfoBase(
            arch_info_t const* SIXTRL_RESTRICT ptr_arch_info,
            node_id_t const* SIXTRL_RESTRICT node_id = nullptr,
            const char *const SIXTRL_RESTRICT platform_name = nullptr,
            const char *const SIXTRL_RESTRICT device_name = nullptr,
            const char *const SIXTRL_RESTRICT description = nullptr );

        SIXTRL_HOST_FN expicit NodeInfoBase(
            arch_info_t const& SIXTRL_RESTRICT_REF arch_info,
            node_id_t const& SIXTRL_RESTRICT_REF node_id,
            std::string const& SIXTRL_RESTRICT_REF
                platform_name = std::string{},
            std::string const& SIXTRL_RESTRICT_REF
                device_name = std::string{},
            std::string const& SIXTRL_RESTRICT_REF
                description = std::string{} );

        SIXTRL_HOST_FN NodeInfoBase(
            arch_id_t const arch_id,
            std::string const& SIXTRL_RESTRICT_REF arch_str,
            platform_id_t const platform_id, device_id_t const device_id,
            std::string const& SIXTRL_RESTRICT_REF platform_name,
            std::string const& SIXTRL_RESTRICT_REF device_name,
            std::string const& SIXTRL_RESTRICT_REF description );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN NodeInfoBase( NodeInfoBase const& other ) = default;
        SIXTRL_HOST_FN NodeInfoBase( NodeInfoBase&& other ) = default;

        SIXTRL_HOST_FN NodeInfoBase& operator=( NodeInfoBase&& rhs ) = default;

        SIXTRL_HOST_FN NodeInfoBase&
        operator=( NodeInfoBase const& rhs ) = default;

        SIXTRL_HOST_FN virtual ~NodeInfoBase() = default;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN node_id_t const& nodeId() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN node_id_t const* ptrNodeId() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_platform_id_t platformId() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN node_device_id_t deviceId() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool hasPlatformName() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN std::string const& platformName() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN char const* ptrPlatformNameStr() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasDeviceName() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN std::string const& deviceName() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN char const* ptrDeviceNameStr() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasDescription() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN std::string const& description() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN char const* ptrDescriptionStr() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        using ArchInfo::reset;

        SIXTRL_HOST_FN void reset( arch_id_t const arch_id,
            const char *const SIXTRL_RESTRICT arch_str,
            platform_id_t const platform_id, device_id_t const device_id,
            const char *const SIXTRL_RESTRICT platform_name,
            const char *const SIXTRL_RESTRICT device_name,
            const char *const SIXTRL_RESTRICT description );

        SIXTRL_HOST_FN void reset( arch_id_t const arch_id,
            std::string const& SIXTRL_RESTRICT_REF arch_str,
            platform_id_t const platform_id, device_id_t const device_id,
            std::string const& SIXTRL_RESTRICT_REF platform_name,
            std::string const& SIXTRL_RESTRICT_REF device_name,
            std::string const& SIXTRL_RESTRICT_REF description );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN friend std::ostream& operator<<(
            std::ostream& SIXTRL_RESTRICT_REF output,
            NodeInfoBase const& SIXTRL_RESTRICT_REF node_info );

        SIXTRL_HOST_FN void print( ::FILE* SIXTRL_RESTRICT output ) const;

        SIXTRL_HOST_FN void printOut() const;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void markAsDefault(
            bool const is_default = true ) SIXTRL_NOEXCEPT;

        template< class Derived > SIXTRL_HOST_FN
        Derived const* asDerivedNodeInfo(
            arch_id_t const required_arch_id,
            bool requires_exact_match = false ) const SIXTRL_NOEXCEPT;

        template< class Derived > SIXTRL_HOST_FN Derived* asDerivedNodeInfo(
            arch_id_t const required_arch_id,
            bool requires_exact_match = false ) SIXTRL_NOEXCEPT;

        protected:

        SIXTRL_HOST_FN virtual void doPrintToOutputStream(
            std::ostream& SIXTRL_RESTRICT_REF output );

        SIXTRL_HOST_FN void doSetNodeId(
            node_id_t const& SIXTRL_RESTRICT_REF node_id );

        SIXTRL_HOST_FN void doSetPlatformName(
            const char *const SIXTRL_RESTRICT platform_name );

        SIXTRL_HOST_FN void doSetDeviceName(
            const char *const SIXTRL_RESTRICT device_name );

        SIXTRL_HOST_FN void doSetDescription(
            const char *const SIXTRL_RESTRICT description );

        private:

        std::string     m_platform_name;
        std::string     m_device_name;
        std::string     m_description;
        node_id_t       m_node_id;
        bool            m_is_default_node;
    };
}

typedef SIXTRL_CXX_NAMESPACE::NodeInfoBase  NS(NodeInfoBase);

#else /* C++, Host */

typedef void NS(NodeInfoBase);

#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <type_traits>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    template< class Derived > Derived const* NodeInfoBase::asDerivedNodeInfo(
        NodeInfoBase::arch_id_t const required_arch_id,
        bool requires_exact_match ) const SIXTRL_NOEXCEPT
    {
        Derived const* ptr_derived = nullptr;

        static_assert( std::is_base_of< NodeInfoBase, Derived >::value,
                       "asDerivedNodeInfo< Derived > requires Dervied to be
                       "derived from SIXTRL_CXX_NAMESPACE::NodeInfoBase" );

        if( ( ( !requires_exact_match ) &&
              ( this->isCompatibleWith( required_arch_id ) ) ) ||
            ( this->isIdenticalTo( required_arch_id ) ) )
        {
            ptr_derived = static_cast< Derived const* >( this );
        }

        return ptr_derived;
    }

    template< class Derived > Derived* NodeInfoBase::asDerivedNodeInfo(
        NodeInfoBase::arch_id_t const required_arch_id,
        bool requires_exact_match ) SIXTRL_NOEXCEPT
    {
        return const_cast< Derived* >( static_cast< NodeInfoBase const& >(
            *this ).asDerivedNodInfo< Derived >(
                required_arch_id, requires_exact_match );
    }
}

#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_CONTEXT_NODE_INFO_BASE_HPP__ */
/* end: sixtracklib/common/context/node_info.hpp */
