#ifndef SIXTRACKLIB_COMMON_CONTROL_KERNEL_CONFIG_BASE_HPP_
#define SIXTRACKLIB_COMMON_CONTROL_KERNEL_CONFIG_BASE_HPP_

#if !defined( SIXTRKL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include <cstddef>
        #include <cstdlib>
        #include <cstring>
        #include <cstdio>
        #include <ostream>
        #include <string>
    #endif /* C++, Host */
#endif /* !defined( SIXTRKL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/arch_base.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

namespace SIXTRL_CXX_NAMESPACE
{
    class KernelConfigBase : public SIXTRL_CXX_NAMESPACE::ArchBase
    {
        private:

        using _arch_base_t = SIXTRL_CXX_NAMESPACE::ArchBase;

        public:

        using arch_id_t   = _arch_base_t::arch_id_t;
        using size_type   = _arch_base_t::size_type;
        using kernel_id_t = SIXTRL_CXX_NAMESPACE::arch_kernel_id_t;

        static constexpr size_type MAX_WORK_ITEMS_DIM  = size_type{ 3 };
        static constexpr size_type MAX_WORK_GROUPS_DIM = size_type{ 3 };

        static constexpr size_type DEFAULT_WORK_ITEMS_DIM  = size_type{ 1 };
        static constexpr size_type DEFAULT_WORK_GROUPS_DIM = size_type{ 1 };

        static constexpr kernel_id_t ILLEGAL_KERNEL_ID =
            SIXTRL_CXX_NAMESPACE::ARCH_ILLEGAL_KERNEL_ID;

        SIXTRL_HOST_FN explicit KernelConfigBase(
            arch_id_t const arch_id,
            char const* SIXTRL_RESTRICT arch_str,
            char const* SIXTRL_RESTRICT config_str = nullptr,
            size_type const work_items_dim = size_type{ 1 },
            size_type const work_groups_dim = size_type{ 1 } );

        SIXTRL_HOST_FN KernelConfigBase(
            KernelConfigBase const& other );

        SIXTRL_HOST_FN KernelConfigBase(
            KernelConfigBase&& other ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN KernelConfigBase& operator=(
            KernelConfigBase const& rhs );

        SIXTRL_HOST_FN KernelConfigBase& operator=(
            KernelConfigBase&& rhs ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN virtual ~KernelConfigBase() = default;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool hasKernelId() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN kernel_id_t kernelId() const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN void setKernelId(
            kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool hasName() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN std::string const& name() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN char const* ptrNameStr() const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN void setName(
            std::string const& SIXTRL_RESTRICT_REF kernel_name );

        SIXTRL_HOST_FN void setName(
            char const* SIXTRL_RESTRICT kernel_name );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN size_type numArguments() const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN void setNumArguments(
            size_type const num_kernel_args ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN size_type numWorkItems(
            size_type const index ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type workItemsDim() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN size_type totalNumWorkItems() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type const*
        workItemsBegin() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type const*
        workItemsEnd() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type* workItemsBegin() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN size_type* workItemsEnd() SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN bool setNumWorkItems(
            size_type const work_items_a ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool setNumWorkItems(
            size_type const work_items_a,
            size_type const work_items_b ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool setNumWorkItems(
            size_type const work_items_a, size_type const work_items_b,
                size_type const work_item_c ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool setNumWorkItems(
            size_type const work_items_dim,
            size_type const* SIXTRL_RESTRICT work_itms_begin ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN size_type workItemOffset(
            size_type const index ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type const*
        workItemOffsetsBegin() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type const*
        workItemOffsetsEnd() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type* workItemOffsetsBegin() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN size_type* workItemOffsetsEnd() SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN bool setWorkItemOffset(
            size_type const offset_a ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool setWorkItemOffset(
            size_type const offset_a,
            size_type const offset_b ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool setWorkItemOffset(
            size_type const offset_a, size_type const offset_b,
                size_type const offset_c ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool setWorkItemOffset(
            size_type const offset_dim,
            size_type const* SIXTRL_RESTRICT offsets_begin ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN size_type workGroupSize(
            size_type const index ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type workGroupsDim() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type const*
        workGroupSizesBegin() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type const*
        workGroupSizesEnd() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type* workGroupSizesBegin() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN size_type* workGroupSizesEnd() SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN bool setWorkGroupSizes(
            size_type const work_groups_a ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool setWorkGroupSizes( size_type const work_groups_a,
                size_type const  work_groups_b ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool setWorkGroupSizes(
            size_type const work_groups_a, size_type const work_groups_b,
                size_type const work_groups_c ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool setWorkGroupSizes( size_type const work_groups_dim,
            size_type const* SIXTRL_RESTRICT work_grps_begin ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN size_type preferredWorkGroupMultiple(
            size_type const index ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type const*
        preferredWorkGroupMultiplesBegin() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type const*
        preferredWorkGroupMultiplesEnd() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type*
        preferredWorkGroupMultiplesBegin() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type*
        preferredWorkGroupMultiplesEnd() SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN bool setPreferredWorkGroupMultiple(
            size_type const work_groups_a ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool setPreferredWorkGroupMultiple(
            size_type const work_groups_a,
            size_type const work_groups_b ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool setPreferredWorkGroupMultiple(
            size_type const work_groups_a, size_type const work_groups_b,
                size_type const work_groups_c ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool setPreferredWorkGroupMultiple(
            size_type const work_groups_dim,
            size_type const* SIXTRL_RESTRICT pref_work_groups_multiple
        ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void clear();

        SIXTRL_HOST_FN void reset( size_type work_items_dim,
            size_type work_groups_dim );

        SIXTRL_HOST_FN bool needsUpdate() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool update();

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void print(
            std::ostream& SIXTRL_RESTRICT_REF output ) const;

        SIXTRL_HOST_FN void print( ::FILE* SIXTRL_RESTRICT output ) const;
        SIXTRL_HOST_FN void printOut() const;

        /* ----------------------------------------------------------------- */

        template< class Derived >
        SIXTRL_HOST_FN Derived const* asDerivedKernelConfig(
            arch_id_t const required_arch_id,
            bool const requires_exact_match = false ) const SIXTRL_NOEXCEPT;

        template< class Derived >
        SIXTRL_HOST_FN Derived* asDerivedKernelConfig(
            arch_id_t const required_arch_id,
            bool const requires_exact_match = false ) SIXTRL_NOEXCEPT;

        protected:

        virtual void doPrintToOutputStream(
            std::ostream& SIXTRL_RESTRICT_REF output ) const;

        virtual void doReset( size_type work_items_dim,
            size_type work_groups_dim );

        virtual bool doUpdate();

        virtual void doClear();

        SIXTRL_HOST_FN void doSetNumWorkItemsValue(
            size_type const index, size_type const value ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetWorkItemOffsetValue(
            size_type const index, size_type const value ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetWorkGroupSizeValue(
            size_type const index, size_type const value ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetPreferredWorkGroupMultiple(
            size_type const index, size_type const value ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetNeedsUpdateFlag(
            bool const needs_update = true ) SIXTRL_NOEXCEPT;

        private:

        SIXTRL_HOST_FN void doClearBaseImpl() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doResetBaseImpl( size_type work_items_dim,
            size_type work_groups_dim ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doPerformWorkItemsCopyBaseImpl(
            KernelConfigBase const& other ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doPerformWorkGroupsCopyBaseImpl(
            KernelConfigBase const& other ) SIXTRL_NOEXCEPT;

        std::string m_name;

        size_type   m_work_items[ MAX_WORK_ITEMS_DIM ];
        size_type   m_work_item_offsets[ MAX_WORK_ITEMS_DIM ];
        size_type   m_work_groups[ MAX_WORK_GROUPS_DIM ];
        size_type   m_pref_work_groups_multiple[ MAX_WORK_GROUPS_DIM ];

        size_type   m_work_items_dim;
        size_type   m_work_groups_dim;
        size_type   m_num_kernel_args;
        kernel_id_t m_kernel_id;

        bool        m_needs_update;
    };

    /* ----------------------------------------------------------------- */

    std::ostream& operator<<( std::ostream& SIXTRL_RESTRICT_REF output,
        KernelConfigBase const& SIXTRL_RESTRICT_REF kernel_config );
}

typedef SIXTRL_CXX_NAMESPACE::KernelConfigBase NS(KernelConfigBase);

#else /* C++, Host */

typedef void NS(KernelConfigBase);

#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <type_traits>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    template< class Derived >
    Derived const* KernelConfigBase::asDerivedKernelConfig(
        KernelConfigBase::arch_id_t const required_arch_id,
        bool requires_exact_match ) const SIXTRL_NOEXCEPT
    {
        Derived const* ptr_derived = nullptr;

        static_assert( std::is_base_of< KernelConfigBase, Derived >::value,
            "asDerivedKernelConfig< Derived > requires Derived to "
            "be derived from SIXTRL_CXX_NAMESPACE::KernelConfigBase" );

        if( ( ( !requires_exact_match ) &&
              ( this->isArchCompatibleWith( required_arch_id ) ) ) ||
            ( this->isArchIdenticalTo( required_arch_id ) ) )
        {
            ptr_derived = static_cast< Derived const* >( this );
        }

        return ptr_derived;
    }

    template< class Derived > Derived* KernelConfigBase::asDerivedKernelConfig(
        KernelConfigBase::arch_id_t const required_arch_id,
        bool requires_exact_match ) SIXTRL_NOEXCEPT
    {
        return const_cast< Derived* >( static_cast< KernelConfigBase const& >(
            *this ).asDerivedKernelConfig< Derived >(
                required_arch_id, requires_exact_match ) );
    }
}

#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_CONTROL_KERNEL_CONFIG_BASE_HPP_ */

/* end: sixtracklib/common/control/kernel_config_base.hpp */
