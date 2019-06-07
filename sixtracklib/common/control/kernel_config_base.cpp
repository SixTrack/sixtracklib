#include "sixtracklib/common/control/kernel_config_base.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/arch_base.hpp"

namespace SIXTRL_CXX_NAMESPACE
{
    KernelConfigBase::KernelConfigBase(
        KernelConfigBase::arch_id_t const arch_id,
        char const* SIXTRL_RESTRICT arch_str,
        char const* SIXTRL_RESTRICT config_str,
        KernelConfigBase::size_type const work_items_dim,
        KernelConfigBase::size_type const work_groups_dim ) :
        SIXTRL_CXX_NAMESPACE::ArchBase( arch_id, arch_str, config_str ),
        m_name(),
        m_num_kernel_args( KernelConfigBase::size_type{ 0 } ),
        m_kernel_id( KernelConfigBase::ILLEGAL_KERNEL_ID ),
        m_needs_update( false )
    {
        this->doResetBaseImpl( work_items_dim, work_groups_dim );
    }

    KernelConfigBase::KernelConfigBase( KernelConfigBase const& other ) :
        SIXTRL_CXX_NAMESPACE::ArchBase( other ),
        m_name( other.m_name ),
        m_num_kernel_args( other.m_num_kernel_args ),
        m_kernel_id( other.m_kernel_id ),
        m_needs_update( other.m_needs_update )
    {
        this->doPerformWorkItemsCopyBaseImpl( other );
        this->doPerformWorkGroupsCopyBaseImpl( other );
    }

    KernelConfigBase::KernelConfigBase(
        KernelConfigBase&& other ) SIXTRL_NOEXCEPT :
        SIXTRL_CXX_NAMESPACE::ArchBase( std::move( other ) ),
        m_name( std::move( other.m_name ) ),
        m_num_kernel_args( std::move( other.m_num_kernel_args ) ),
        m_kernel_id( std::move( other.m_kernel_id ) ),
        m_needs_update( std::move( other.m_needs_update ) )
    {
        this->doPerformWorkItemsCopyBaseImpl( other );
        this->doPerformWorkGroupsCopyBaseImpl( other );
        other.doClearBaseImpl();
    }

    KernelConfigBase& KernelConfigBase::operator=(
        KernelConfigBase const& rhs )
    {
        SIXTRL_CXX_NAMESPACE::ArchBase::operator=( rhs );

        if( this != &rhs )
        {
            this->m_name = rhs.m_name;
            this->m_num_kernel_args = rhs.m_num_kernel_args;
            this->m_kernel_id = rhs.m_kernel_id;
            this->m_needs_update = rhs.m_needs_update;

            this->doPerformWorkItemsCopyBaseImpl( rhs );
            this->doPerformWorkGroupsCopyBaseImpl( rhs );
        }

        return *this;
    }

    KernelConfigBase& KernelConfigBase::operator=(
        KernelConfigBase&& rhs ) SIXTRL_NOEXCEPT
    {
        SIXTRL_CXX_NAMESPACE::ArchBase::operator=( std::move( rhs ) );

        if( this != &rhs )
        {
            this->m_name = std::move( rhs.m_name );
            this->m_num_kernel_args = std::move( rhs.m_num_kernel_args );
            this->m_kernel_id = std::move( rhs.m_kernel_id );
            this->m_needs_update = std::move( rhs.m_needs_update );

            this->doPerformWorkItemsCopyBaseImpl( rhs );
            this->doPerformWorkGroupsCopyBaseImpl( rhs );
            rhs.doClearBaseImpl();
        }

        return *this;
    }

    /* --------------------------------------------------------------------- */

    bool KernelConfigBase::hasKernelId() const SIXTRL_NOEXCEPT
    {
        return ( this->m_kernel_id != KernelConfigBase::ILLEGAL_KERNEL_ID );
    }

    KernelConfigBase::kernel_id_t
    KernelConfigBase::kernelId() const SIXTRL_NOEXCEPT
    {
        return this->m_kernel_id;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    void KernelConfigBase::setKernelId(
        KernelConfigBase::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        this->m_kernel_id = kernel_id;
    }

    /* --------------------------------------------------------------------- */

    bool KernelConfigBase::hasName() const SIXTRL_NOEXCEPT
    {
        return ( !this->m_name.empty() );
    }

    std::string const& KernelConfigBase::name() const SIXTRL_NOEXCEPT
    {
        return this->m_name;
    }

    char const* KernelConfigBase::ptrNameStr() const SIXTRL_NOEXCEPT
    {
        return this->m_name.c_str();
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    void KernelConfigBase::setName(
        std::string const& SIXTRL_RESTRICT_REF kernel_name )
    {
        this->m_name = kernel_name;
    }

    void KernelConfigBase::setName(
        char const* SIXTRL_RESTRICT kernel_name )
    {
        if( ( kernel_name != nullptr ) &&
            ( std::strlen( kernel_name ) > std::size_t{ 0 } ) )
        {
            this->m_name = kernel_name;
        }
        else
        {
            this->m_name.clear();
        }
    }

    /* --------------------------------------------------------------------- */

    KernelConfigBase::size_type
    KernelConfigBase::numArguments() const SIXTRL_NOEXCEPT
    {
        return this->m_num_kernel_args;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    void KernelConfigBase::setNumArguments(
        KernelConfigBase::size_type const num_kernel_args ) SIXTRL_NOEXCEPT
    {
        this->m_num_kernel_args = num_kernel_args;
    }

    /* --------------------------------------------------------------------- */

    KernelConfigBase::size_type KernelConfigBase::numWorkItems(
        KernelConfigBase::size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ( index < this->workItemsDim() )
            ? this->m_work_items[ index ] : KernelConfigBase::size_type{ 0 };
    }

    KernelConfigBase::size_type
    KernelConfigBase::workItemsDim() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_work_items_dim <
                       KernelConfigBase::MAX_WORK_ITEMS_DIM );

        return this->m_work_items_dim;
    }

    KernelConfigBase::size_type
    KernelConfigBase::totalNumWorkItems() const SIXTRL_NOEXCEPT
    {
        using size_t = KernelConfigBase::size_type;

        size_t total_num = size_t{ 0 };
        size_t const dim = this->workItemsDim();

        for( size_t ii = size_t{ 0 } ; ii < dim ; ++ii )
        {
            total_num *= this->m_work_items[ ii ];
        }

        return total_num;
    }

    KernelConfigBase::size_type const*
    KernelConfigBase::workItemsBegin() const SIXTRL_NOEXCEPT
    {
        return &this->m_work_items[ 0 ];
    }

    KernelConfigBase::size_type const*
    KernelConfigBase::workItemsEnd() const SIXTRL_NOEXCEPT
    {
        return &this->m_work_items[ this->workItemsDim() ];
    }

    KernelConfigBase::size_type*
    KernelConfigBase::workItemsBegin() SIXTRL_NOEXCEPT
    {
        return &this->m_work_items[ 0 ];
    }

    KernelConfigBase::size_type*
    KernelConfigBase::workItemsEnd() SIXTRL_NOEXCEPT
    {
        return &this->m_work_items[ this->workItemsDim() ];
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    KernelConfigBase::status_t KernelConfigBase::setNumWorkItems(
        KernelConfigBase::size_type const work_items_a ) SIXTRL_NOEXCEPT
    {
        using size_t = KernelConfigBase::size_type;

        if( ( this->workItemsDim() == size_t{ 1 } ) &&
            ( work_items_a > size_t{ 0 } ) )
        {
            this->m_work_items[ 0 ] = work_items_a;
            return st::ARCH_STATUS_SUCCESS;
        }

        return st::ARCH_STATUS_GENERAL_FAILURE;
    }

    KernelConfigBase::status_t KernelConfigBase::setNumWorkItems(
        KernelConfigBase::size_type const work_items_a,
        KernelConfigBase::size_type const work_items_b ) SIXTRL_NOEXCEPT
    {
        using size_t = KernelConfigBase::size_type;

        if( ( this->workItemsDim() == size_t{ 1 } ) &&
            ( work_items_a > size_t{ 0 } ) && ( work_items_b > size_t{ 0 } ) )
        {
            this->m_work_items[ 0 ] = work_items_a;
            this->m_work_items[ 1 ] = work_items_b;
            return st::ARCH_STATUS_SUCCESS;
        }

        return st::ARCH_STATUS_GENERAL_FAILURE;
    }

    KernelConfigBase::status_t KernelConfigBase::setNumWorkItems(
        KernelConfigBase::size_type const work_items_a,
        KernelConfigBase::size_type const work_items_b,
        KernelConfigBase::size_type const work_items_c ) SIXTRL_NOEXCEPT
    {
        using size_t = KernelConfigBase::size_type;

        if( ( this->workItemsDim() == size_t{ 1 } ) &&
            ( work_items_a > size_t{ 0 } ) &&
            ( work_items_b > size_t{ 0 } ) &&
            ( work_items_c > size_t{ 0 } ) )
        {
            this->m_work_items[ 0 ] = work_items_a;
            this->m_work_items[ 1 ] = work_items_b;
            this->m_work_items[ 2 ] = work_items_c;
            return st::ARCH_STATUS_SUCCESS;
        }

        return st::ARCH_STATUS_GENERAL_FAILURE;
    }

    KernelConfigBase::status_t KernelConfigBase::setNumWorkItems(
        KernelConfigBase::size_type const work_items_dim,
        KernelConfigBase::size_type const*
            SIXTRL_RESTRICT work_items ) SIXTRL_NOEXCEPT
    {
        using size_t = KernelConfigBase::size_type;
        size_t const dim = this->workItemsDim();

        if( ( dim == work_items_dim ) && ( work_items != nullptr ) )
        {
            size_t total_num = size_t{ 0 };

            for( size_t ii = size_t{ 0 }; ii < dim ; ++ii )
            {
                total_num *= work_items[ ii ];
                this->m_work_items[ ii ] = work_items[ ii ];
            }

            return ( total_num > size_t{ 0 } );
        }

        return st::ARCH_STATUS_GENERAL_FAILURE;
    }

    /* --------------------------------------------------------------------- */

    KernelConfigBase::size_type KernelConfigBase::workItemOffset(
        KernelConfigBase::size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ( index < this->workItemsDim() )
            ? this->m_work_item_offsets[ index ]
            : KernelConfigBase::size_type{ 0 };
    }

    KernelConfigBase::size_type const*
    KernelConfigBase::workItemOffsetsBegin() const SIXTRL_NOEXCEPT
    {
        return &this->m_work_item_offsets[ 0 ];
    }

    KernelConfigBase::size_type const*
    KernelConfigBase::workItemOffsetsEnd() const SIXTRL_NOEXCEPT
    {
        return &this->m_work_item_offsets[ this->workItemsDim() ];
    }

    KernelConfigBase::size_type*
    KernelConfigBase::workItemOffsetsBegin() SIXTRL_NOEXCEPT
    {
        return &this->m_work_item_offsets[ 0 ];
    }

    KernelConfigBase::size_type*
    KernelConfigBase::workItemOffsetsEnd() SIXTRL_NOEXCEPT
    {
        return &this->m_work_item_offsets[ this->workItemsDim() ];
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    KernelConfigBase::status_t KernelConfigBase::setWorkItemOffset(
        KernelConfigBase::size_type const offset_a ) SIXTRL_NOEXCEPT
    {
        using size_t = KernelConfigBase::size_type;

        if( this->workItemsDim() == size_t{ 1 } )
        {
            this->m_work_item_offsets[ 0 ] = offset_a;
            return st::ARCH_STATUS_SUCCESS;
        }

        return st::ARCH_STATUS_GENERAL_FAILURE;
    }

    KernelConfigBase::status_t KernelConfigBase::setWorkItemOffset(
        KernelConfigBase::size_type const offset_a,
        KernelConfigBase::size_type const offset_b ) SIXTRL_NOEXCEPT
    {
        using size_t = KernelConfigBase::size_type;

        if( this->workItemsDim() == size_t{ 1 } )
        {
            this->m_work_item_offsets[ 0 ] = offset_a;
            this->m_work_item_offsets[ 1 ] = offset_b;
            return st::ARCH_STATUS_SUCCESS;
        }

        return st::ARCH_STATUS_GENERAL_FAILURE;
    }

    KernelConfigBase::status_t KernelConfigBase::setWorkItemOffset(
        KernelConfigBase::size_type const offset_a,
        KernelConfigBase::size_type const offset_b,
        KernelConfigBase::size_type const offset_c ) SIXTRL_NOEXCEPT
    {
        using size_t = KernelConfigBase::size_type;

        if( this->workItemsDim() == size_t{ 1 } )
        {
            this->m_work_item_offsets[ 0 ] = offset_a;
            this->m_work_item_offsets[ 1 ] = offset_b;
            this->m_work_item_offsets[ 2 ] = offset_c;
            return st::ARCH_STATUS_SUCCESS;
        }

        return st::ARCH_STATUS_GENERAL_FAILURE;
    }

    KernelConfigBase::status_t KernelConfigBase::setWorkItemOffset(
        KernelConfigBase::size_type const offsets_dim,
        KernelConfigBase::size_type const*
            SIXTRL_RESTRICT offsets_begin ) SIXTRL_NOEXCEPT
    {
        if( this->workItemsDim() == offsets_dim )
        {
            std::copy( offsets_begin, offsets_begin + offsets_dim,
                       this->workItemOffsetsBegin() );

            return st::ARCH_STATUS_SUCCESS;
        }

        return st::ARCH_STATUS_GENERAL_FAILURE;
    }

    /* --------------------------------------------------------------------- */

    KernelConfigBase::size_type KernelConfigBase::workGroupSize(
        KernelConfigBase::size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ( index < this->workGroupsDim() )
            ? this->m_work_groups[ index ] : KernelConfigBase::size_type{ 0 };
    }

    KernelConfigBase::size_type
    KernelConfigBase::workGroupsDim() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_work_groups_dim <
                       KernelConfigBase::MAX_WORK_GROUPS_DIM );

        return this->m_work_groups_dim;
    }

    KernelConfigBase::size_type const*
    KernelConfigBase::workGroupSizesBegin() const SIXTRL_NOEXCEPT
    {
        return &this->m_work_groups[ 0 ];
    }

    KernelConfigBase::size_type const*
    KernelConfigBase::workGroupSizesEnd() const SIXTRL_NOEXCEPT
    {
        return &this->m_work_groups[ this->workGroupsDim() ];
    }

    KernelConfigBase::size_type*
    KernelConfigBase::workGroupSizesBegin() SIXTRL_NOEXCEPT
    {
        return &this->m_work_groups[ 0 ];
    }

    KernelConfigBase::size_type*
    KernelConfigBase::workGroupSizesEnd() SIXTRL_NOEXCEPT
    {
        return &this->m_work_groups[ this->workGroupsDim() ];
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    KernelConfigBase::status_t KernelConfigBase::setWorkGroupSizes(
        KernelConfigBase::size_type const work_groups_a ) SIXTRL_NOEXCEPT
    {
        using size_t = KernelConfigBase::size_type;

        if( ( this->workGroupsDim() == size_t{ 1 } ) &&
            ( work_groups_a > size_t{ 0 } ) )
        {
            this->m_work_groups[ 0 ] = work_groups_a;
            return st::ARCH_STATUS_SUCCESS;
        }

        return st::ARCH_STATUS_GENERAL_FAILURE;
    }

    KernelConfigBase::status_t KernelConfigBase::setWorkGroupSizes(
        KernelConfigBase::size_type const work_groups_a,
        KernelConfigBase::size_type const work_groups_b ) SIXTRL_NOEXCEPT
    {
        using size_t = KernelConfigBase::size_type;

        if( ( this->workGroupsDim() == size_t{ 1 } ) &&
            ( work_groups_a > size_t{ 0 } ) && ( work_groups_b > size_t{ 0 } ) )
        {
            this->m_work_groups[ 0 ] = work_groups_a;
            this->m_work_groups[ 1 ] = work_groups_b;
            return st::ARCH_STATUS_SUCCESS;
        }

        return st::ARCH_STATUS_GENERAL_FAILURE;
    }

    KernelConfigBase::status_t KernelConfigBase::setWorkGroupSizes(
        KernelConfigBase::size_type const work_groups_a,
        KernelConfigBase::size_type const work_groups_b,
        KernelConfigBase::size_type const work_groups_c ) SIXTRL_NOEXCEPT
    {
        using size_t = KernelConfigBase::size_type;

        if( ( this->workGroupsDim() == size_t{ 1 } ) &&
            ( work_groups_a > size_t{ 0 } ) &&
            ( work_groups_b > size_t{ 0 } ) &&
            ( work_groups_c > size_t{ 0 } ) )
        {
            this->m_work_groups[ 0 ] = work_groups_a;
            this->m_work_groups[ 1 ] = work_groups_b;
            this->m_work_groups[ 2 ] = work_groups_c;
            return st::ARCH_STATUS_SUCCESS;
        }

        return st::ARCH_STATUS_GENERAL_FAILURE;
    }

    KernelConfigBase::status_t KernelConfigBase::setWorkGroupSizes(
        KernelConfigBase::size_type const work_groups_dim,
        KernelConfigBase::size_type const*
            SIXTRL_RESTRICT work_groups ) SIXTRL_NOEXCEPT
    {
        using size_t = KernelConfigBase::size_type;
        size_t const dim = this->workGroupsDim();

        if( ( dim == work_groups_dim ) && ( work_groups != nullptr ) )
        {
            size_t total_num = size_t{ 0 };

            for( size_t ii = size_t{ 0 }; ii < dim ; ++ii )
            {
                total_num *= work_groups[ ii ];
                this->m_work_groups[ ii ] = work_groups[ ii ];
            }

            return ( total_num > size_t{ 0 } );
        }

        return st::ARCH_STATUS_GENERAL_FAILURE;
    }

    /* --------------------------------------------------------------------- */

    KernelConfigBase::size_type KernelConfigBase::preferredWorkGroupMultiple(
        KernelConfigBase::size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ( index < this->workGroupsDim() )
            ? this->m_pref_work_groups_multiple[ index ]
            : KernelConfigBase::size_type{ 1 };
    }

    KernelConfigBase::size_type const*
    KernelConfigBase::preferredWorkGroupMultiplesBegin() const SIXTRL_NOEXCEPT
    {
        return &this->m_pref_work_groups_multiple[ 0 ];
    }

    KernelConfigBase::size_type const*
    KernelConfigBase::preferredWorkGroupMultiplesEnd() const SIXTRL_NOEXCEPT
    {
        return &this->m_pref_work_groups_multiple[ this->workGroupsDim() ];
    }

    KernelConfigBase::size_type*
    KernelConfigBase::preferredWorkGroupMultiplesBegin() SIXTRL_NOEXCEPT
    {
        return &this->m_pref_work_groups_multiple[ 0 ];
    }

    KernelConfigBase::size_type*
    KernelConfigBase::preferredWorkGroupMultiplesEnd() SIXTRL_NOEXCEPT
    {
        return &this->m_pref_work_groups_multiple[ this->workGroupsDim() ];
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    KernelConfigBase::status_t KernelConfigBase::setPreferredWorkGroupMultiple(
        KernelConfigBase::size_type const pref_wg_multi_a ) SIXTRL_NOEXCEPT
    {
        using size_t = KernelConfigBase::size_type;

        if( ( pref_wg_multi_a > size_t{ 0 } ) &&
            ( this->workGroupsDim() == size_t{ 1 } ) )
        {
            this->m_pref_work_groups_multiple[ 0 ] = pref_wg_multi_a;
            return st::ARCH_STATUS_SUCCESS;
        }

        return st::ARCH_STATUS_GENERAL_FAILURE;
    }

    KernelConfigBase::status_t KernelConfigBase::setPreferredWorkGroupMultiple(
        KernelConfigBase::size_type const pref_wg_multi_a,
        KernelConfigBase::size_type const pref_wg_multi_b ) SIXTRL_NOEXCEPT
    {
        using size_t = KernelConfigBase::size_type;

        if( ( pref_wg_multi_a > size_t{ 0 } ) &&
            ( pref_wg_multi_b > size_t{ 0 } ) &&
            ( this->workGroupsDim() == size_t{ 2 } ) )
        {
            this->m_pref_work_groups_multiple[ 0 ] = pref_wg_multi_a;
            this->m_pref_work_groups_multiple[ 1 ] = pref_wg_multi_b;
            return st::ARCH_STATUS_SUCCESS;
        }

        return st::ARCH_STATUS_GENERAL_FAILURE;
    }

    KernelConfigBase::status_t KernelConfigBase::setPreferredWorkGroupMultiple(
        KernelConfigBase::size_type const pref_wg_multi_a,
        KernelConfigBase::size_type const pref_wg_multi_b,
        KernelConfigBase::size_type const pref_wg_multi_c ) SIXTRL_NOEXCEPT
    {
        using size_t = KernelConfigBase::size_type;

        if( ( pref_wg_multi_a > size_t{ 0 } ) &&
            ( pref_wg_multi_b > size_t{ 0 } ) &&
            ( pref_wg_multi_c > size_t{ 0 } ) &&
            ( this->workGroupsDim() == size_t{ 3 } ) )
        {
            this->m_pref_work_groups_multiple[ 0 ] = pref_wg_multi_a;
            this->m_pref_work_groups_multiple[ 1 ] = pref_wg_multi_b;
            this->m_pref_work_groups_multiple[ 2 ] = pref_wg_multi_c;
            return st::ARCH_STATUS_SUCCESS;
        }

        return st::ARCH_STATUS_GENERAL_FAILURE;
    }

    KernelConfigBase::status_t KernelConfigBase::setPreferredWorkGroupMultiple(
        KernelConfigBase::size_type const work_groups_dim,
        KernelConfigBase::size_type const* SIXTRL_RESTRICT pref_wg_multi
    ) SIXTRL_NOEXCEPT
    {
        using size_t = KernelConfigBase::size_type;

        size_t const dim = this->workGroupsDim();

        bool success = ( ( dim == work_groups_dim ) &&
                         ( pref_wg_multi != nullptr ) );

        if( success )
        {
            for( size_t ii = size_t{ 0 }; ii < dim ; ++ii )
            {
                success = ( pref_wg_multi[ ii ] > size_t{ 0 } );
                this->m_pref_work_groups_multiple[ ii ] = pref_wg_multi[ ii ];
            }
        }

        return ( success )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    /* --------------------------------------------------------------------- */

    void KernelConfigBase::clear()
    {
        this->doClear();
    }

    void KernelConfigBase::reset( KernelConfigBase::size_type work_items_dim,
        KernelConfigBase::size_type work_groups_dim )
    {
        this->doReset( work_items_dim, work_groups_dim );
    }

    bool KernelConfigBase::needsUpdate() const SIXTRL_NOEXCEPT
    {
        return this->m_needs_update;
    }

    KernelConfigBase::status_t KernelConfigBase::update()
    {
        KernelConfigBase::status_t const update_status = this->doUpdate();

        if( update_status == st::ARCH_STATUS_SUCCESS )
        {
            this->doSetNeedsUpdateFlag( false );
        }

        return update_status;
    }

    /* --------------------------------------------------------------------- */

    void KernelConfigBase::doClear()
    {
        this->doClearBaseImpl();
    }

    void KernelConfigBase::doReset( KernelConfigBase::size_type work_items_dim,
        KernelConfigBase::size_type work_groups_dim )
    {
        this->doResetBaseImpl( work_items_dim, work_groups_dim );
    }

    KernelConfigBase::status_t KernelConfigBase::doUpdate()
    {
        KernelConfigBase::status_t const status = st::ARCH_STATUS_SUCCESS;
        return status;
    }

    void KernelConfigBase::doSetNeedsUpdateFlag(
        bool const needs_update ) SIXTRL_NOEXCEPT
    {
        this->m_needs_update = needs_update;
    }

    /* --------------------------------------------------------------------- */

    void KernelConfigBase::doSetNumWorkItemsValue(
        KernelConfigBase::size_type const index,
        KernelConfigBase::size_type const value ) SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( index < KernelConfigBase::MAX_WORK_ITEMS_DIM );
        this->m_work_items[ index ] = value;
    }

    void KernelConfigBase::doSetWorkItemOffsetValue(
        KernelConfigBase::size_type const index,
        KernelConfigBase::size_type const value ) SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( index < KernelConfigBase::MAX_WORK_ITEMS_DIM );
        this->m_work_item_offsets[ index ] = value;
    }

    void KernelConfigBase::doSetWorkGroupSizeValue(
        KernelConfigBase::size_type const index,
        KernelConfigBase::size_type const value ) SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( index < KernelConfigBase::MAX_WORK_GROUPS_DIM );
        this->m_work_groups[ index ] = value;
    }

    void KernelConfigBase::doSetPreferredWorkGroupMultiple(
        KernelConfigBase::size_type const index,
        KernelConfigBase::size_type const value ) SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( index < KernelConfigBase::MAX_WORK_GROUPS_DIM );
        this->m_pref_work_groups_multiple[ index ] = value;
    }

    /* --------------------------------------------------------------------- */

    void KernelConfigBase::doClearBaseImpl() SIXTRL_NOEXCEPT
    {
        using size_t = KernelConfigBase::size_type;

        this->m_name.clear();
        this->m_num_kernel_args = size_t{ 0 };
        this->m_kernel_id = KernelConfigBase::ILLEGAL_KERNEL_ID;
        this->m_needs_update = false;

        if( this->m_work_items_dim > size_t{ 0 } )
        {
            SIXTRL_ASSERT( this->m_work_items_dim <= MAX_WORK_ITEMS_DIM );

            std::fill( this->workItemsBegin(),
                       this->workItemsEnd(), size_t{ 0 } );

            std::fill( this->workItemOffsetsBegin(),
                       this->workItemOffsetsEnd(), size_t{ 0 } );
        }

        if( this->m_work_groups_dim > size_t{ 0 } )
        {
            SIXTRL_ASSERT( this->m_work_groups_dim <= MAX_WORK_GROUPS_DIM );

            std::fill( this->workGroupSizesBegin(),
                       this->workGroupSizesEnd(), size_t{ 0 } );

            std::fill( this->preferredWorkGroupMultiplesBegin(),
                       this->preferredWorkGroupMultiplesEnd(), size_t{ 1 } );
        }
    }

    void KernelConfigBase::doPerformWorkItemsCopyBaseImpl(
        KernelConfigBase const& other ) SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( other.m_work_items_dim >
                       KernelConfigBase::size_type{ 0 } );

        SIXTRL_ASSERT( other.m_work_items_dim <=
                       KernelConfigBase::MAX_WORK_ITEMS_DIM );

        this->m_work_items_dim = other.m_work_items_dim;

        std::copy( other.workItemsBegin(), other.workItemsEnd(),
                   this->workItemsBegin() );

        std::copy( other.workItemOffsetsBegin(), other.workItemOffsetsEnd(),
                   this->workItemOffsetsBegin() );

        return;
    }

    void KernelConfigBase::doPerformWorkGroupsCopyBaseImpl(
        KernelConfigBase const& other ) SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( other.m_work_groups_dim >
                       KernelConfigBase::size_type{ 0 } );

        SIXTRL_ASSERT( other.m_work_groups_dim <=
                       KernelConfigBase::MAX_WORK_GROUPS_DIM );

        this->m_work_groups_dim = other.m_work_groups_dim;

        std::copy( other.workGroupSizesBegin(), other.workGroupSizesEnd(),
                   this->workGroupSizesBegin() );

        std::copy( other.preferredWorkGroupMultiplesBegin(),
                   other.preferredWorkGroupMultiplesEnd(),
                   this->preferredWorkGroupMultiplesBegin() );

        return;
    }

    void KernelConfigBase::doResetBaseImpl(
        KernelConfigBase::size_type work_items_dim,
        KernelConfigBase::size_type work_groups_dim ) SIXTRL_NOEXCEPT
    {
        using size_t = KernelConfigBase::size_type;

        auto begin = this->workItemsBegin();
        auto end   = begin;
        std::advance( end, KernelConfigBase::MAX_WORK_ITEMS_DIM );
        std::fill( begin, end, size_t{ 0 } );

        begin = this->workItemOffsetsBegin();
        end = begin;
        std::advance( end, KernelConfigBase::MAX_WORK_ITEMS_DIM );
        std::fill( begin, end, size_t{ 0 } );

        begin = this->workGroupSizesBegin();
        end = begin;
        std::advance( end, KernelConfigBase::MAX_WORK_ITEMS_DIM );
        std::fill( begin, end, size_t{ 0 } );

        begin = this->preferredWorkGroupMultiplesBegin();
        end = begin;
        std::advance( end, KernelConfigBase::MAX_WORK_ITEMS_DIM );
        std::fill( begin, end, size_t{ 0 } );

        if( ( work_items_dim > size_t{ 0 } ) &&
            ( MAX_WORK_GROUPS_DIM >= work_items_dim ) )
        {
            this->m_work_items_dim = work_items_dim;
        }
        else
        {
            this->m_work_items_dim = size_t{ 0 };
        }

        if( ( work_groups_dim > size_t{ 0 } ) &&
            ( work_groups_dim <= MAX_WORK_GROUPS_DIM ) )
        {
            this->m_work_groups_dim = work_groups_dim;
        }
        else
        {
            this->m_work_groups_dim = size_t{ 0 };
        }

        return;
    }

    /* --------------------------------------------------------------------- */

    void KernelConfigBase::print(
        std::ostream& SIXTRL_RESTRICT_REF output ) const
    {
        this->doPrintToOutputStream( output );
    }

    void KernelConfigBase::print( ::FILE* SIXTRL_RESTRICT output ) const
    {
        if( output != nullptr )
        {
            std::ostringstream a2str;
            this->print( a2str );
            std::string const string_output( a2str.str() );

            if( !string_output.empty() )
            {
                int const ret = std::fprintf(
                    output, "%s", string_output.c_str() );

                SIXTRL_ASSERT( ret > int{ 0 } );
                ( void )ret;
            }
        }
    }

    void KernelConfigBase::printOut() const
    {
        this->print( std::cout );
    }

    /* --------------------------------------------------------------------- */

    void KernelConfigBase::doPrintToOutputStream(
        std::ostream& SIXTRL_RESTRICT_REF output ) const
    {
        using size_t = KernelConfigBase::size_type;

        if( this->needsUpdate() )
        {
            output << "!!! WARNING: Preliminary values, "
                   << "call update() before using !!!\r\n\r\n";
        }

        if( this->hasName() )
        {
            output << "kernel name          : " << this->m_name << "\r\n";
        }


        output << "num kernel arguments : "
               << this->m_num_kernel_args
               << "\r\n";

        if( this->workItemsDim() > size_t{ 0 } )
        {
            output << "work items dim       : " << this->workItemsDim()
                   << "total num work items : " << this->totalNumWorkItems()
                   << "\r\n";

            if( this->workItemsDim() == size_t{ 1 } )
            {
                output << "num work items       : "
                       << this->numWorkItems( size_t{ 0 } ) << "\r\n"
                       << "work items offset    : "
                       << this->workItemOffset( size_t{ 0 } ) << "\r\n";
            }
            else
            {
                std::ostringstream a2str;

                output << "num work items       : "
                       << "[ " << std::setw( 5 )
                       << this->numWorkItems( size_t{ 0 } );

                a2str  << "work items offset    : "
                       << "[ " << std::setw( 5 )
                       << this->workItemOffset( size_t{ 0 } );

                size_t ii = size_t{ 1 };

                for( ; ii < this->workItemsDim() ; ++ii )
                {
                    output << " / "
                           << std::setw( 5 ) << this->numWorkItems( ii );

                    a2str  << " / "
                           << std::setw( 5 ) << this->workItemOffset( ii );
                }

                output << " ] \r\n" << a2str.str() << " ] \r\n";
            }
        }

        if( this->workGroupsDim() > size_t{ 0 } )
        {
            output << "work groups dim      : " << this->workGroupsDim()
                   << "\r\n";

            if( this->workGroupsDim() == size_t{ 1 } )
            {
                output << "num work groups      : "
                       << this->workGroupSize( size_t{ 0 } ) << "\r\n"
                       << "pref work group mult : "
                       << this->preferredWorkGroupMultiple( size_t{ 0 } )
                       << "\r\n";
            }
            else
            {
                std::ostringstream a2str;

                output << "[ " << std::setw( 5 )
                       << this->workGroupSize( size_t{ 0 } );

                a2str  << "[ " << std::setw( 5 )
                       << this->preferredWorkGroupMultiple( size_t{ 0 } );

                size_t ii = size_t{ 1 };

                for( ; ii  < this->workGroupsDim() ; ++ii )
                {
                    output << " / "
                           << std::setw( 5 ) << this->workGroupSize( ii );

                    a2str  << " / "
                           << std::setw( 5 )
                           << this->preferredWorkGroupMultiple( ii );
                }

                output << " ]\r\n" << a2str.str() << " ]\r\n";
            }
        }
    }

    /* ===================================================================== */

    std::ostream& operator<<( std::ostream& SIXTRL_RESTRICT_REF output,
        KernelConfigBase const& SIXTRL_RESTRICT_REF config )
    {
        config.print( output );
        return output;
    }
}

/* end: sixtracklib/common/control/kernel_config_base.cpp */
