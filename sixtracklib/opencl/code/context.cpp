#include "sixtracklib/opencl/context.h"

#if !defined( __CUDACC__ )

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iterator>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/compute_arch.h"
#include "sixtracklib/opencl/private/base_context.h"

namespace SIXTRL_NAMESPACE
{
    ClContext::ClContext() :
        ClContextBase(),
        m_tracking_kernel_id( ClContextBase::kernel_id_t{ -1 } )
    {

    }

    ClContext::ClContext( ClContext::size_type const node_index ) :
        ClContextBase(),
        m_tracking_kernel_id( ClContextBase::kernel_id_t{ -1 } )
    {

    }

    ClContext::ClContext( ClContext::node_id_t const node_id ) :
        ClContextBase(),
        m_tracking_kernel_id( ClContextBase::kernel_id_t{ -1 } )
    {
        using size_t = ClContextBase::size_type;
        size_t const node_index = this->findAvailableNodesIndex(
            NS(ComputeNodeId_get_platform_id)( &node_id ),
            NS(ComputeNodeId_get_device_id)( &node_id ) );

        if( node_index < this->numAvailableNodes() )
        {
            this->doInitDefaultKernelsBaseImpl( node_index );
            ClContextBase::doSelectNode( node_index );
        }
    }

    ClContext::ClContext( char const* node_id_str ) :
        ClContextBase(),
        m_tracking_kernel_id( ClContextBase::kernel_id_t{ -1 } )
    {
        using size_t = ClContextBase::size_type;
        size_t const node_index = this->findAvailableNodesIndex( node_id_str );

        this->doInitDefaultProgramsBaseImpl();

        if( node_index < this->numAvailableNodes() )
        {
            ClContextBase::doSelectNode( node_index );
        }
    }

    ClContext::ClContext(
        ClContext::platform_id_t const platform_idx,
        ClContext::device_id_t const device_idx ) :
        ClContextBase(),
        m_tracking_kernel_id( ClContextBase::kernel_id_t{ -1 } )
    {
        using size_t = ClContextBase::size_type;
        size_t const node_index =
            this->findAvailableNodesIndex( platform_idx, device_idx );

        if( node_index < this->numAvailableNodes() )
        {
            this->doInitDefaultKernelsBaseImpl( node_index );
            ClContextBase::doSelectNode( node_index );
        }
    }

    ClContext::~ClContext() SIXTRL_NOEXCEPT
    {

    }

    bool ClContext::hasTrackingKernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
                 ( this->m_tracking_kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( this->m_tracking_kernel_id ) <
                   this->numAvailableKernels() ) );
    }

    ClContext::kernel_id_t ClContext::trackingKernelId() const SIXTRL_NOEXCEPT
    {
        return ( this->hasTrackingKernel() )
            ? this->m_tracking_kernel_id
            : kernel_id_t{ - 1 };
    }

    bool ClContext::setTrackingKernelId(
        ClContext::kernel_id_t const kernel_id )
    {
        bool success = false;

        if( ( this->hasSelectedNode() ) &&
            ( kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( kernel_id ) <
              this->numAvailableKernels() ) )
        {
            this->m_tracking_kernel_id = kernel_id;
            success = true;
        }

        return success;
    }

    int ClContext::track( NS(ClArgument)& particles_arg,
               NS(ClArgument)& beam_elements_arg,
               num_turns_t num_turns = num_turns_t{ 1 } );

    int ClContext::track( kernel_id_t const tracking_kernel_id,
               NS(ClArgument)& particles_arg,
               NS(ClArgument)& beam_elements_arg,
               num_turns_t const num_turns = num_turns_t{ 1 } );

    protected:

    virtual bool doInitDefaultKernels( size_type node_index );

    private:

    bool doInitDefaultKernelsBaseImpl( size_type node_index );

}



#endif /* !defined( __CUDACC__ ) */

/* end: */
