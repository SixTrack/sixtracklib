#include "sixtracklib/opencl/context.h"

#if !defined( __CUDACC__ )

#include <chrono>
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

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/internal/compute_arch.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/opencl/internal/base_context.h"

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    ClContext::ClContext() :
        ClContextBase(),
        m_track_until_turn_program_id( ClContextBase::program_id_t{ -1 } ),
        m_track_single_turn_program_id( ClContextBase::program_id_t{ -1 } ),
        m_track_elem_by_elem_program_id( ClContextBase::program_id_t{ -1 } ),
        m_assign_be_mon_io_buffer_program_id( ClContextBase::program_id_t{ -1 } ),
        m_clear_be_mon_program_id( ClContextBase::program_id_t{ -1 } ),
        m_track_single_turn_kernel_id( ClContextBase::kernel_id_t{ -1 } ),
        m_track_elem_by_elem_kernel_id( ClContextBase::kernel_id_t{ -1 } ),
        m_assign_be_mon_io_buffer_kernel_id( ClContextBase::kernel_id_t{ -1 } ),
        m_clear_be_mon_kernel_id( ClContextBase::kernel_id_t{ -1 } ),
        m_use_optimized_tracking( true ),
        m_enable_beam_beam( true )
    {
        this->doInitDefaultProgramsPrivImpl();
    }

    ClContext::ClContext( ClContext::size_type const node_index ) :
        ClContextBase(),
        m_track_until_turn_program_id( ClContextBase::program_id_t{ -1 } ),
        m_track_single_turn_program_id( ClContextBase::program_id_t{ -1 } ),
        m_track_elem_by_elem_program_id( ClContextBase::program_id_t{ -1 } ),
        m_assign_be_mon_io_buffer_program_id( ClContextBase::program_id_t{ -1 } ),
        m_clear_be_mon_program_id( ClContextBase::program_id_t{ -1 } ),
        m_track_single_turn_kernel_id( ClContextBase::kernel_id_t{ -1 } ),
        m_track_elem_by_elem_kernel_id( ClContextBase::kernel_id_t{ -1 } ),
        m_assign_be_mon_io_buffer_kernel_id( ClContextBase::kernel_id_t{ -1 } ),
        m_clear_be_mon_kernel_id( ClContextBase::kernel_id_t{ -1 } ),
        m_use_optimized_tracking( true ),
        m_enable_beam_beam( true )
    {
        using base_t = ClContextBase;

        this->doInitDefaultProgramsPrivImpl();

        if( ( node_index < this->numAvailableNodes() ) &&
            ( base_t::doSelectNode( node_index ) ) )
        {
            base_t::doInitDefaultKernels();
            this->doInitDefaultKernelsPrivImpl();
        }
    }

    ClContext::ClContext( ClContext::node_id_t const node_id ) :
        ClContextBase(),
        m_track_until_turn_program_id( ClContextBase::program_id_t{ -1 } ),
        m_track_single_turn_program_id( ClContextBase::program_id_t{ -1 } ),
        m_track_elem_by_elem_program_id( ClContextBase::program_id_t{ -1 } ),
        m_assign_be_mon_io_buffer_program_id( ClContextBase::program_id_t{ -1 } ),
        m_clear_be_mon_program_id( ClContextBase::program_id_t{ -1 } ),
        m_track_single_turn_kernel_id( ClContextBase::kernel_id_t{ -1 } ),
        m_track_elem_by_elem_kernel_id( ClContextBase::kernel_id_t{ -1 } ),
        m_assign_be_mon_io_buffer_kernel_id( ClContextBase::kernel_id_t{ -1 } ),
        m_clear_be_mon_kernel_id( ClContextBase::kernel_id_t{ -1 } ),
        m_use_optimized_tracking( true ),
        m_enable_beam_beam( true )
    {
        using base_t = ClContextBase;
        using size_t = base_t::size_type;

        this->doInitDefaultProgramsPrivImpl();

        size_t const node_index = this->findAvailableNodesIndex(
            NS(ComputeNodeId_get_platform_id)( &node_id ),
            NS(ComputeNodeId_get_device_id)( &node_id ) );

        if( ( node_index < this->numAvailableNodes() ) &&
            ( base_t::doSelectNode( node_index ) ) )
        {
            base_t::doInitDefaultKernels();
            this->doInitDefaultKernelsPrivImpl();
        }
    }

    ClContext::ClContext( char const* node_id_str ) :
        ClContextBase(),
        m_track_until_turn_program_id( ClContextBase::program_id_t{ -1 } ),
        m_track_single_turn_program_id( ClContextBase::program_id_t{ -1 } ),
        m_track_elem_by_elem_program_id( ClContextBase::program_id_t{ -1 } ),
        m_assign_be_mon_io_buffer_program_id( ClContextBase::program_id_t{ -1 } ),
        m_clear_be_mon_program_id( ClContextBase::program_id_t{ -1 } ),
        m_track_single_turn_kernel_id( ClContextBase::kernel_id_t{ -1 } ),
        m_track_elem_by_elem_kernel_id( ClContextBase::kernel_id_t{ -1 } ),
        m_assign_be_mon_io_buffer_kernel_id( ClContextBase::kernel_id_t{ -1 } ),
        m_clear_be_mon_kernel_id( ClContextBase::kernel_id_t{ -1 } ),
        m_use_optimized_tracking( true ),
        m_enable_beam_beam( true )
    {
        using base_t = ClContextBase;
        using size_t = base_t::size_type;

        this->doInitDefaultProgramsPrivImpl();

        size_t node_index = this->findAvailableNodesIndex( node_id_str );

        if( node_index >= this->numAvailableNodes() )
        {
            node_id_t const default_node_id = this->defaultNodeId();

            platform_id_t const platform_index =
                NS(ComputeNodeId_get_platform_id)( &default_node_id );

            device_id_t const device_index =
                NS(ComputeNodeId_get_device_id)( &default_node_id );

            node_index = this->findAvailableNodesIndex(
                platform_index, device_index );
        }

        if( ( node_index < this->numAvailableNodes() ) &&
            ( base_t::doSelectNode( node_index ) ) )
        {
            base_t::doInitDefaultKernels();
            this->doInitDefaultKernelsPrivImpl();
        }
    }

    ClContext::ClContext(
        ClContext::platform_id_t const platform_idx,
        ClContext::device_id_t const device_idx ) :
        ClContextBase(),
        m_track_until_turn_program_id( ClContextBase::program_id_t{ -1 } ),
        m_track_single_turn_program_id( ClContextBase::program_id_t{ -1 } ),
        m_track_elem_by_elem_program_id( ClContextBase::program_id_t{ -1 } ),
        m_assign_be_mon_io_buffer_program_id( ClContextBase::program_id_t{ -1 } ),
        m_clear_be_mon_program_id( ClContextBase::program_id_t{ -1 } ),
        m_track_single_turn_kernel_id( ClContextBase::kernel_id_t{ -1 } ),
        m_track_elem_by_elem_kernel_id( ClContextBase::kernel_id_t{ -1 } ),
        m_assign_be_mon_io_buffer_kernel_id( ClContextBase::kernel_id_t{ -1 } ),
        m_clear_be_mon_kernel_id( ClContextBase::kernel_id_t{ -1 } ),
        m_use_optimized_tracking( true ),
        m_enable_beam_beam( true )
    {
        using base_t = ClContextBase;
        using size_t = base_t::size_type;

        this->doInitDefaultProgramsPrivImpl();

        size_t const node_index =
            this->findAvailableNodesIndex( platform_idx, device_idx );

        if( ( node_index < this->numAvailableNodes() ) &&
            ( base_t::doSelectNode( node_index ) ) )
        {
            base_t::doInitDefaultKernels();
            this->doInitDefaultKernelsPrivImpl();
        }
    }

    ClContext::~ClContext() SIXTRL_NOEXCEPT
    {

    }

    bool ClContext::hasTrackingKernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
                 ( this->m_track_until_turn_kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >( this->m_track_until_turn_kernel_id ) <
                   this->numAvailableKernels() ) );
    }

    ClContext::kernel_id_t ClContext::trackingKernelId() const SIXTRL_NOEXCEPT
    {
        return ( this->hasTrackingKernel() )
            ? this->m_track_until_turn_kernel_id : kernel_id_t{ - 1 };
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
            program_id_t const tracking_program_id =
                this->programIdByKernelId( kernel_id );

            if( ( tracking_program_id >= program_id_t{ 0 } ) &&
                ( static_cast< size_type >( tracking_program_id ) <
                  this->numAvailablePrograms() ) )
            {
                this->m_track_until_turn_kernel_id  = kernel_id;
                this->m_track_until_turn_program_id = tracking_program_id;
                success = true;
            }
        }

        return success;
    }

    int ClContext::track( ClContext::num_turns_t const turn )
    {
        int success = -1;

        if( this->hasTrackingKernel() )
        {
            success = this->track( turn, this->trackingKernelId() );
        }

        return success;

    }

    int ClContext::track( ClContext::num_turns_t const turn,
                          ClContext::kernel_id_t const track_kernel_id )
    {
        int success = -1;

        if( ( this->hasSelectedNode() ) &&
            ( track_kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( track_kernel_id ) <
                this->numAvailableKernels() ) )
        {
            int64_t const turn_arg = static_cast< int64_t >( turn );
            this->assignKernelArgumentValue( track_kernel_id, 2u, turn_arg );

            success = ( this->runKernel( track_kernel_id,
                        this->lastExecNumWorkItems( track_kernel_id ),
                        this->lastExecWorkGroupSize( track_kernel_id ) ) )
                    ? 0 : -1;
        }

        return success;
    }

    int ClContext::track( ClArgument& SIXTRL_RESTRICT_REF particles_arg,
        ClArgument& SIXTRL_RESTRICT_REF beam_elements_arg,
        ClContext::num_turns_t const turn )
    {
        SIXTRL_ASSERT( ( this->hasSelectedNode() ) &&
                       ( this->hasTrackingKernel() ) );

        return this->track( particles_arg, beam_elements_arg, turn,
                            this->trackingKernelId() );
    }

    int ClContext::track( ClArgument& particles_arg,
                          ClArgument& beam_elements_arg,
                          ClContext::num_turns_t const until_turn,
                          ClContext::kernel_id_t const track_kernel_id )
    {
        int success = -1;

        SIXTRL_ASSERT( this->hasSelectedNode() );
        SIXTRL_ASSERT( ( track_kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( track_kernel_id ) <
              this->numAvailableKernels() ) );

        SIXTRL_ASSERT( particles_arg.usesCObjectBuffer() );
        NS(Buffer)* particles_buffer = particles_arg.ptrCObjectBuffer();
        SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( particles_buffer ) );

        SIXTRL_ASSERT( beam_elements_arg.usesCObjectBuffer() );
        SIXTRL_ASSERT( !NS(Buffer_needs_remapping)(
            beam_elements_arg.ptrCObjectBuffer() ) );

        size_type const num_kernel_args = this->kernelNumArgs( track_kernel_id );
        SIXTRL_ASSERT(  num_kernel_args >= 3u );

        size_type const total_num_particles =
            NS(Particles_buffer_get_total_num_of_particles)( particles_buffer);

        SIXTRL_ASSERT( total_num_particles > size_type{ 0 } );

        int64_t const until_turn_arg = static_cast< int64_t >( until_turn );

        this->assignKernelArgument( track_kernel_id, 0u, particles_arg );
        this->assignKernelArgument( track_kernel_id, 1u, beam_elements_arg );
        this->assignKernelArgumentValue( track_kernel_id, 2u, until_turn_arg );

        if( num_kernel_args > 3u )
        {
            this->assignKernelArgumentClBuffer(
                track_kernel_id, 3u, this->internalSuccessFlagBuffer() );
        }

        success = ( this->runKernel( track_kernel_id, total_num_particles ) )
            ? 0 : -1;

        if( ( success == 0 ) && ( num_kernel_args > 3u ) )
        {
            cl::CommandQueue* ptr_queue = this->openClQueue();
            SIXTRL_ASSERT( ptr_queue != nullptr );

            int32_t success_flag = int32_t{ -1 };
            cl_int cl_ret = ptr_queue->enqueueReadBuffer(
                this->internalSuccessFlagBuffer(), CL_TRUE, 0,
                sizeof( success_flag ), &success_flag );

            if( cl_ret == CL_SUCCESS )
            {
                success = ( int )success_flag;
            }

            ptr_queue->finish();
        }

        return success;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ClContext::hasSingleTurnTrackingKernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
                 ( this->m_track_single_turn_kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >(
                     this->m_track_single_turn_kernel_id ) <
                         this->numAvailableKernels() ) );
    }

    ClContext::kernel_id_t
    ClContext::singleTurnTackingKernelId() const SIXTRL_NOEXCEPT
    {
        return ( this->hasSingleTurnTrackingKernel() )
            ? this->m_track_single_turn_kernel_id : kernel_id_t{ -1 };
    }

    bool ClContext::setSingleTurnTrackingKernelId(
         ClContext::kernel_id_t const track_kernel_id )
    {
        bool success = false;

        if( ( this->hasSelectedNode() ) &&
            ( track_kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( track_kernel_id ) <
              this->numAvailableKernels() ) )
        {
            program_id_t const tracking_program_id =
                this->programIdByKernelId( track_kernel_id );

            if( ( tracking_program_id >= program_id_t{ 0 } ) &&
                ( static_cast< size_type >( tracking_program_id ) <
                  this->numAvailablePrograms() ) )
            {
                this->m_track_single_turn_kernel_id  = track_kernel_id;
                this->m_track_single_turn_program_id = tracking_program_id;
                success = true;
            }
        }

        return success;
    }

    int ClContext::trackSingleTurn()
    {
        return ( this->hasSingleTurnTrackingKernel() )
            ? this->trackSingleTurn( this->singleTurnTackingKernelId() )
            : -1;
    }

    int ClContext::trackSingleTurn(
        ClContext::kernel_id_t const track_kernel_id )
    {
        return ( this->runKernel( track_kernel_id,
            this->lastExecNumWorkItems( track_kernel_id ),
            this->lastExecWorkGroupSize( track_kernel_id ) ) ) ? 0 : -1;
    }

    int ClContext::trackSingleTurn(
        ClArgument& SIXTRL_RESTRICT_REF particles_arg,
        ClArgument& SIXTRL_RESTRICT_REF beam_elements_arg )
    {
        return ( this->hasSingleTurnTrackingKernel() )
            ? this->trackSingleTurn( particles_arg, beam_elements_arg,
                                     this->singleTurnTackingKernelId() )
            : -1;
    }

    int ClContext::trackSingleTurn(
        ClArgument& particles_arg, ClArgument& beam_elements_arg,
        ClContext::kernel_id_t const track_kernel_id )
    {
        int success = -1;

        SIXTRL_ASSERT( this->hasSelectedNode() );
        SIXTRL_ASSERT( ( track_kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( track_kernel_id ) <
              this->numAvailableKernels() ) );

        SIXTRL_ASSERT( particles_arg.usesCObjectBuffer() );
        NS(Buffer)* particles_buffer = particles_arg.ptrCObjectBuffer();
        SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( particles_buffer ) );

        SIXTRL_ASSERT( beam_elements_arg.usesCObjectBuffer() );
        SIXTRL_ASSERT( !NS(Buffer_needs_remapping)(
            beam_elements_arg.ptrCObjectBuffer() ) );

        size_type const num_kernel_args = this->kernelNumArgs( track_kernel_id );
        SIXTRL_ASSERT(  num_kernel_args >= 2u );

        size_type const total_num_particles =
            NS(Particles_buffer_get_total_num_of_particles)( particles_buffer);

        SIXTRL_ASSERT( total_num_particles > size_type{ 0 } );

        this->assignKernelArgument( track_kernel_id, 0u, particles_arg );
        this->assignKernelArgument( track_kernel_id, 1u, beam_elements_arg );

        if( num_kernel_args > 2u )
        {
            this->assignKernelArgumentClBuffer(
                track_kernel_id, 2u, this->internalSuccessFlagBuffer() );
        }

        success = ( this->runKernel(
            track_kernel_id, total_num_particles ) ) ? 0 : -1;

        if( ( success == 0 ) && ( num_kernel_args > 2u ) )
        {
            cl::CommandQueue* ptr_queue = this->openClQueue();
            SIXTRL_ASSERT( ptr_queue != nullptr );

            int32_t success_flag = int32_t{ -1 };
            cl_int cl_ret = ptr_queue->enqueueReadBuffer(
                this->internalSuccessFlagBuffer(), CL_TRUE, 0,
                sizeof( success_flag ), &success_flag );

            if( cl_ret == CL_SUCCESS )
            {
                success = ( int )success_flag;
            }

            ptr_queue->finish();
        }

        return success;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ClContext::hasElementByElementTrackingKernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
                 ( this->m_track_elem_by_elem_kernel_id >= kernel_id_t{ 0 } ) &&
                 ( static_cast< size_type >(
                     this->m_track_elem_by_elem_kernel_id ) <
                         this->numAvailableKernels() ) );
    }

    ClContext::kernel_id_t
    ClContext::elementByElementTrackingKernelId() const SIXTRL_NOEXCEPT
    {
        return ( this->hasElementByElementTrackingKernel() )
            ? this->m_track_elem_by_elem_kernel_id : kernel_id_t{ -1 };
    }

    bool ClContext::setElementByElementTrackingKernelId(
        ClContext::kernel_id_t const track_kernel_id )
    {
        bool success = false;

        if( ( this->hasSelectedNode() ) &&
            ( track_kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( track_kernel_id ) <
              this->numAvailableKernels() ) )
        {
            program_id_t const tracking_program_id =
                this->programIdByKernelId( track_kernel_id );

            if( ( tracking_program_id >= program_id_t{ 0 } ) &&
                ( static_cast< size_type >( tracking_program_id ) <
                  this->numAvailablePrograms() ) )
            {
                this->m_track_elem_by_elem_kernel_id  = track_kernel_id;
                this->m_track_elem_by_elem_program_id = tracking_program_id;
                success = true;
            }
        }

        return success;
    }

    int ClContext::trackElementByElement(
        ClContext::size_type const io_particle_block_offset )
    {
        return ( this->hasElementByElementTrackingKernel() )
            ? this->trackElementByElement( io_particle_block_offset,
                this->elementByElementTrackingKernelId() ) : -1;
    }

    int ClContext::trackElementByElement(
        ClContext::size_type const io_particle_block_offset,
        ClContext::kernel_id_t const track_kernel_id )
    {
        if( this->hasElementByElementTrackingKernel() )
        {
            this->assignKernelArgumentValue(
                track_kernel_id, 3u, io_particle_block_offset );

            if( this->runKernel( track_kernel_id,
                this->lastExecNumWorkItems( track_kernel_id ),
                this->lastExecWorkGroupSize( track_kernel_id ) ) )
            {
                return 0;
            }
        }

        return -1;
    }

    int ClContext::trackElementByElement(
        ClArgument& SIXTRL_RESTRICT_REF particles_arg,
        ClArgument& SIXTRL_RESTRICT_REF beam_elements_arg,
        ClArgument& SIXTRL_RESTRICT_REF elem_by_elem_buffer,
        ClContext::size_type io_particle_block_offset )
    {
        return this->trackElementByElement( particles_arg, beam_elements_arg,
            elem_by_elem_buffer, io_particle_block_offset,
                this->elementByElementTrackingKernelId() );
    }

    int ClContext::trackElementByElement(
        ClArgument& SIXTRL_RESTRICT_REF particles_arg,
        ClArgument& SIXTRL_RESTRICT_REF beam_elements_arg,
        ClArgument& SIXTRL_RESTRICT_REF elem_by_elem_buffer_arg,
        ClContext::size_type io_particle_block_offset,
        ClContext::kernel_id_t const track_kernel_id )
    {
        int success = -1;

        SIXTRL_ASSERT( this->hasSelectedNode() );
        SIXTRL_ASSERT( ( track_kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( track_kernel_id ) <
              this->numAvailableKernels() ) );

        SIXTRL_ASSERT( particles_arg.usesCObjectBuffer() );
        NS(Buffer)* particles_buffer = particles_arg.ptrCObjectBuffer();
        SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( particles_buffer ) );

        SIXTRL_ASSERT( beam_elements_arg.usesCObjectBuffer() );
        SIXTRL_ASSERT( !NS(Buffer_needs_remapping)(
            beam_elements_arg.ptrCObjectBuffer() ) );

        SIXTRL_ASSERT( elem_by_elem_buffer_arg.usesCObjectBuffer() );
        SIXTRL_ASSERT( !NS(Buffer_needs_remapping)(
            elem_by_elem_buffer_arg.ptrCObjectBuffer() ) );

        size_type const num_kernel_args = this->kernelNumArgs( track_kernel_id );
        SIXTRL_ASSERT(  num_kernel_args >= 4u );

        size_type const total_num_particles =
            NS(Particles_buffer_get_total_num_of_particles)( particles_buffer);

        SIXTRL_ASSERT( total_num_particles > size_type{ 0 } );

        this->assignKernelArgument( track_kernel_id, 0u, particles_arg );
        this->assignKernelArgument( track_kernel_id, 1u, beam_elements_arg );
        this->assignKernelArgument( track_kernel_id, 2u, elem_by_elem_buffer_arg );

        this->assignKernelArgumentValue(
            track_kernel_id, 3u, io_particle_block_offset );

        if( num_kernel_args > 4u )
        {
            this->assignKernelArgumentClBuffer(
                track_kernel_id, 4u, this->internalSuccessFlagBuffer() );
        }

        success = ( !this->runKernel( track_kernel_id, total_num_particles ) )
             ? 0 : -1;

        if( ( success == 0 ) && ( num_kernel_args > 3u ) )
        {
            cl::CommandQueue* ptr_queue = this->openClQueue();
            SIXTRL_ASSERT( ptr_queue != nullptr );

            int32_t success_flag = int32_t{ -1 };
            cl_int cl_ret = ptr_queue->enqueueReadBuffer(
                this->internalSuccessFlagBuffer(), CL_TRUE, 0,
                sizeof( success_flag ), &success_flag );

            if( cl_ret == CL_SUCCESS )
            {
                success = ( int )success_flag;
            }

            ptr_queue->finish();
        }

        return success;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ClContext::hasAssignBeamMonitorIoBufferKernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
            ( this->m_assign_be_mon_io_buffer_kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >(
                this->m_assign_be_mon_io_buffer_kernel_id ) <
                this->numAvailableKernels() ) );
    }

    ClContext::kernel_id_t const
    ClContext::assignBeamMonitorIoBufferKernelId() const SIXTRL_NOEXCEPT
    {
        return ( this->hasAssignBeamMonitorIoBufferKernel() )
            ? this->m_assign_be_mon_io_buffer_kernel_id : kernel_id_t{ -1 };
    }

    bool ClContext::setAssignBeamMonitorIoBufferKernelId(
        ClContext::kernel_id_t const track_kernel_id )
    {
        bool success = false;

        if( ( this->hasSelectedNode() ) &&
            ( track_kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( track_kernel_id ) <
              this->numAvailableKernels() ) )
        {
            program_id_t const tracking_program_id =
                this->programIdByKernelId( track_kernel_id );

            if( ( tracking_program_id >= program_id_t{ 0 } ) &&
                ( static_cast< size_type >( tracking_program_id ) <
                  this->numAvailablePrograms() ) )
            {
                this->m_assign_be_mon_io_buffer_kernel_id  = track_kernel_id;
                this->m_assign_be_mon_io_buffer_program_id = tracking_program_id;
                success = true;
            }
        }

        return success;
    }

    int ClContext::assignBeamMonitorIoBuffer(
        ClArgument& SIXTRL_RESTRICT_REF beam_elements_arg,
        ClArgument& SIXTRL_RESTRICT_REF io_buffer_arg,
        ClContext::size_type const num_particles,
        ClContext::size_type const io_particle_block_offset  )
    {
        int success = -1;

        kernel_id_t const kernel_id = this->assignBeamMonitorIoBufferKernelId();
        kernel_id_t const max_kernel_id = this->numAvailableKernels();

        if( ( kernel_id >= kernel_id_t{ 0 } ) && ( kernel_id <  max_kernel_id ) )
        {
            success = this->assignBeamMonitorIoBuffer( beam_elements_arg,
                io_buffer_arg, num_particles, io_particle_block_offset, kernel_id );
        }

        return success;
    }

    int ClContext::assignBeamMonitorIoBuffer(
        ClArgument& SIXTRL_RESTRICT_REF beam_elements_arg,
        ClArgument& SIXTRL_RESTRICT_REF io_buffer_arg,
        ClContext::size_type const num_particles,
        ClContext::size_type const io_particle_block_offset,
        ClContext::kernel_id_t const assign_kernel_id )
    {
        int success = -1;

        SIXTRL_ASSERT( this->hasSelectedNode() );
        SIXTRL_ASSERT( ( assign_kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( assign_kernel_id ) <
              this->numAvailableKernels() ) );

        SIXTRL_ASSERT( beam_elements_arg.usesCObjectBuffer() );
        SIXTRL_ASSERT( !NS(Buffer_needs_remapping)(
            beam_elements_arg.ptrCObjectBuffer() ) );

        SIXTRL_ASSERT( io_buffer_arg.usesCObjectBuffer() );
        SIXTRL_ASSERT( !NS(Buffer_needs_remapping)(
            io_buffer_arg.ptrCObjectBuffer() ) );

        size_type const num_kernel_args = this->kernelNumArgs( assign_kernel_id );
        SIXTRL_ASSERT(  num_kernel_args >= 4u );

        this->assignKernelArgument( assign_kernel_id, 0u, beam_elements_arg );
        this->assignKernelArgument( assign_kernel_id, 1u, io_buffer_arg );
        this->assignKernelArgumentValue( assign_kernel_id, 2u, num_particles );

        this->assignKernelArgumentValue(
            assign_kernel_id, 3u, io_particle_block_offset );

        if( num_kernel_args > 4u )
        {
            this->assignKernelArgumentClBuffer(
                assign_kernel_id, 4u, this->internalSuccessFlagBuffer() );
        }

        success = ( this->runKernel( assign_kernel_id,
                this->kernelPreferredWorkGroupSizeMultiple( assign_kernel_id ) ) )
            ? 0 : -1;

        if( ( success == 0 ) && ( num_kernel_args > 3u ) )
        {
            cl::CommandQueue* ptr_queue = this->openClQueue();
            SIXTRL_ASSERT( ptr_queue != nullptr );

            int32_t success_flag = int32_t{ -1 };
            cl_int cl_ret = ptr_queue->enqueueReadBuffer(
                this->internalSuccessFlagBuffer(), CL_TRUE, 0,
                sizeof( success_flag ), &success_flag );

            if( cl_ret == CL_SUCCESS )
            {
                success = ( int )success_flag;
            }

            ptr_queue->finish();
        }

        return success;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    bool ClContext::hasClearBeamMonitorIoBufferAssignmentKernel() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasSelectedNode() ) &&
            ( this->m_clear_be_mon_kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >(
                this->m_clear_be_mon_kernel_id ) < this->numAvailableKernels() ) );
    }

    ClContext::kernel_id_t
    ClContext::clearBeamMonitorIoBufferAssignmentKernelId() const SIXTRL_NOEXCEPT
    {
        return ( this->hasClearBeamMonitorIoBufferAssignmentKernel() )
            ? this->m_clear_be_mon_kernel_id : kernel_id_t{ -1 };
    }

    bool ClContext::setClearBeamMonitorIoBufferAssignmentKernelId(
        ClContext::kernel_id_t const clear_assign_kernel_id )
    {
        bool success = false;

        if( ( this->hasSelectedNode() ) &&
            ( clear_assign_kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( clear_assign_kernel_id ) <
              this->numAvailableKernels() ) )
        {
            program_id_t const clear_assign_program_id =
                this->programIdByKernelId( clear_assign_kernel_id );

            if( ( clear_assign_program_id >= program_id_t{ 0 } ) &&
                ( static_cast< size_type >( clear_assign_program_id ) <
                  this->numAvailablePrograms() ) )
            {
                this->m_clear_be_mon_kernel_id  = clear_assign_kernel_id;
                this->m_clear_be_mon_program_id = clear_assign_program_id;
                success = true;
            }
        }

        return success;
    }

    int ClContext::clearBeamMonitorIoBufferAssignment(
        ClArgument& SIXTRL_RESTRICT_REF beam_elements_arg )
    {
        return ( this->hasClearBeamMonitorIoBufferAssignmentKernel() )
            ? this->clearBeamMonitorIoBufferAssignment(
                beam_elements_arg, this->m_clear_be_mon_kernel_id )
            : -1;

    }

    int ClContext::clearBeamMonitorIoBufferAssignment(
        ClArgument& SIXTRL_RESTRICT_REF beam_elements_arg,
        ClContext::kernel_id_t const clear_assign_kernel_id )
    {
        int success = -1;

        SIXTRL_ASSERT( this->hasSelectedNode() );
        SIXTRL_ASSERT( ( clear_assign_kernel_id >= kernel_id_t{ 0 } ) &&
            ( static_cast< size_type >( clear_assign_kernel_id ) <
              this->numAvailableKernels() ) );

        SIXTRL_ASSERT( beam_elements_arg.usesCObjectBuffer() );
        NS(Buffer)* beam_elements_buffer = beam_elements_arg.ptrCObjectBuffer();
        SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( beam_elements_buffer ) );

        size_type const num_kernel_args = this->kernelNumArgs( clear_assign_kernel_id );
        SIXTRL_ASSERT(  num_kernel_args >= 1u );

        size_type const num_beam_elements = NS(Buffer_get_num_of_objects)(
            beam_elements_buffer );

        this->assignKernelArgument( clear_assign_kernel_id, 0u, beam_elements_arg );

        if( num_kernel_args >= 2u )
        {
            this->assignKernelArgumentClBuffer(
                clear_assign_kernel_id, 1u, this->internalSuccessFlagBuffer() );
        }

        success = ( this->runKernel(
            clear_assign_kernel_id, num_beam_elements ) ) ? 0 : -1;

        if( ( success == 0 ) && ( num_kernel_args > 1u ) )
        {
            cl::CommandQueue* ptr_queue = this->openClQueue();
            SIXTRL_ASSERT( ptr_queue != nullptr );

            int32_t success_flag = int32_t{ -1 };
            cl_int cl_ret = ptr_queue->enqueueReadBuffer(
                this->internalSuccessFlagBuffer(), CL_TRUE, 0,
                sizeof( success_flag ), &success_flag );

            if( cl_ret == CL_SUCCESS )
            {
                success = ( int )success_flag;
            }

            ptr_queue->finish();
        }

        return success;
    }

    /* --------------------------------------------------------------------- */

    bool ClContext::useOptimizedTrackingByDefault() const SIXTRL_NOEXCEPT
    {
        return this->m_use_optimized_tracking;
    }

    void ClContext::enableOptimizedtrackingByDefault()
    {
        if( ( !this->useOptimizedTrackingByDefault() ) &&
            ( !this->hasSelectedNode() ) )
        {
            this->clear();
            this->m_use_optimized_tracking = true;
            this->doInitDefaultPrograms();
            this->doInitDefaultKernels();
        }

        return;
    }

    void ClContext::disableOptimizedTrackingByDefault()
    {
        if( ( this->useOptimizedTrackingByDefault() ) &&
            ( !this->hasSelectedNode() ) )
        {
            this->clear();

            this->m_use_optimized_tracking = false;
            this->doInitDefaultPrograms();
            this->doInitDefaultKernels();
        }

        return;
    }

    bool ClContext::isBeamBeamTrackingEnabled() const SIXTRL_NOEXCEPT
    {
        return this->m_enable_beam_beam;
    }

    void ClContext::enableBeamBeamTracking()
    {
        if( ( !this->isBeamBeamTrackingEnabled() ) &&
            ( !this->hasSelectedNode() ) )
        {
            this->clear();

            this->m_enable_beam_beam = true;
            this->doInitDefaultPrograms();
            this->doInitDefaultKernels();
        }

        return;
    }

    void ClContext::disableBeamBeamTracking()
    {
        if( (  this->isBeamBeamTrackingEnabled() ) &&
            ( !this->hasSelectedNode() ) )
        {
            this->clear();

             this->m_enable_beam_beam = false;
            this->doInitDefaultPrograms();
            this->doInitDefaultKernels();
        }

        return;
    }


    bool ClContext::doInitDefaultPrograms()
    {
        return ( ( ClContextBase::doInitDefaultPrograms() ) &&
                 ( this->doInitDefaultProgramsPrivImpl() ) );
    }

    bool ClContext::doInitDefaultKernels()
    {
        return ( ( ClContextBase::doInitDefaultKernels() ) &&
                 ( this->doInitDefaultKernelsPrivImpl() ) );
    }

    bool ClContext::doInitDefaultProgramsPrivImpl()
    {
        bool success = false;

        std::string path_to_kernel_dir( NS(PATH_TO_BASE_DIR) );
        path_to_kernel_dir += "sixtracklib/opencl/kernels/";

        std::string path_to_particles_track_prog     = path_to_kernel_dir;
        std::string path_to_particles_track_opt_prog = path_to_kernel_dir;
        std::string path_to_assign_io_buffer_prog    = path_to_kernel_dir;

        if( !this->debugMode() )
        {
            path_to_particles_track_prog     +=
                "track_particles.cl";

            path_to_particles_track_opt_prog +=
                "track_particles_optimized_priv_particles.cl";

            path_to_assign_io_buffer_prog    +=
                "be_monitors_assign_io_buffer.cl";
        }
        else
        {
            path_to_particles_track_prog     +=
                "track_particles_debug.cl";

            path_to_particles_track_opt_prog +=
                "track_particles_optimized_priv_particles_debug.cl";

            path_to_assign_io_buffer_prog    +=
                "be_monitors_assign_io_buffer_debug.cl";
        }

        std::string track_compile_options = "-D_GPUCODE=1";
        track_compile_options += " -DSIXTRL_BUFFER_ARGPTR_DEC=__private";
        track_compile_options += " -DSIXTRL_BUFFER_DATAPTR_DEC=__global";
        track_compile_options += " -DSIXTRL_PARTICLE_ARGPTR_DEC=__global";
        track_compile_options += " -DSIXTRL_PARTICLE_DATAPTR_DEC=__global";

        if( !this->isBeamBeamTrackingEnabled() )
        {
            track_compile_options += " -DSIXTRL_DISABLE_BEAM_BEAM=1";
        }

        track_compile_options += " -I";
        track_compile_options += NS(PATH_TO_SIXTRL_INCLUDE_DIR);

        std::string track_optimized_compile_options = "-D_GPUCODE=1";
        track_optimized_compile_options += " -DSIXTRL_BUFFER_ARGPTR_DEC=__private";
        track_optimized_compile_options += " -DSIXTRL_BUFFER_DATAPTR_DEC=__global";
        track_optimized_compile_options += " -DSIXTRL_PARTICLE_ARGPTR_DEC=__private";
        track_optimized_compile_options += " -DSIXTRL_PARTICLE_DATAPTR_DEC=__private";

        if( !this->isBeamBeamTrackingEnabled() )
        {
            track_optimized_compile_options += " -DSIXTRL_DISABLE_BEAM_BEAM=1";
        }

        track_optimized_compile_options += " -I";
        track_optimized_compile_options += NS(PATH_TO_SIXTRL_INCLUDE_DIR);

        std::string assign_io_buffer_compile_options = " -D_GPUCODE=1";
        assign_io_buffer_compile_options += " -DSIXTRL_BUFFER_ARGPTR_DEC=__private";
        assign_io_buffer_compile_options += " -DSIXTRL_BUFFER_DATAPTR_DEC=__global";
        assign_io_buffer_compile_options += " -DSIXTRL_PARTICLE_ARGPTR_DEC=__global";
        assign_io_buffer_compile_options += " -DSIXTRL_PARTICLE_DATAPTR_DEC=__global";
        assign_io_buffer_compile_options += " -I";
        assign_io_buffer_compile_options += NS(PATH_TO_SIXTRL_INCLUDE_DIR);

        program_id_t const track_program_id = this->addProgramFile(
            path_to_particles_track_prog, track_compile_options );

        program_id_t const track_optimized_program_id = this->addProgramFile(
            path_to_particles_track_opt_prog, track_optimized_compile_options );

        program_id_t const io_buffer_program_id = this->addProgramFile(
            path_to_assign_io_buffer_prog, assign_io_buffer_compile_options );

        if( ( track_program_id            >= program_id_t{ 0 } ) &&
            ( track_optimized_program_id  >= program_id_t{ 0 } ) &&
            ( io_buffer_program_id        >= program_id_t{ 0 } ) )
        {
            if( !this->useOptimizedTrackingByDefault() )
            {
                this->m_track_until_turn_program_id   = track_program_id;
                this->m_track_single_turn_program_id  = track_program_id;
                this->m_track_elem_by_elem_program_id = track_program_id;
            }
            else
            {
                this->m_track_until_turn_program_id   = track_optimized_program_id;
                this->m_track_single_turn_program_id  = track_optimized_program_id;
                this->m_track_elem_by_elem_program_id = track_optimized_program_id;
            }

            this->m_assign_be_mon_io_buffer_program_id = io_buffer_program_id;
            this->m_clear_be_mon_program_id            = io_buffer_program_id;

            success = true;
        }

        return success;
    }

    bool ClContext::doInitDefaultKernelsPrivImpl()
    {
        bool success = false;

        if( this->hasSelectedNode() )
        {
            program_id_t const max_program_id = static_cast< program_id_t >(
                this->numAvailablePrograms() );

            if( ( this->m_track_until_turn_program_id >= program_id_t{ 0 } ) &&
                ( this->m_track_until_turn_program_id <  max_program_id ) )
            {
                std::string kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
                kernel_name += "Track_particles_until_turn";

                if( this->useOptimizedTrackingByDefault() )
                {
                    kernel_name += "_opt_pp";
                }

                if( this->debugMode() )
                {
                    kernel_name += "_debug";
                }

                kernel_name += "_opencl";

                kernel_id_t const kernel_id = this->enableKernel(
                    kernel_name.c_str(), this->m_track_until_turn_program_id );

                if( kernel_id >= kernel_id_t{ 0 } )
                {
                    success = this->setTrackingKernelId( kernel_id );
                }
            }

            if( ( success ) &&
                ( this->m_track_single_turn_program_id >= program_id_t{ 0 } ) &&
                ( this->m_track_single_turn_program_id <  max_program_id ) )
            {
                std::string kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
                kernel_name += "Track_particles_single_turn";

                if( this->useOptimizedTrackingByDefault() )
                {
                    kernel_name += "_opt_pp";
                }

                if( this->debugMode() )
                {
                    kernel_name += "_debug";
                }

                kernel_name += "_opencl";

                kernel_id_t const kernel_id = this->enableKernel(
                    kernel_name.c_str(), this->m_track_single_turn_program_id );

                if( kernel_id >= kernel_id_t{ 0 } )
                {
                    success = this->setSingleTurnTrackingKernelId( kernel_id );
                }
            }

            if( ( success ) &&
                ( this->m_track_elem_by_elem_program_id >= program_id_t{ 0 } ) &&
                ( this->m_track_elem_by_elem_program_id <  max_program_id ) )
            {
                std::string kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
                kernel_name += "Track_particles_elem_by_elem";

                if( this->useOptimizedTrackingByDefault() )
                {
                    kernel_name += "_opt_pp";
                }

                if( this->debugMode() )
                {
                    kernel_name += "_debug";
                }

                kernel_name += "_opencl";

                kernel_id_t const kernel_id = this->enableKernel(
                    kernel_name.c_str(), this->m_track_elem_by_elem_program_id );

                if( kernel_id >= kernel_id_t{ 0 } )
                {
                    success = this->setElementByElementTrackingKernelId( kernel_id );
                }
            }

            if( ( success ) &&
                ( this->m_assign_be_mon_io_buffer_program_id >= program_id_t{ 0 } ) &&
                ( this->m_assign_be_mon_io_buffer_program_id <  max_program_id ) )
            {
                std::string kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
                kernel_name += "BeamMonitor_assign_io_buffer_from_offset";

                if( this->debugMode() )
                {
                    kernel_name += "_debug";
                }

                kernel_name += "_opencl";

                kernel_id_t const kernel_id = this->enableKernel(
                    kernel_name.c_str(), this->m_assign_be_mon_io_buffer_program_id );

                if( kernel_id >= kernel_id_t{ 0 } )
                {
                    success = this->setAssignBeamMonitorIoBufferKernelId( kernel_id );
                }
            }

            if( ( success ) &&
                ( this->m_clear_be_mon_program_id >= program_id_t{ 0 } ) &&
                ( this->m_clear_be_mon_program_id <  max_program_id ) )
            {
                std::string kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
                kernel_name += "BeamMonitor_clear_all_line_obj";

                if( this->debugMode() )
                {
                    kernel_name += "_debug";
                }

                kernel_name += "_opencl";

                kernel_id_t const kernel_id = this->enableKernel(
                    kernel_name.c_str(), this->m_clear_be_mon_program_id );

                if( kernel_id >= kernel_id_t{ 0 } )
                {
                    success = this->setClearBeamMonitorIoBufferAssignmentKernelId(
                        kernel_id );
                }
            }
        }

        return success;
    }
}

#endif /* defined( __cplusplus ) */

/* ========================================================================= */

SIXTRL_HOST_FN NS(ClContext)* NS(ClContext_create)()
{
    return new SIXTRL_CXX_NAMESPACE::ClContext;
}

SIXTRL_HOST_FN NS(ClContext)* NS(ClContext_new)( const char* node_id_str )
{
    return new SIXTRL_CXX_NAMESPACE::ClContext( node_id_str );
}

SIXTRL_HOST_FN void NS(ClContext_delete)( NS(ClContext)* SIXTRL_RESTRICT ctx )
{
    delete ctx;
}

SIXTRL_HOST_FN void NS(ClContext_clear)( NS(ClContext)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) ctx->clear();
    return;
}

SIXTRL_HOST_FN bool NS(ClContext_has_tracking_kernel)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->hasTrackingKernel() : false;
}


SIXTRL_HOST_FN int NS(ClContext_get_tracking_kernel_id)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->trackingKernelId() : -1;
}

SIXTRL_HOST_FN bool NS(ClContext_set_tracking_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, int const tracking_kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->setTrackingKernelId( tracking_kernel_id ) : false;
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(ClContext_continue_tracking)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(context_num_turns_t) const until_turn )
{
    return ( ctx != nullptr ) ? ctx->track( until_turn ) : -1;
}

SIXTRL_HOST_FN int NS(ClContext_continue_tracking_with_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, int const track_kernel_id,
    NS(context_num_turns_t) const until_turn )
{
    return ( ctx != nullptr ) ? ctx->track( until_turn, track_kernel_id ) : -1;
}

SIXTRL_HOST_FN int NS(ClContext_track)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_particles_arg,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_beam_elements_arg,
    NS(context_num_turns_t) const until_turn )
{
    return ( ( ctx != nullptr ) && ( ptr_particles_arg != nullptr ) &&
             ( ptr_beam_elements_arg != nullptr ) )
        ? ctx->track( *ptr_particles_arg, *ptr_beam_elements_arg, until_turn )
        : -1;
}

SIXTRL_HOST_FN int NS(ClContext_track_with_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_particles_arg,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_beam_elements_arg,
    NS(context_num_turns_t) const until_turn, int const tracking_kernel_id )
{
    return ( ( ctx != nullptr ) && ( ptr_particles_arg != nullptr ) &&
             ( ptr_beam_elements_arg != nullptr ) )
        ? ctx->track( *ptr_particles_arg,
                      *ptr_beam_elements_arg, until_turn, tracking_kernel_id )
        : -1;
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN bool NS(ClContext_has_single_turn_tracking_kernel)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->hasSingleTurnTrackingKernel() : false;
}

SIXTRL_HOST_FN int NS(ClContext_get_single_turn_tracking_kernel_id)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->singleTurnTackingKernelId() : -1;
}

SIXTRL_HOST_FN bool NS(ClContext_set_single_turn_tracking_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, int const tracking_kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->setSingleTurnTrackingKernelId( tracking_kernel_id ) : false;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_HOST_FN int NS(ClContext_continue_tracking_single_turn)(
    NS(ClContext)* SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->trackSingleTurn() : -1;
}

SIXTRL_HOST_FN int NS(ClContext_continue_tracking_single_turn_with_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, int const kernel_id )
{
    return ( ctx != nullptr ) ? ctx->trackSingleTurn( kernel_id ) : -1;
}

SIXTRL_HOST_FN int NS(ClContext_track_single_turn)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_particles_arg,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_beam_elements_arg )
{
    return ( ( ctx != nullptr ) && ( ptr_particles_arg != nullptr ) &&
             ( ptr_beam_elements_arg != nullptr ) )
        ? ctx->trackSingleTurn( *ptr_particles_arg, *ptr_beam_elements_arg )
        : -1;
}

SIXTRL_HOST_FN int NS(ClContext_track_single_turn_with_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_particles_arg,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_beam_elements_arg,
    int const tracking_kernel_id )
{
    return ( ( ctx != nullptr ) && ( ptr_particles_arg != nullptr ) &&
             ( ptr_beam_elements_arg != nullptr ) )
        ? ctx->trackSingleTurn( *ptr_particles_arg, *ptr_beam_elements_arg,
                                 tracking_kernel_id ) : -1;
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN bool NS(ClContext_has_element_by_element_tracking_kernel)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->hasElementByElementTrackingKernel() : false;
}

SIXTRL_HOST_FN int NS(ClContext_get_element_by_element_tracking_kernel_id)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return (ctx != nullptr ) ? ctx->elementByElementTrackingKernelId() : -1;
}

SIXTRL_HOST_FN bool NS(ClContext_set_element_by_element_tracking_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, int const tracking_kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->setElementByElementTrackingKernelId( tracking_kernel_id )
        : false;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_HOST_FN int NS(ClContext_continue_tracking_element_by_element)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(buffer_size_t) const io_particle_block_offset )
{
    return ( ctx != nullptr )
        ? ctx->trackElementByElement( io_particle_block_offset )
        : -1;
}

SIXTRL_HOST_FN int NS(ClContext_continue_tracking_element_by_element_with_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(buffer_size_t) const io_particle_block_offset,
    int const kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->trackElementByElement( io_particle_block_offset, kernel_id )
        : -1;
}

SIXTRL_HOST_FN int NS(ClContext_track_element_by_element)(
    NS(ClContext)*  SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_particles_arg,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_beam_elements_arg,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_elem_by_elem_buffer_arg,
    NS(buffer_size_t) const io_particle_block_offset )
{
    return ( ( ctx != nullptr ) && ( ptr_particles_arg != nullptr ) &&
             ( ptr_beam_elements_arg != nullptr ) &&
             ( ptr_elem_by_elem_buffer_arg != nullptr ) )
        ? ctx->trackElementByElement( *ptr_particles_arg, *ptr_beam_elements_arg,
                                      *ptr_elem_by_elem_buffer_arg,
                                      io_particle_block_offset )
        : -1;
}

SIXTRL_HOST_FN int NS(ClContext_track_element_by_element_with_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_particles_arg,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_beam_elements_arg,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_elem_by_elem_buffer_arg,
    NS(buffer_size_t) const io_particle_block_offset,
    int const tracking_kernel_id )
{
    return ( ( ctx != nullptr ) && ( ptr_particles_arg != nullptr ) &&
             ( ptr_beam_elements_arg != nullptr ) &&
             ( ptr_elem_by_elem_buffer_arg != nullptr ) )
        ? ctx->trackElementByElement( *ptr_particles_arg, *ptr_beam_elements_arg,
                                      *ptr_elem_by_elem_buffer_arg,
                                      io_particle_block_offset, tracking_kernel_id )
        : -1;
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN bool NS(ClContext_has_assign_beam_monitor_io_buffer_kernel)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return (ctx != nullptr ) ? ctx->hasAssignBeamMonitorIoBufferKernel() : false;
}

SIXTRL_HOST_FN int NS(ClContext_get_assign_beam_monitor_io_buffer_kernel_id)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->assignBeamMonitorIoBufferKernelId() : -1;
}

SIXTRL_HOST_FN bool NS(ClContext_set_assign_beam_monitor_io_buffer_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, int const assign_kernel_id )
{
    return ( ctx != nullptr )
        ? ctx->setAssignBeamMonitorIoBufferKernelId( assign_kernel_id )
        : false;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_HOST_FN int NS(ClContext_assign_beam_monitor_io_buffer)(
    NS(ClContext)*  SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_beam_elements_arg,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_io_buffer_arg,
    NS(buffer_size_t) const num_particles,
    NS(buffer_size_t) const io_particle_block_offset )
{
    return ( ( ctx != nullptr ) && ( ptr_beam_elements_arg != nullptr ) &&
             ( ptr_io_buffer_arg != nullptr ) )
        ? ctx->assignBeamMonitorIoBuffer(
            *ptr_beam_elements_arg, *ptr_io_buffer_arg,
            num_particles, io_particle_block_offset )
        : -1;
}

SIXTRL_HOST_FN int NS(ClContext_assign_beam_monitor_io_buffer_with_kernel_id)(
    NS(ClContext)*  SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_beam_elements_arg,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_io_buffer_arg,
    NS(buffer_size_t) const num_particles,
    NS(buffer_size_t) const io_particle_block_offset,
    int const assign_kernel_id )
{
    return ( ( ctx != nullptr ) && ( ptr_beam_elements_arg != nullptr ) &&
             ( ptr_io_buffer_arg != nullptr ) )
        ? ctx->assignBeamMonitorIoBuffer(
            *ptr_beam_elements_arg, *ptr_io_buffer_arg,
            num_particles, io_particle_block_offset, assign_kernel_id )
        : -1;
}

SIXTRL_HOST_FN bool NS(ClContext_has_clear_beam_monitor_io_assignment_kernel)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->hasClearBeamMonitorIoBufferAssignmentKernel()
        : false;
}

SIXTRL_HOST_FN int NS(ClContext_get_clear_beam_monitor_io_assignment_kernel_id)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return (ctx != nullptr )
        ? ctx->clearBeamMonitorIoBufferAssignmentKernelId() : -1;
}

SIXTRL_HOST_FN bool NS(ClContext_set_clear_beam_monitor_io_assignment_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, int const kernel_id )
{
    bool success = false;

    if( ctx != nullptr )
    {
        success = ctx->setClearBeamMonitorIoBufferAssignmentKernelId( kernel_id );
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_HOST_FN int NS(ClContext_clear_beam_monitor_io_assignment)(
    NS(ClContext)*  SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT beam_elements_arg )
{
    return ( ( ctx != nullptr ) && ( beam_elements_arg != nullptr ) )
        ? ctx->clearBeamMonitorIoBufferAssignment( *beam_elements_arg )
        : -1;
}

SIXTRL_HOST_FN int NS(ClContext_clear_beam_monitor_io_assignment_with_kernel)(
    NS(ClContext)*  SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT beam_elements_arg,
    int const kernel_id )
{
    return ( ( ctx != nullptr ) && ( beam_elements_arg != nullptr ) )
        ? ctx->clearBeamMonitorIoBufferAssignment(
            *beam_elements_arg, kernel_id )
        : -1;
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN bool NS(ClContext_uses_optimized_tracking_by_default)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->useOptimizedTrackingByDefault() : false;
}

SIXTRL_HOST_FN void NS(ClContext_enable_optimized_tracking_by_default)(
    NS(ClContext)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) ctx->enableOptimizedtrackingByDefault();
    return;
}

SIXTRL_HOST_FN void NS(ClContext_disable_optimized_tracking_by_default)(
    NS(ClContext)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) ctx->disableOptimizedTrackingByDefault();
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN bool NS(ClContext_is_beam_beam_tracking_enabled)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->isBeamBeamTrackingEnabled() : false;
}


SIXTRL_HOST_FN void NS(ClContext_enable_beam_beam_tracking)(
    NS(ClContext)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr )
    {
        ctx->enableBeamBeamTracking();
    }

    return;
}

SIXTRL_HOST_FN void NS(ClContext_disable_beam_beam_tracking)(
    NS(ClContext)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr )
    {
        ctx->disableBeamBeamTracking();
    }

    return;
}

#endif /* !defined( __CUDACC__ ) */

/* end: sixtracklib/opencl/internal/context.cpp */
