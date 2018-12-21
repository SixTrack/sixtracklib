#include "sixtracklib/opencl/track_job_cl.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>

#if !defined( SIXTRL_NO_INCLUDES )
    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* defined( __cplusplus ) */

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/internal/track_job_base.h"
    #include "sixtracklib/common/output/output_buffer.h"

    #include "sixtracklib/opencl/context.h"
    #include "sixtracklib/opencl/argument.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    TrackJobCl::TrackJobCl(
        char const* SIXTRL_RESTRICT device_id_str,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        TrackJobCl::size_type const until_turn,
        TrackJobCl::size_type const num_elem_by_elem_turns ) :
        TrackJobCl::_base_t(),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr ),
        m_elem_by_elem_index_offset( TrackJobCl::size_type{ 0 } ),
        m_beam_monitor_index_offset( TrackJobCl::size_type{ 0 } ),
        m_particle_block_idx( TrackJobCl::size_type{ 0 } ),
        m_owns_output_buffer( false )
    {
        using size_t        = TrackJobCl::size_type;
        using index_t       = NS(particle_index_t);
        using cl_context_t  = TrackJobCl::cl_context_t;
        using ptr_context_t = TrackJobCl::ptr_cl_context_t;
        using c_buffer_t    = TrackJobCl::c_buffer_t;

        bool success = false;

        c_buffer_t* output_buffer  = ::NS(Buffer_new)( size_t{ 0 } );
        success = ( output_buffer != nullptr );
        this->m_owns_output_buffer = true;

        size_t const slot_size =  ::NS(Buffer_get_slot_size)( output_buffer );
        success &= ( slot_size == ::NS(Buffer_get_slot_size)(
            particles_buffer ) );

        success &= ( slot_size == ::NS(Buffer_get_slot_size)(
            beam_elements_buffer ) );

        ptr_context_t ptr_context( new cl_context_t( device_id_str ) );
        this->m_ptr_context = std::move( ptr_context );
        success &= ( this->m_ptr_context.get() != nullptr );

        index_t min_turn_id = index_t{ -1 };
        success &= this->doInitBuffersClImpl( particles_buffer,
            beam_elements_buffer, output_buffer, num_elem_by_elem_turns,
            until_turn, &this->m_particle_block_idx, size_t{ 1 },
            &this->m_elem_by_elem_index_offset,
            &this->m_beam_monitor_index_offset, &min_turn_id );

        ::NS(Particles)* particles = ::NS(Particles_buffer_get_particles)(
            particles_buffer, this->m_particle_block_idx );
        success &= ( particles != nullptr );

        bool performed_tracking = false;

        if( num_elem_by_elem_turns > ( size_t{ 0 } ) )
        {
            performed_tracking = true;

            success &= ( 0 == ::NS(ClContext_track_element_by_element)(
                this->m_ptr_context.get(),
                this->m_ptr_particles_buffer_arg.get(),
                this->m_ptr_beam_elements_buffer_arg.get(),
                this->m_ptr_output_buffer_arg.get(),
                num_elem_by_elem_turns, this->m_elem_by_elem_index_offset ) );
        }

        if( until_turn > size_t{ 0 } )
        {
            performed_tracking = true;

            success &= ( 0 == ::NS(ClContext_track)( this->m_ptr_context.get(),
                this->m_ptr_particles_buffer_arg.get(),
                this->m_ptr_beam_elements_buffer_arg.get(), until_turn ) );
        }

        if( ( success ) && ( performed_tracking ) )
        {
            unsigned char* out_begin_data_ptr = ::NS(Buffer_get_data_begin)(
                    this->m_ptr_beam_elements_buffer_arg->ptrCObjectBuffer() );

            success &= ( 0 == ::NS(ManagedBuffer_remap)(
                out_begin_data_ptr, slot_size ) );
        }

        SIXTRL_ASSERT( success );
        ( void )success;
    }

    TrackJobCl::TrackJobCl(
        char const* SIXTRL_RESTRICT device_id_str,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        TrackJobCl::size_type const until_turn,
        TrackJobCl::size_type const num_elem_by_elem_turns ) :
        TrackJobCl::_base_t(),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr ),
        m_elem_by_elem_index_offset( TrackJobCl::size_type{ 0 } ),
        m_beam_monitor_index_offset( TrackJobCl::size_type{ 0 } ),
        m_particle_block_idx( TrackJobCl::size_type{ 0 } ),
        m_owns_output_buffer( false )
    {
        using size_t        = TrackJobCl::size_type;
        using index_t       = NS(particle_index_t);
        using cl_context_t  = TrackJobCl::cl_context_t;
        using ptr_context_t = TrackJobCl::ptr_cl_context_t;

        bool success = ( output_buffer != nullptr );

        size_t const slot_size =  ::NS(Buffer_get_slot_size)( output_buffer );
        success &= ( slot_size == ::NS(Buffer_get_slot_size)(
            particles_buffer ) );

        success &= ( slot_size == ::NS(Buffer_get_slot_size)(
            beam_elements_buffer ) );

        ptr_context_t ptr_context( new cl_context_t( device_id_str ) );
        this->m_ptr_context = std::move( ptr_context );
        success &= ( this->m_ptr_context.get() != nullptr );

        index_t min_turn_id = index_t{ -1 };
        success &= this->doInitBuffersClImpl( particles_buffer,
            beam_elements_buffer, output_buffer, num_elem_by_elem_turns,
            until_turn, &this->m_particle_block_idx, size_t{ 1 },
            &this->m_elem_by_elem_index_offset,
            &this->m_beam_monitor_index_offset, &min_turn_id );

        ::NS(Particles)* particles = ::NS(Particles_buffer_get_particles)(
            particles_buffer, this->m_particle_block_idx );
        success &= ( particles != nullptr );

         bool performed_tracking = false;

        if( num_elem_by_elem_turns > ( size_t{ 0 } ) )
        {
            performed_tracking = true;

            success &= ( 0 == ::NS(ClContext_track_element_by_element)(
                this->m_ptr_context.get(),
                this->m_ptr_particles_buffer_arg.get(),
                this->m_ptr_beam_elements_buffer_arg.get(),
                this->m_ptr_output_buffer_arg.get(),
                num_elem_by_elem_turns, this->m_elem_by_elem_index_offset ) );
        }

        if( until_turn > size_t{ 0 } )
        {
            performed_tracking = true;

            success &= ( 0 == ::NS(ClContext_track)( this->m_ptr_context.get(),
                this->m_ptr_particles_buffer_arg.get(),
                this->m_ptr_beam_elements_buffer_arg.get(), until_turn ) );
        }

        if( ( success ) && ( performed_tracking ) )
        {
            unsigned char* out_begin_data_ptr = ::NS(Buffer_get_data_begin)(
                    this->m_ptr_beam_elements_buffer_arg->ptrCObjectBuffer() );

            success &= ( 0 == ::NS(ManagedBuffer_remap)(
                out_begin_data_ptr, slot_size ) );
        }

        SIXTRL_ASSERT( success );
        ( void )success;
    }

    TrackJobCl::~TrackJobCl()
    {
        if( this->m_owns_output_buffer )
        {
            ::NS(Buffer_delete)( this->doGetPtrOutputBuffer() );
            this->doSetPtrToOutputBuffer( nullptr );
        }
    }

    bool TrackJobCl::track( TrackJobCl::size_type const until_turn )
    {
        using size_t = TrackJobCl::size_type;

        size_t const slot_size = ::NS(Buffer_get_slot_size)(
            this->doGetPtrOutputBuffer() );

        bool success = ( this->doGetPtrOutputBuffer() != nullptr );
        success &= ( slot_size > size_t{ 0 } );
        success &= ( this->m_ptr_context.get() != nullptr );

        success &= ( 0 == ::NS(ClContext_track)( this->m_ptr_context.get(),
                this->m_ptr_particles_buffer_arg.get(),
                this->m_ptr_beam_elements_buffer_arg.get(),
                until_turn ) );

        return success;
    }

    bool TrackJobCl::trackElemByElem( TrackJobCl::size_type const until_turn )
    {
        ( void )until_turn;
        return false;
    }

    void TrackJobCl::collect()
    {
        using size_t = TrackJobCl::size_type;

        SIXTRL_ASSERT( this->m_ptr_context.get() != nullptr );

        SIXTRL_ASSERT( this->m_ptr_particles_buffer_arg.get() != nullptr );
        SIXTRL_ASSERT( this->m_ptr_particles_buffer_arg->usesCObjectBuffer() );
        SIXTRL_ASSERT( this->m_ptr_particles_buffer_arg->ptrCObjectBuffer()
                       != nullptr );
        SIXTRL_ASSERT( this->m_ptr_particles_buffer_arg->ptrCObjectBuffer() ==
                       this->doGetPtrParticlesBuffer() );
        SIXTRL_ASSERT( this->m_ptr_particles_buffer_arg->context() ==
                       this->m_ptr_context.get() );

        SIXTRL_ASSERT( this->m_ptr_beam_elements_buffer_arg.get() != nullptr );
        SIXTRL_ASSERT( this->m_ptr_beam_elements_buffer_arg->usesCObjectBuffer() );
        SIXTRL_ASSERT( this->m_ptr_beam_elements_buffer_arg->ptrCObjectBuffer()
                       != nullptr );
        SIXTRL_ASSERT( this->m_ptr_beam_elements_buffer_arg->ptrCObjectBuffer() ==
                       this->doGetPtrBeamElementsBuffer() );
        SIXTRL_ASSERT( this->m_ptr_beam_elements_buffer_arg->context() ==
                       this->m_ptr_context.get() );

        SIXTRL_ASSERT( this->m_ptr_output_buffer_arg.get() != nullptr );
        SIXTRL_ASSERT( this->m_ptr_output_buffer_arg->usesCObjectBuffer() );
        SIXTRL_ASSERT( this->m_ptr_output_buffer_arg->ptrCObjectBuffer()
                       != nullptr );
        SIXTRL_ASSERT( this->m_ptr_output_buffer_arg->ptrCObjectBuffer() ==
                       this->doGetPtrOutputBuffer() );
        SIXTRL_ASSERT( this->m_ptr_output_buffer_arg->context() ==
                       this->m_ptr_context.get() );

        this->m_ptr_output_buffer_arg->read(
            this->doGetPtrOutputBuffer() );

        this->m_ptr_particles_buffer_arg->read(
            this->doGetPtrParticlesBuffer() );

        this->m_ptr_beam_elements_buffer_arg->read(
            this->doGetPtrBeamElementsBuffer() );

        size_t const slot_size = ::NS(Buffer_get_slot_size)(
            this->doGetPtrOutputBuffer() );

        bool success = ( slot_size > size_t{ 0 } );

        success &= ( slot_size == ::NS(Buffer_get_slot_size)(
            this->doGetPtrParticlesBuffer() ) );

        success &= ( slot_size == ::NS(Buffer_get_slot_size)(
            this->doGetPtrBeamElementsBuffer() ) );

        unsigned char* output_buffer_data_begin =
            ::NS(Buffer_get_data_begin)( this->doGetPtrOutputBuffer() );

        unsigned char* particles_buffer_data_begin =
            ::NS(Buffer_get_data_begin)( this->doGetPtrParticlesBuffer() );

        unsigned char* beam_elements_data_begin =
            ::NS(Buffer_get_data_begin)( this->doGetPtrBeamElementsBuffer() );

        success &= ( 0 == ::NS(ManagedBuffer_remap)(
            output_buffer_data_begin, slot_size ) );

        success &= ( 0 == ::NS(ManagedBuffer_remap)(
            particles_buffer_data_begin, slot_size ) );

        success &= ( 0 == ::NS(ManagedBuffer_remap)(
            beam_elements_data_begin, slot_size ) );

        SIXTRL_ASSERT( success );
        ( void )success;

        return;
    }

    TrackJobCl::cl_context_t& TrackJobCl::context() SIXTRL_RESTRICT
    {
        SIXTRL_ASSERT( this->m_ptr_context.get() != nullptr );
        return *this->m_ptr_context;
    }

    TrackJobCl::cl_context_t const& TrackJobCl::context() const SIXTRL_RESTRICT
    {
        SIXTRL_ASSERT( this->m_ptr_context.get() != nullptr );
        return *this->m_ptr_context;
    }

    ::NS(ClContext)* TrackJobCl::ptrContext() SIXTRL_RESTRICT
    {
        return this->m_ptr_context.get();
    }

    ::NS(ClContext) const*  TrackJobCl::ptrContext() const SIXTRL_RESTRICT
    {
        return this->m_ptr_context.get();
    }

    bool TrackJobCl::doInitBuffers(
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT belements_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        TrackJobCl::size_type const num_elem_by_elem_turns,
        TrackJobCl::size_type const until_turn,
        TrackJobCl::size_type const* SIXTRL_RESTRICT particle_blk_idx_begin,
        TrackJobCl::size_type const  particle_blk_idx_length,
        TrackJobCl::size_type* SIXTRL_RESTRICT ptr_elem_by_elem_index_offset,
        TrackJobCl::size_type* SIXTRL_RESTRICT ptr_beam_monitor_index_offset,
        TrackJobCl::particle_index_t* SIXTRL_RESTRICT ptr_min_turn_id )
    {
        return this->doInitBuffersClImpl(
            particles_buffer, belements_buffer, output_buffer,
            num_elem_by_elem_turns, until_turn, particle_blk_idx_begin,
            particle_blk_idx_length, ptr_elem_by_elem_index_offset,
            ptr_beam_monitor_index_offset, ptr_min_turn_id );
    }

    bool TrackJobCl::doInitBuffersClImpl(
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT belements_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        TrackJobCl::size_type const num_elem_by_elem_turns,
        TrackJobCl::size_type const until_turn,
        TrackJobCl::size_type const* SIXTRL_RESTRICT particle_blk_idx_begin,
        TrackJobCl::size_type const  particle_blk_idx_length,
        TrackJobCl::size_type* SIXTRL_RESTRICT ptr_elem_by_elem_index_offset,
        TrackJobCl::size_type* SIXTRL_RESTRICT ptr_beam_monitor_index_offset,
        TrackJobCl::particle_index_t* SIXTRL_RESTRICT ptr_min_turn_id )
    {
        using size_t  = TrackJobCl::size_type;
        using index_t = NS(particle_index_t);

        size_t elem_by_elem_index_offset = size_t{ 0 };
        size_t beam_monitor_index_offset = size_t{ 0 };
        index_t min_turn_id = index_t{ -1 };

        bool success = ( this->m_ptr_context.get()  != nullptr );
        success &= ( particle_blk_idx_begin  != nullptr );
        success &= ( particle_blk_idx_length == size_t{ 1 } );

        ptr_cl_arg_t ptr_particles_buffer(
            new cl_arg_t( particles_buffer, this->m_ptr_context.get() ) );
        this->m_ptr_particles_buffer_arg = std::move( ptr_particles_buffer );
        success &= ( this->m_ptr_particles_buffer_arg.get() != nullptr );

        ::NS(Particles)* particles = ::NS(Particles_buffer_get_particles)(
            particles_buffer, this->m_particle_block_idx );
        success &= ( particles != nullptr );

        ptr_cl_arg_t ptr_beam_elements_buffer(
            new cl_arg_t( belements_buffer, this->m_ptr_context.get() ) );
        this->m_ptr_beam_elements_buffer_arg = std::move( ptr_beam_elements_buffer );
        success &= ( this->m_ptr_beam_elements_buffer_arg.get() != nullptr );
        this->doSetPtrToBeamElementsBuffer( belements_buffer );

        success &= ( 0 == ::NS(OutputBuffer_prepare)( belements_buffer,
            output_buffer, particles, num_elem_by_elem_turns,
            &elem_by_elem_index_offset, &beam_monitor_index_offset,
            &min_turn_id ) );

        ptr_cl_arg_t ptr_output_buffer(
            new cl_arg_t( output_buffer, this->m_ptr_context.get() ) );

        this->m_ptr_output_buffer_arg = std::move( ptr_output_buffer );
        success &= ( this->m_ptr_output_buffer_arg.get() != nullptr );

        success &= ( 0 == ::NS(ClContext_assign_beam_monitor_out_buffer)(
                this->m_ptr_context.get(),
                this->m_ptr_beam_elements_buffer_arg.get(),
                this->m_ptr_output_buffer_arg.get(),
                min_turn_id, beam_monitor_index_offset ) );

        if( success )
        {
            if(  ptr_elem_by_elem_index_offset != nullptr )
            {
                *ptr_elem_by_elem_index_offset  = elem_by_elem_index_offset;
            }

            if(  ptr_beam_monitor_index_offset != nullptr )
            {
                *ptr_beam_monitor_index_offset  = beam_monitor_index_offset;
            }

            if(  ptr_min_turn_id != nullptr )
            {
                *ptr_min_turn_id  = min_turn_id;
            }

            this->doSetPtrToParticlesBuffer( particles_buffer );
            this->doSetPtrToBeamElementsBuffer( belements_buffer );
            this->doSetPtrToOutputBuffer( output_buffer );
        }

        return success;
    }
}


SIXTRL_HOST_FN NS(TrackJobCl)* NS(TrackJobCl_new)(
    char const* SIXTRL_RESTRICT device_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(buffer_size_t) const num_elem_by_elem_turns,
    NS(buffer_size_t) const until_turn )
{
    using track_job_t = SIXTRL_CXX_NAMESPACE::TrackJobCl;
    track_job_t* track_job = new track_job_t(
        device_id_str, particles_buffer, beam_elements_buffer,
        num_elem_by_elem_turns, until_turn );

    return track_job;
}


SIXTRL_HOST_FN NS(TrackJobCl)*
NS(TrackJobCl_new_using_output_buffer)(
    char const* SIXTRL_RESTRICT device_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const num_elem_by_elem_turns,
    NS(buffer_size_t) const until_turn )
{
    using track_job_t = SIXTRL_CXX_NAMESPACE::TrackJobCl;

    track_job_t* track_job = new track_job_t(
        device_id_str, particles_buffer, beam_elements_buffer,
        output_buffer, num_elem_by_elem_turns, until_turn );

    return track_job;
}

SIXTRL_HOST_FN void NS(TrackJobCl_delete)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job )
{
    delete track_job;
}

SIXTRL_HOST_FN bool NS(TrackJobCl_track)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    NS(buffer_size_t) const until_turn )
{
    return ( track_job != nullptr )
        ? track_job->track( until_turn ) : false;
}

SIXTRL_HOST_FN bool NS(TrackJobCl_track_elem_by_elem)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    NS(buffer_size_t) const until_turn )
{
    return ( track_job != nullptr )
        ? track_job->trackElemByElem( until_turn ) : false;
}

SIXTRL_HOST_FN NS(ClContext)*
NS(TrackJobCl_get_context)( NS(TrackJobCl)* SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr ) ? track_job->ptrContext() : nullptr;
}

SIXTRL_HOST_FN NS(ClContext) const*
NS(TrackJobCl_get_const_context)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr ) ? track_job->ptrContext() : nullptr;
}

SIXTRL_HOST_FN void NS(TrackJobCl_collect)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job )
{
    if ( track_job != nullptr ) track_job->collect();
    return;
}

SIXTRL_HOST_FN NS(Buffer)* NS(TrackJobCl_get_particle_buffer)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr ) ? track_job->ptrParticlesBuffer() : nullptr;
}

SIXTRL_HOST_FN NS(Buffer)* NS(TrackJobCl_get_output_buffer)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr ) ? track_job->ptrOutputBuffer() : nullptr;
}

SIXTRL_HOST_FN NS(Buffer)* NS(TrackJobCl_get_beam_elements_buffer)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr ) ? track_job->ptrBeamElementsBuffer() : nullptr;
}

/* end: /opencl/internal/track_job_cl.cpp */
