#ifndef SIXTRACKLIB_SIXTRACKLIB_COMMON_INTERNAL_TRACK_JOB_BASE_H__
#define SIXTRACKLIB_SIXTRACKLIB_COMMON_INTERNAL_TRACK_JOB_BASE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <algorithm>
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <memory>
        #include <string>
        #include <set>
    #else /* !defined( __cplusplus ) */
        #include <stdbool.h>
        #include <stddef.h>
        #include <stdint.h>
        #include <stdlib.h>
    #endif /* !defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
        #include "sixtracklib/common/particles.hpp"
    #endif /* defined( __cplusplus ) */

    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/context/context_abs_base.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    class TrackJobBase
    {
        public:

        using context_t             = ContextBase;
        using c_buffer_t            = NS(Buffer);
        using buffer_t              = SIXTRL_CXX_NAMESPACE::Buffer;
        using size_type             = SIXTRL_UINT64_T;
        using elem_by_elem_config_t = ::NS(ElemByElemConfig);
        using elem_by_elem_order_t  = ::NS(elem_by_elem_order_t);
        using particles_t           = ::NS(Particles);
        using particle_index_t      = ::NS(particle_index_t);
        using track_status_t        = SIXTRL_INT32_T;

        /* ---------------------------------------------------------------- */

        char const* configString() const SIXTRL_NOEXCEPT;

        /* ---------------------------------------------------------------- */

        bool hasContext() const SIXTRL_NOEXCEPT;
        context_t const* context() const SIXTRL_NOEXCEPT;
        context_t* context() SIXTRL_NOEXCEPT;

        /* ---------------------------------------------------------------- */

        elem_by_elem_config_t const& elemByElemConfig() const SIXTRL_NOEXCEPT;
        elem_by_elem_config_t const*
            ptrElemByElemConfig() const SIXTRL_NOEXCEPT;

        elem_by_elem_order_t elemByElemStoreOrder() const SIXTRL_NOEXCEPT;
        void setElemByElemStoreOrder( elem_by_elem_order_t const order );

        /* ---------------------------------------------------------------- */

        bool track( size_type const until_turn );

        bool track( size_type const until_turn,
                              buffer_t& SIXTRL_RESTRICT_REF output_buffer );

        bool track( size_type const until_turn,
                              c_buffer_t* SIXTRL_RESTRICT_REF output_buffer );

        /* ---------------------------------------------------------------- */

        track_status_t      lastTrackStatus()       const SIXTRL_NOEXCEPT;

        c_buffer_t const*   ptrOutputBuffer()       const SIXTRL_NOEXCEPT;
        buffer_t const&     outputBuffer()          const SIXTRL_NOEXCEPT;

        c_buffer_t const*   ptrParticlesBuffer()    const SIXTRL_NOEXCEPT;
        buffer_t const&     particlesBuffer()       const SIXTRL_NOEXCEPT;

        c_buffer_t const*   ptrBeamElementsBuffer() const SIXTRL_NOEXCEPT;
        buffer_t const&     beamElementsBuffer()    const SIXTRL_NOEXCEPT;

        /* ---------------------------------------------------------------- */

        virtual ~TrackJobBase() = default;

        protected:

        using ptr_context_t                 = std::unique_ptr< context_t >;
        using particle_index_buffer_t       = std::set< size_type >;

        using particle_index_iterator       =
            particle_index_buffer_t::iterator;

        using particle_index_const_iterator =
            particle_index_buffer_t::const_iterator;

        TrackJobBase() SIXTRL_NOEXCEPT;
        explicit TrackJobBase( ptr_context_t&& context );

        TrackJobBase( TrackJobBase const& other ) = default;
        TrackJobBase( TrackJobBase&& other )      = default;

        TrackJobBase& operator=( TrackJobBase const& rhs ) = default;
        TrackJobBase& operator=( TrackJobBase&& rhs )      = default;

        virtual bool doPerformConfig( char const* SIXTRL_RESTRICT config_str );

        virtual bool doInitOutputBuffer(
            c_buffer_t* SIXTRL_RESTRICT output_buffer,
            const c_buffer_t *const SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT belements_buffer,
            size_type const until_turn,
            size_type const num_elem_by_elem_turns,
            size_type const out_buffer_index_offset );

        virtual track_status_t doTrackUntilTurn(
            size_type const until_turn,
            c_buffer_t* SIXTRL_RESTRICT output_buffer );

        virtual track_status_t doTrackElemByElem(
            size_type const elem_by_elem_turns,
            elem_by_elem_config_t const* SIXTRL_RESTRICT elem_by_elem_config,
            c_buffer_t* SIXTRL_RESTRICT output_buffer );

        virtual bool doCollectParticlesBuffer(
            c_buffer_t* SIXTRL_RESTRICT particle_buffer );

        virtual bool doCollectBeamElementsBuffer(
            c_buffer_t* SIXTRL_RESTRICT particle_buffer );

        virtual bool doCollectOutputBuffer(
            c_buffer_t* SIXTRL_RESTRICT particle_buffer );

        particle_index_iterator particleIndexBegin() SIXTRL_NOEXCEPT;
        particle_index_iterator particleIndexEnd()   SIXTRL_NOEXCEPT;

        particle_index_const_iterator
            constParticleIndexBegin() const SIXTRL_NOEXCEPT;

        particle_index_const_iterator
            constParticleIndexEnd() const SIXTRL_NOEXCEPT;

        void doClearParticleIndexBuffer();
        bool doAddParticleIndexToBuffer( size_type const index );

        template< typename IndexIter >
        size_type doAddParticleIndicesToBuffer(
            IndexIter begin, IndexIter end );

        size_type doGetParticleIndexBufferSize() const SIXTRL_NOEXCEPT;

        void doSetContext( ptr_context_t&& ptr_context );
        void doClearContext() SIXTRL_NOEXCEPT;

        bool doSetLastTrackStatus(
            track_status_t const last_status ) SIXTRL_NOEXCEPT;

        elem_by_elem_config_t& doGetElemByElemConfig() SIXTRL_NOEXCEPT;

        elem_by_elem_config_t*
        doGetPtrElemByElemConfig() SIXTRL_NOEXCEPT;

        elem_by_elem_config_t const*
        doGetPtrElemByElemConfig() const SIXTRL_NOEXCEPT;

        buffer_t&         doGetParticlesBuffer()          SIXTRL_RESTRICT;
        buffer_t const&   doGetParticlesBuffer()    const SIXTRL_RESTRICT;
        c_buffer_t*       doGetPtrParticlesBuffer()       SIXTRL_NOEXCEPT;
        c_buffer_t const* doGetPtrParticlesBuffer() const SIXTRL_NOEXCEPT;
        void doSetPtrToParticlesBuffer( c_buffer_t* SIXTRL_RESTRICT buffer );

        buffer_t&         doGetBeamElementsBuffer()          SIXTRL_RESTRICT;
        buffer_t const&   doGetBeamElementsBuffer()    const SIXTRL_RESTRICT;
        c_buffer_t*       doGetPtrBeamElementsBuffer()       SIXTRL_NOEXCEPT;
        c_buffer_t const* doGetPtrBeamElementsBuffer() const SIXTRL_NOEXCEPT;
        void doSetPtrToBeamElementsBuffer( c_buffer_t* SIXTRL_RESTRICT buf );

        buffer_t&         doGetOutputBuffer()          SIXTRL_RESTRICT;
        buffer_t const&   doGetOutputBuffer()    const SIXTRL_RESTRICT;
        c_buffer_t*       doGetPtrOutputBuffer()       SIXTRL_NOEXCEPT;
        c_buffer_t const* doGetPtrOutputBuffer() const SIXTRL_NOEXCEPT;
        void doSetPtrToOutputBuffer( c_buffer_t* SIXTRL_RESTRICT buffer );

        size_type doGetElemByElemOutBufferIndex() const SIXTRL_NOEXCEPT;

        void doSetElemByElemOutBufferIndex(
            size_type const index ) SIXTRL_NOEXCEPT;

        size_type doGetBeamMonitorOutBufferIndexOffset() const SIXTRL_NOEXCEPT;

        void doSetBeamMonitorOutBufferIndexOffset(
            size_type const index ) SIXTRL_NOEXCEPT;

        private:

        bool doPerformConfigBaseImpl(
            char const* SIXTRL_RESTRICT config_str );

        bool doInitOutputBufferBaseImpl(
            c_buffer_t* SIXTRL_RESTRICT output_buffer,
            const c_buffer_t *const SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT belements_buffer,
            size_type const until_turn,
            size_type const num_elem_by_elem_turns,
            size_type const out_buffer_index_offset );

        bool doCollectParticlesBufferBaseImpl(
            c_buffer_t* SIXTRL_RESTRICT buffer );

        bool doCollectBeamElementsBufferBaseImpl(
            c_buffer_t* SIXTRL_RESTRICT buffer );

        bool doCollectOutputBufferBaseImpl(
            c_buffer_t* SIXTRL_RESTRICT buffer );

        bool doTrackUntilTurnBaseImpl(
            size_type until_turn, c_buffer_t* SIXTRL_RESTRICT output_buffer );

        bool doTrackElemByElemBaseImpl(
            size_type num_elem_by_elem_turns,
            elem_by_elem_config_t const* SIXTRL_RESTRICT elem_by_elem_config,
            c_buffer_t* SIXTRL_RESTRICT output_buffer );

        elem_by_elem_config_t       m_elem_by_elem_config;
        std::string                 m_config_str;
        particle_index_buffer_t     m_particle_indices;

        buffer_t                    m_particles_buffer_wrapper;
        buffer_t                    m_beam_elements_buffer_wrapper;
        buffer_t                    m_output_buffer_wrapper;

        c_buffer_t*                 m_ptr_particles_buffer;
        c_buffer_t*                 m_ptr_beam_elements_buffer;
        c_buffer_t*                 m_ptr_output_buffer;

        ptr_context_t               m_ptr_context;
        track_status_t              m_last_track_status;
        elem_by_elem_order_t        m_elem_by_elem_order;
        size_type                   m_elem_by_elem_out_buffer_index;
        size_type                   m_beam_monitor_out_buffer_index_offset;
    };

    /* --------------------------------------------------------------------- */

    template< typename IndexIter >
    TrackJobBase::size_type TrackJobBase::doAddParticleIndicesToBuffer(
        IndexIter begin, IndexIter end )
    {
        using size_t = TrackJobBase::size_type;

        size_t const initial_size = this->m_particle_indices.size();

        if( std::distance( begin, end ) > std::ptrdiff_t{ 0 } )
        {
            this->m_particle_indices.insert( begin, end );
        }

        size_t const final_size = this->m_particle_indices.size();

        return ( final_size > initial_size )
            ? ( final_size - initial_size ) : size_t{ 0 };
    }

    /* --------------------------------------------------------------------- */
    /* Implementation of public, private and protected inline functions:     */
    /* --------------------------------------------------------------------- */

    SIXTRL_INLINE bool TrackJobBase::track(
        TrackJobBase::size_type const until_turn,
        TrackJobBase::buffer_t& SIXTRL_RESTRICT_REF output_buffer )
    {
        return this->track( until_turn, output_buffer.getCApiPtr() );
    }

    SIXTRL_INLINE TrackJobBase::particle_index_iterator
    TrackJobBase::particleIndexBegin() SIXTRL_NOEXCEPT
    {
        return this->m_particle_indices.begin();
    }

    SIXTRL_INLINE TrackJobBase::particle_index_iterator
    TrackJobBase::particleIndexEnd() SIXTRL_NOEXCEPT
    {
        return this->m_particle_indices.end();
    }

    SIXTRL_INLINE  TrackJobBase::particle_index_const_iterator
    TrackJobBase::constParticleIndexBegin() const SIXTRL_NOEXCEPT
    {
        return this->m_particle_indices.begin();
    }

    SIXTRL_INLINE TrackJobBase::particle_index_const_iterator
    TrackJobBase::constParticleIndexEnd() const SIXTRL_NOEXCEPT
    {
        return this->m_particle_indices.end();
    }

    SIXTRL_INLINE void TrackJobBase::doClearParticleIndexBuffer()
    {
        this->m_particle_indices.clear();
        return;
    }

    SIXTRL_INLINE bool TrackJobBase::doAddParticleIndexToBuffer(
        TrackJobBase::size_type const index )
    {
        auto ret = this->m_particle_indices.insert( index );
        return ret.second;
    }

    SIXTRL_INLINE TrackJobBase::size_type
    TrackJobBase::doGetParticleIndexBufferSize() const SIXTRL_NOEXCEPT
    {
        return this->m_particle_indices.size();
    }

    SIXTRL_INLINE void TrackJobBase::doSetContext(
        TrackJobBase::ptr_context_t&& ptr_context )
    {
        this->m_ptr_context = std::move( ptr_context );
        return;
    }

    SIXTRL_INLINE void TrackJobBase::doClearContext() SIXTRL_NOEXCEPT
    {
        this->m_ptr_context.reset( nullptr );
        return;
    }

    SIXTRL_INLINE bool TrackJobBase::doSetLastTrackStatus(
        TrackJobBase::track_status_t const last_status ) SIXTRL_NOEXCEPT
    {
        using status_t = TrackJobBase::track_status_t;
        this->m_last_track_status = last_status;

        return ( this->m_last_track_status == status_t{ 0 } );
    }

    SIXTRL_INLINE TrackJobBase::elem_by_elem_config_t&
    TrackJobBase::doGetElemByElemConfig() SIXTRL_NOEXCEPT
    {
        return this->m_elem_by_elem_config;
    }

    SIXTRL_INLINE TrackJobBase::elem_by_elem_config_t*
    TrackJobBase::doGetPtrElemByElemConfig() SIXTRL_NOEXCEPT
    {
        return &this->m_elem_by_elem_config;
    }

    SIXTRL_INLINE TrackJobBase::elem_by_elem_config_t const*
    TrackJobBase::doGetPtrElemByElemConfig() const SIXTRL_NOEXCEPT
    {
        return &this->m_elem_by_elem_config;
    }

    SIXTRL_INLINE TrackJobBase::buffer_t&
    TrackJobBase::doGetParticlesBuffer() SIXTRL_RESTRICT
    {
        using _this_t = TrackJobBase;
        return const_cast< _this_t::buffer_t& >( static_cast< _this_t const& >(
            *this ).doGetParticlesBuffer() );
    }

    SIXTRL_INLINE TrackJobBase::buffer_t const&
    TrackJobBase::doGetParticlesBuffer() const SIXTRL_RESTRICT
    {
        SIXTRL_ASSERT( this->m_ptr_particles_buffer != nullptr );
        SIXTRL_ASSERT( this->m_particles_buffer_wrapper.getCApiPtr() ==
                       this->m_ptr_particles_buffer );

        return this->m_particles_buffer_wrapper;
    }

    SIXTRL_INLINE TrackJobBase::c_buffer_t*
    TrackJobBase::doGetPtrParticlesBuffer() SIXTRL_NOEXCEPT
    {
        return this->m_ptr_particles_buffer;
    }

    SIXTRL_INLINE TrackJobBase::c_buffer_t const*
    TrackJobBase::doGetPtrParticlesBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_particles_buffer;
    }

    SIXTRL_INLINE void TrackJobBase::doSetPtrToParticlesBuffer(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT buffer )
    {
        this->m_ptr_particles_buffer = buffer;

        if( buffer != nullptr )
        {
            this->m_particles_buffer_wrapper = *buffer;
            SIXTRL_ASSERT( this->m_particles_buffer_wrapper.getCApiPtr() ==
                           this->m_ptr_particles_buffer );
        }
        else
        {
            this->m_particles_buffer_wrapper.clear( false );
        }

        return;
    }

    SIXTRL_INLINE TrackJobBase::buffer_t&
    TrackJobBase::doGetBeamElementsBuffer() SIXTRL_RESTRICT
    {
        using _this_t = TrackJobBase;
        return const_cast< _this_t::buffer_t& >( static_cast< _this_t const& >(
            *this ).doGetBeamElementsBuffer() );
    }

    SIXTRL_INLINE TrackJobBase::buffer_t const&
    TrackJobBase::doGetBeamElementsBuffer() const SIXTRL_RESTRICT
    {
        SIXTRL_ASSERT( this->m_ptr_beam_elements_buffer != nullptr );
        SIXTRL_ASSERT( this->m_beam_elements_buffer_wrapper.getCApiPtr() ==
                       this->m_ptr_beam_elements_buffer );

        return this->m_beam_elements_buffer_wrapper;
    }

    SIXTRL_INLINE TrackJobBase::c_buffer_t*
    TrackJobBase::doGetPtrBeamElementsBuffer() SIXTRL_NOEXCEPT
    {
        return this->m_ptr_beam_elements_buffer;
    }

    SIXTRL_INLINE TrackJobBase::c_buffer_t const*
    TrackJobBase::doGetPtrBeamElementsBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_beam_elements_buffer;
    }

    SIXTRL_INLINE void TrackJobBase::doSetPtrToBeamElementsBuffer(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT buffer )
    {
        this->m_ptr_beam_elements_buffer = buffer;

        if( buffer != nullptr )
        {
            this->m_beam_elements_buffer_wrapper = *buffer;
            SIXTRL_ASSERT( this->m_beam_elements_buffer_wrapper.getCApiPtr() ==
                           this->m_ptr_beam_elements_buffer );
        }
        else
        {
            this->m_beam_elements_buffer_wrapper.clear( false );
        }

        return;
    }

    SIXTRL_INLINE TrackJobBase::buffer_t&
    TrackJobBase::doGetOutputBuffer() SIXTRL_RESTRICT
    {
        using _this_t = TrackJobBase;
        return const_cast< _this_t::buffer_t& >( static_cast< _this_t const& >(
            *this ).doGetOutputBuffer() );
    }

    SIXTRL_INLINE TrackJobBase::buffer_t const&
    TrackJobBase::doGetOutputBuffer() const SIXTRL_RESTRICT
    {
        SIXTRL_ASSERT( this->m_ptr_output_buffer != nullptr );
        SIXTRL_ASSERT( this->m_output_buffer_wrapper.getCApiPtr() ==
                       this->m_ptr_output_buffer );

        return this->m_output_buffer_wrapper;
    }

    SIXTRL_INLINE TrackJobBase::c_buffer_t*
    TrackJobBase::doGetPtrOutputBuffer() SIXTRL_NOEXCEPT
    {
        return this->m_ptr_output_buffer;
    }

    SIXTRL_INLINE TrackJobBase::c_buffer_t const*
    TrackJobBase::doGetPtrOutputBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_output_buffer;
    }

    SIXTRL_INLINE void TrackJobBase::doSetPtrToOutputBuffer(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT buffer )
    {
        this->m_ptr_output_buffer = buffer;

        if( buffer != nullptr )
        {
            this->m_output_buffer_wrapper = *buffer;
            SIXTRL_ASSERT( this->m_output_buffer_wrapper.getCApiPtr() ==
                           this->m_ptr_output_buffer );
        }
        else
        {
            this->m_output_buffer_wrapper.clear( false );
        }

        return;
    }

    SIXTRL_INLINE TrackJobBase::size_type
    TrackJobBase::doGetElemByElemOutBufferIndex() const SIXTRL_NOEXCEPT
    {
        return this->m_elem_by_elem_out_buffer_index;
    }

    SIXTRL_INLINE void TrackJobBase::doSetElemByElemOutBufferIndex(
        TrackJobBase::size_type const index ) SIXTRL_NOEXCEPT
    {
        this->m_elem_by_elem_out_buffer_index = index;
        return;
    }

    SIXTRL_INLINE TrackJobBase::size_type
    TrackJobBase::doGetBeamMonitorOutBufferIndexOffset() const SIXTRL_NOEXCEPT
    {
        return this->m_beam_monitor_out_buffer_index_offset;
    }

    SIXTRL_INLINE void TrackJobBase::doSetBeamMonitorOutBufferIndexOffset(
        TrackJobBase::size_type const index ) SIXTRL_NOEXCEPT
    {
        this->m_beam_monitor_out_buffer_index_offset = index;
        return;
    }

    SIXTRL_INLINE bool TrackJobBase::doPerformConfigBaseImpl(
            char const* SIXTRL_RESTRICT config_str )
    {
        bool success = false;

        if( config_str != nullptr )
        {
            this->m_config_str = std::string( config_str );
            success = true;
        }
        else
        {
            this->m_config_str.clear();
        }

        return success;
    }

    SIXTRL_INLINE bool TrackJobBase::doTrackUntilTurnBaseImpl(
            size_type until_turn, c_buffer_t* SIXTRL_RESTRICT output_buffer )
    {
        ( void )output_buffer;
        ( void )until_turn;
        return true;
    }

    SIXTRL_INLINE bool TrackJobBase::doTrackElemByElemBaseImpl(
        size_type num_elem_by_elem_turns,
        elem_by_elem_config_t const* SIXTRL_RESTRICT elem_by_elem_config,
        c_buffer_t* SIXTRL_RESTRICT output_buffer )
    {
        ( void )num_elem_by_elem_turns;
        ( void )elem_by_elem_config;
        ( void )output_buffer;

        return true;
    }

    SIXTRL_INLINE bool TrackJobBase::doCollectParticlesBufferBaseImpl(
            TrackJobBase::c_buffer_t* SIXTRL_RESTRICT buffer )
    {
        ( void )buffer;
        return true;
    }

    SIXTRL_INLINE bool TrackJobBase::doCollectBeamElementsBufferBaseImpl(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT buffer )
    {
        ( void )buffer;
        return true;
    }

    SIXTRL_INLINE bool TrackJobBase::doCollectOutputBufferBaseImpl(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT buffer )
    {
        ( void )buffer;
        return true;
    }
}

typedef SIXTRL_CXX_NAMESPACE::TrackJobBase::track_status_t  NS(track_status_t);

#else /* defined( __cplusplus ) */

typedef SIXTRL_INT32_T  NS(track_status_t);

#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_SIXTRACKLIB_COMMON_INTERNAL_TRACK_JOB_BASE_H__ */
/*end: sixtracklib/common/internal/track_job_base.h */
