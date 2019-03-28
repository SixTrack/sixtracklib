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
        #include <vector>
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
    #endif /* defined( __cplusplus ) */

    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/context/context_abs_base.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

typedef SIXTRL_INT64_T  NS(track_job_type_t);
typedef SIXTRL_INT32_T  NS(track_status_t);
typedef SIXTRL_UINT32_T NS(track_job_output_flag_t);

#if defined( __cplusplus ) && !defined( _GPUCODE )

SIXTRL_STATIC_VAR NS(track_job_output_flag_t) const
    NS(TRACK_JOB_OUTPUT_NONE) = ( NS(track_job_output_flag_t ) )0x00;

SIXTRL_STATIC_VAR NS(track_job_output_flag_t) const
    NS(TRACK_JOB_OUTPUT_ELEM_BY_ELEM) = ( NS(track_job_output_flag_t ) )0x01;

SIXTRL_STATIC_VAR NS(track_job_output_flag_t) const
    NS(TRACK_JOB_OUTPUT_BEAM_MONITORS) = ( NS(track_job_output_flag_t ) )0x02;

namespace SIXTRL_CXX_NAMESPACE
{
    using track_job_type_t        = ::NS(track_job_type_t);
    using track_status_t          = ::NS(track_status_t);
    using track_job_output_flag_t = ::NS(track_job_output_flag_t);

    static track_job_output_flag_t const TRACK_JOB_OUTPUT_NONE =
        track_job_output_flag_t{ 0x00 };

    static track_job_output_flag_t const TRACK_JOB_OUTPUT_ELEM_BY_ELEM =
        track_job_output_flag_t{ 0x01 };

    static track_job_output_flag_t const TRACK_JOB_OUTPUT_BEAM_MONITORS =
        track_job_output_flag_t{ 0x02 };

    class TrackJobBase
    {
        public:

        using buffer_t                = Buffer;
        using c_buffer_t              = ::NS(Buffer);
        using elem_by_elem_config_t   = ::NS(ElemByElemConfig);
        using elem_by_elem_order_t    = ::NS(elem_by_elem_order_t);
        using particle_index_t        = ::NS(particle_index_t);
        using size_type               = Buffer::size_type;
        using type_t                  = SIXTRL_CXX_NAMESPACE::track_job_type_t;
        using track_status_t          = SIXTRL_CXX_NAMESPACE::track_status_t;

        using track_job_output_flag_t =
            SIXTRL_CXX_NAMESPACE::track_job_output_flag_t;

        SIXTRL_HOST_FN void clear();
        SIXTRL_HOST_FN void collect();

        SIXTRL_HOST_FN track_status_t track(
            size_type const until_turn );

        SIXTRL_HOST_FN track_status_t trackElemByElem(
            size_type const dump_elem_by_elem_turns );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool reset(
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer   = nullptr,
            size_type const dump_elem_by_elem_turns = size_type{ 0 } );


        template< typename ParSetIndexIter  >
        SIXTRL_HOST_FN bool reset(
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            ParSetIndexIter  particle_set_indices_begin,
            ParSetIndexIter  particle_set_indices_end,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const dump_elem_by_elem_turns = size_type{ 0 } );

        SIXTRL_HOST_FN bool reset(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const dump_elem_by_elem_turns = size_type{ 0 } );

        SIXTRL_HOST_FN bool reset(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            size_type const num_particle_sets,
            size_type const* SIXTRL_RESTRICT particle_set_indices_begin,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const dump_elem_by_elem_turns = size_type{ 0 } );

        template< typename ParSetIndexIter  >
        SIXTRL_HOST_FN bool reset(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            ParSetIndexIter  particle_set_indices_begin,
            ParSetIndexIter  particle_set_indices_end,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const dump_elem_by_elem_turns = size_type{ 0 } );

        SIXTRL_HOST_FN bool assignOutputBuffer(
            buffer_t& SIXTRL_RESTRICT_REF output_buffer );

        SIXTRL_HOST_FN bool assignOutputBuffer(
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN track_job_type_t type()          const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN std::string const& typeStr()     const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN char const* ptrTypeStr()         const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasDeviceIdStr()            const SIXTRL_RESTRICT;
        SIXTRL_HOST_FN std::string const& deviceIdStr() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN char const* ptrDeviceIdStr()     const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasConfigStr()              const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN std::string const& configStr()   const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN char const* ptrConfigStr()       const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN size_type numParticleSets() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type const*
        particleSetIndicesBegin() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type const*
        particleSetIndicesEnd() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type particleSetIndex( size_type const n ) const;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN particle_index_t minParticleId() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN particle_index_t maxParticleId() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN particle_index_t minElementId()  const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN particle_index_t maxElementId()  const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN particle_index_t
        minInitialTurnId() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN particle_index_t
        maxInitialTurnId() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN buffer_t* ptrParticlesBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN buffer_t const*
        ptrParticlesBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t* ptrCParticlesBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t const*
        ptrCParticlesBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN buffer_t* ptrBeamElementsBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN buffer_t const*
        ptrBeamElementsBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t* ptrCBeamElementsBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t const*
        ptrCBeamElementsBuffer() const SIXTRL_NOEXCEPT;

        /* ---------------------------------------------------------------- */

        SIXTRL_HOST_FN bool hasOutputBuffer()      const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool ownsOutputBuffer()     const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool hasElemByElemOutput()  const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool hasBeamMonitorOutput() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type
        beamMonitorsOutputBufferOffset() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type
        elemByElemOutputBufferOffset() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type numElemByElemTurns() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN buffer_t* ptrOutputBuffer() SIXTRL_RESTRICT;
        SIXTRL_HOST_FN buffer_t* ptrOutputBuffer() const SIXTRL_RESTRICT;

        SIXTRL_HOST_FN c_buffer_t* ptrCOutputBuffer() SIXTRL_RESTRICT;

        SIXTRL_HOST_FN c_buffer_t const*
        ptrCOutputBuffer() const SIXTRL_RESTRICT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool hasBeamMonitors()      const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN size_type numBeamMonitors() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type const*
        beamMonitorIndicesBegin() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type const*
        beamMonitorIndicesEnd() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type beamMonitorIndex( size_type const n ) const;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool hasElemByElemConfig() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN elem_by_elem_config_t const*
        ptrElemByElemConfig() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN elem_by_elem_config_t*
        ptrElemByElemConfig() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool elemByElemRolling() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool defaultElemByElemRolling() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void setDefaultElemByElemRolling(
            bool is_rolling ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN elem_by_elem_order_t
        elemByElemOrder() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN elem_by_elem_order_t
        defaultElemByElemOrder() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void setDefaultElemByElemOrder(
            elem_by_elem_order_t const order ) SIXTRL_NOEXCEPT;

        /* ---------------------------------------------------------------- */

        protected:

        using ptr_output_buffer_t =
            std::unique_ptr< buffer_t >;

        using ptr_elem_by_elem_config_t =
            std::unique_ptr< elem_by_elem_config_t >;

        SIXTRL_HOST_FN TrackJobBase(
            const char *const SIXTRL_RESTRICT type_str,
            track_job_type_t const type_id );

        SIXTRL_HOST_FN TrackJobBase( TrackJobBase const& other );
        SIXTRL_HOST_FN TrackJobBase( TrackJobBase&& other ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN TrackJobBase& operator=(
            TrackJobBase const& rhs );

        SIXTRL_HOST_FN TrackJobBase& operator=(
            TrackJobBase&& rhs ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN virtual ~TrackJobBase() = default;

        SIXTRL_HOST_FN virtual void doClear();

        SIXTRL_HOST_FN virtual void doCollect();

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN virtual bool doPrepareParticlesStructures(
            c_buffer_t* SIXTRL_RESTRICT ptr_particles_buffer );

        SIXTRL_HOST_FN virtual bool doPrepareBeamElementsStructures(
            c_buffer_t* SIXTRL_RESTRICT ptr_beam_elem_buffer );

        SIXTRL_HOST_FN virtual bool doPrepareOutputStructures(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const dump_elem_by_elem_turns );

        SIXTRL_HOST_FN virtual bool doAssignOutputBufferToBeamMonitors(
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT output_buffer );

        SIXTRL_HOST_FN virtual bool doReset(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const dump_elem_by_elem_turns );

        SIXTRL_HOST_FN virtual bool doAssignNewOutputBuffer(
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN virtual track_status_t doTrackUntilTurn(
            size_type const until_turn );

        SIXTRL_HOST_FN virtual track_status_t doTrackElemByElem(
            size_type const dump_elem_by_elem_turns );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN virtual void doParseConfigStr(
            const char *const SIXTRL_RESTRICT config_str );

        SIXTRL_HOST_FN void doSetDeviceIdStr(
            const char *const SIXTRL_RESTRICT device_id_str );

        SIXTRL_HOST_FN void doSetConfigStr(
            const char *const SIXTRL_RESTRICT config_str );

        SIXTRL_HOST_FN void doSetPtrParticleBuffer(
            buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetPtrBeamElementsBuffer(
            buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetPtrOutputBuffer(
            buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetPtrCParticleBuffer(
            c_buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetPtrCBeamElementsBuffer(
            c_buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetPtrCOutputBuffer(
            c_buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetBeamMonitorOutputBufferOffset(
            size_type const output_buffer_offset ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetElemByElemOutputIndexOffset(
            size_type const target_num_output_turns ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetNumElemByElemOutputTurns(
            size_type const dump_elem_by_elem_turns ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetBeamMonitorOutputEnabledFlag(
            bool const beam_monitor_flag ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetElemByElemOutputEnabledFlag(
            bool const elem_by_elem_flag ) SIXTRL_NOEXCEPT;

        template< typename Iter >
        SIXTRL_HOST_FN void doSetParticleSetIndices( Iter begin, Iter end,
            const c_buffer_t *const SIXTRL_RESTRICT pbuffer = nullptr );

        SIXTRL_HOST_FN void doInitDefaultParticleSetIndices();

        template< typename BeMonitorIndexIter >
        SIXTRL_HOST_FN void doSetBeamMonitorIndices(
            BeMonitorIndexIter begin, BeMonitorIndexIter end );

        SIXTRL_HOST_FN void doInitDefaultBeamMonitorIndices();

        SIXTRL_HOST_FN void doSetMinParticleId(
            particle_index_t const min_particle_id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetMaxParticleId(
            particle_index_t const max_particle_id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetMinElementId(
            particle_index_t const min_element_id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetMaxElementId(
            particle_index_t const max_element_id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetMinInitialTurnId(
            particle_index_t const min_initial_turn_id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetMaxInitialTurnId(
            particle_index_t const max_initial_turn_id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doUpdateStoredOutputBuffer(
            ptr_output_buffer_t&& ptr_output_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doUpdateStoredElemByElemConfig(
            ptr_elem_by_elem_config_t&& ptr_config ) SIXTRL_NOEXCEPT;

        private:

        SIXTRL_HOST_FN void doClearBaseImpl() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doParseConfigStrBaseImpl(
            const char *const SIXTRL_RESTRICT config_str );

        std::string                     m_type_str;
        std::string                     m_device_id_str;
        std::string                     m_config_str;

        std::vector< size_type >        m_particle_set_indices;
        std::vector< size_type >        m_beam_monitor_indices;

        ptr_output_buffer_t             m_my_output_buffer;
        ptr_elem_by_elem_config_t       m_my_elem_by_elem_config;

        buffer_t*   SIXTRL_RESTRICT     m_ptr_particles_buffer;
        buffer_t*   SIXTRL_RESTRICT     m_ptr_beam_elem_buffer;
        buffer_t*   SIXTRL_RESTRICT     m_ptr_output_buffer;

        c_buffer_t* SIXTRL_RESTRICT     m_ptr_c_particles_buffer;
        c_buffer_t* SIXTRL_RESTRICT     m_ptr_c_beam_elem_buffer;
        c_buffer_t* SIXTRL_RESTRICT     m_ptr_c_output_buffer;

        size_type                       m_be_mon_output_buffer_offset;
        size_type                       m_elem_by_elem_output_offset;
        size_type                       m_dump_elem_by_elem_turns;

        type_t                          m_type_id;
        elem_by_elem_order_t            m_default_elem_by_elem_order;

        particle_index_t                m_min_particle_id;
        particle_index_t                m_max_particle_id;

        particle_index_t                m_min_element_id;
        particle_index_t                m_max_element_id;

        particle_index_t                m_min_initial_turn_id;
        particle_index_t                m_max_initial_turn_id;

        bool                            m_default_elem_by_elem_rolling;
        bool                            m_has_beam_monitor_output;
        bool                            m_has_elem_by_elem_output;
    };

    SIXTRL_EXTERN SIXTRL_HOST_FN track_job_output_flag_t
    TrackJob_needs_output_buffer(
        Buffer const& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer const& SIXTRL_RESTRICT_REF beam_elements_buffer,
        Buffer::size_type const dump_elem_by_elem_turns ) SIXTRL_NOEXCEPT;
}

typedef SIXTRL_CXX_NAMESPACE::TrackJobBase NS(TrackJobBase);

#else /* defined( __cplusplus ) && !defined( _GPUCODE ) */

typedef void NS(TrackJobBase);

#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#if !defined( _GPUCODE )

/* ------------------------------------------------------------------------- */
/* ----  Declaration of C99 API bindings / functions                    ---- */
/* ------------------------------------------------------------------------- */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_job_output_flag_t)
NS(TrackJob_needs_output_buffer)(
    const NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
    const NS(Buffer) *const SIXTRL_RESTRICT beam_elem_buffer,
    NS(buffer_size_t) const dump_elem_by_elem_turns );

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#if defined( __cplusplus )

/* ------------------------------------------------------------------------- */
/* ---- implementation of inline and template C++ methods/functions     ---- */
/* ------------------------------------------------------------------------- */

namespace SIXTRL_CXX_NAMESPACE
{
    template< typename ParSetIndexIter  >
    SIXTRL_HOST_FN bool TrackJobBase::reset(
        TrackJobBase::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        ParSetIndexIter  particle_set_indices_begin,
        ParSetIndexIter  particle_set_indices_end,
        TrackJobBase::buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        size_type const dump_elem_by_elem_turns )
    {
        using c_buffer_t = TrackJobBase::c_buffer_t;

        bool success = false;

        this->doClear();

        c_buffer_t* ptr_pb  = particles_buffer.getCApiPtr();
        c_buffer_t* ptr_eb  = beam_elements_buffer.getCApiPtr();
        c_buffer_t* ptr_out = ( ptr_output_buffer != nullptr )
            ? ptr_output_buffer->getCApiPtr() : nullptr;

        if( ( particle_set_indices_begin !=
              particle_set_indices_end ) &&
            ( std::distance( particle_set_indices_begin,
                particle_set_indices_end ) > std::ptrdiff_t{ 0 } ) )
        {
            this->doSetParticleSetIndices(
                particle_set_indices_begin, particle_set_indices_end, ptr_pb );
        }
        else
        {
            this->doInitDefaultBeamMonitorIndices();
        }

        success = this->doReset(
            ptr_pb, ptr_eb, ptr_out, dump_elem_by_elem_turns );

        if( success )
        {
            this->doSetPtrParticleBuffer( &particles_buffer );
            this->doSetPtrBeamElementsBuffer( &beam_elements_buffer );

            if( ( ptr_out != nullptr ) && ( this->hasOutputBuffer() ) )
            {
                this->doSetPtrOutputBuffer( ptr_output_buffer );
            }
        }

        return success;
    }

    template< typename ParSetIndexIter  >
    SIXTRL_HOST_FN bool TrackJobBase::reset(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        ParSetIndexIter  particle_set_indices_begin,
        ParSetIndexIter  particle_set_indices_end,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobBase::size_type const dump_elem_by_elem_turns )
    {
        this->doClear();

        if( ( particle_set_indices_begin !=
              particle_set_indices_end ) &&
            ( std::distance( particle_set_indices_begin,
                particle_set_indices_end ) > std::ptrdiff_t{ 0 } ) )
        {
            this->doSetParticleSetIndices( particle_set_indices_begin,
                particle_set_indices_end, particles_buffer );
        }
        else
        {
            this->doInitDefaultBeamMonitorIndices();
        }

        bool success = this->doReset( particles_buffer, beam_elements_buffer,
            ptr_output_buffer, dump_elem_by_elem_turns );

        /*
        if( success )
        {
            this->doSetPtrCParticleBuffer( particles_buffer );
            this->doSetPtrCBeamElementsBuffer( beam_elements_buffer );
        }
        */

        return success;
    }

    template< typename ParSetIndexIter >
    SIXTRL_HOST_FN void TrackJobBase::doSetParticleSetIndices(
        ParSetIndexIter begin, ParSetIndexIter end,
        const TrackJobBase::c_buffer_t *const SIXTRL_RESTRICT pbuffer )
    {
        using diff_t   = std::ptrdiff_t;
        using size_t   = TrackJobBase::size_type;
        using obj_it_t = SIXTRL_BUFFER_DATAPTR_DEC ::NS(Object) const*;

        diff_t const temp_len = std::distance( begin, end );

        if( temp_len >= diff_t{ 0 } )
        {
            this->m_particle_set_indices.clear();

            if( temp_len > diff_t{ 0 } )
            {
                this->m_particle_set_indices.clear();
                this->m_particle_set_indices.reserve(
                    static_cast< size_t >( temp_len ) );

                if( pbuffer == nullptr )
                {
                    this->m_particle_set_indices.assign( begin, end );
                }
                else
                {
                    size_t const SETS_IN_BUFFER =
                        ::NS(Buffer_get_num_of_objects)( pbuffer );

                    obj_it_t obj = nullptr;

                    for( ParSetIndexIter it = begin ; it != end ; ++it )
                    {
                        obj = ::NS(Buffer_get_const_object)( pbuffer, *it );

                        if( ( *it < SETS_IN_BUFFER ) && ( obj != nullptr ) &&
                            ( ::NS(Object_get_type_id)( obj ) ==
                                ::NS(OBJECT_TYPE_PARTICLE) ) )
                        {
                            this->m_particle_set_indices.push_back( *it );
                        }
                    }
                }

                std::sort( this->m_particle_set_indices.begin(),
                           this->m_particle_set_indices.end() );

                this->m_particle_set_indices.erase( std::unique(
                    this->m_particle_set_indices.begin(),
                    this->m_particle_set_indices.end() ),
                    this->m_particle_set_indices.end() );
            }
        }

        return;
    }

    template< typename BeMonitorIndexIter >
    SIXTRL_HOST_FN void TrackJobBase::doSetBeamMonitorIndices(
        BeMonitorIndexIter begin, BeMonitorIndexIter end )
    {
        using diff_t = std::ptrdiff_t;
        using size_t = TrackJobBase::size_type;

        diff_t const temp_len = std::distance( begin, end );

        if( temp_len >= diff_t{ 0 } )
        {
            this->m_beam_monitor_indices.clear();

            if( temp_len > diff_t{ 0 } )
            {
                this->m_beam_monitor_indices.reserve(
                    static_cast< size_t >( temp_len ) );

                this->m_beam_monitor_indices.assign( begin, end );

                std::sort( this->m_beam_monitor_indices.begin(),
                           this->m_beam_monitor_indices.end() );

                this->m_beam_monitor_indices.erase( std::unique(
                    this->m_beam_monitor_indices.begin(),
                    this->m_beam_monitor_indices.end() ),
                    this->m_beam_monitor_indices.end() );
            }
        }

        return;
    }
}

#endif /* defined( __cplusplus ) */


#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_SIXTRACKLIB_COMMON_INTERNAL_TRACK_JOB_BASE_H__ */
/*end: sixtracklib/common/internal/track_job_base.h */
