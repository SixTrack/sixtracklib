#ifndef SIXTRACKLIB_COMMON_TRACK_TRACK_JOB_BASE_HPP__
#define SIXTRACKLIB_COMMON_TRACK_TRACK_JOB_BASE_HPP__

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
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/arch_base.hpp"
    #include "sixtracklib/common/track/definitions.h"

    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
        #include "sixtracklib/common/particles/particles_addr.hpp"
    #endif /* defined( __cplusplus ) */

    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/particles/particles_addr.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/output/output_buffer.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )

namespace SIXTRL_CXX_NAMESPACE
{
    class TrackJobBaseNew : public SIXTRL_CXX_NAMESPACE::ArchBase
    {
        private:

        using _arch_base_t          = SIXTRL_CXX_NAMESPACE::ArchBase;

        public:

        using arch_id_t             = _arch_base_t::arch_id_t;
        using buffer_t              = SIXTRL_CXX_NAMESPACE::Buffer;
        using c_buffer_t            = buffer_t::c_api_t;
        using size_type             = buffer_t::size_type;
        using status_t              = ::NS(controller_status_t);
        using track_status_t        = ::NS(track_status_t);

        using elem_by_elem_config_t = ::NS(ElemByElemConfig);
        using elem_by_elem_order_t  = ::NS(elem_by_elem_order_t);
        using particle_index_t      = ::NS(particle_index_t);
        using collect_flag_t        = ::NS(track_job_collect_flag_t);
        using output_buffer_flag_t  = ::NS(output_buffer_flag_t);
        using success_flag_t        = ::NS(controller_success_flag_t);
        using particles_addr_t      = ::NS(ParticlesAddr);
        using num_particles_t       = ::NS(particle_num_elements_t);

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool usesController() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool usesArguments() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN static bool IsCollectFlagSet(
            collect_flag_t const haystack,
            collect_flag_t const needle ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN static size_type
        DefaultNumParticleSetIndices() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN static size_type const*
        DefaultParticleSetIndicesBegin() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN static size_type const*
        DefaultParticleSetIndicesEnd() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void clear();

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN collect_flag_t collect();
        SIXTRL_HOST_FN collect_flag_t collect( collect_flag_t const flags );

        SIXTRL_HOST_FN bool collectParticles();
        SIXTRL_HOST_FN bool collectBeamElements();
        SIXTRL_HOST_FN bool collectOutput();
        SIXTRL_HOST_FN bool collectSuccessFlag();

        SIXTRL_HOST_FN void enableCollectParticles()  SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN void disableCollectParticles() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool isCollectingParticles() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void enableCollectBeamElements()  SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN void disableCollectBeamElements() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool isCollectingBeamElements() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void enableCollectOutput()  SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN void disableCollectOutput() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool isCollectingOutput() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN collect_flag_t collectFlags() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN void setCollectFlags(
            collect_flag_t const flag ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool requiresCollecting() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN status_t fetchParticleAddresses();

        SIXTRL_HOST_FN bool hasParticleAddresses() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN particles_addr_t const* particleAddresses(
            size_type const index = size_type{ 0 } ) const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN track_status_t trackUntil(
            size_type const until_turn );

        SIXTRL_HOST_FN track_status_t trackElemByElem(
            size_type const until_turn_elem_by_elem );

        SIXTRL_HOST_FN track_status_t trackLine(
            size_type const beam_elements_begin_index,
            size_type const beam_elements_end_index,
            bool const finish_turn = false );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN virtual ~TrackJobBaseNew() = default;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool reset(
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer   = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 } );

        SIXTRL_HOST_FN bool reset(
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            size_type const particle_set_index,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 } );

        template< typename ParSetIndexIter  >
        SIXTRL_HOST_FN bool reset(
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            ParSetIndexIter  particle_set_indices_begin,
            ParSetIndexIter  particle_set_indices_end,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 } );

        SIXTRL_HOST_FN bool reset(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 } );

        SIXTRL_HOST_FN bool reset(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            size_type const particle_set_index,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 } );

        SIXTRL_HOST_FN bool reset(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            size_type const num_particle_sets,
            size_type const* SIXTRL_RESTRICT particle_set_indices_begin,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 } );

        template< typename ParSetIndexIter  >
        SIXTRL_HOST_FN bool reset(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            ParSetIndexIter  particle_set_indices_begin,
            ParSetIndexIter  particle_set_indices_end,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 } );

        SIXTRL_HOST_FN bool selectParticleSet(
            size_type const particle_set_index );

        SIXTRL_HOST_FN bool assignOutputBuffer(
            buffer_t& SIXTRL_RESTRICT_REF output_buffer );

        SIXTRL_HOST_FN bool assignOutputBuffer(
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN size_type numParticleSets()   const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type const*
        particleSetIndicesBegin() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type const*
        particleSetIndicesEnd() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type particleSetIndex( size_type const n ) const;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN num_particles_t
        totalNumParticles() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN particle_index_t minParticleId() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN particle_index_t maxParticleId() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN particle_index_t minElementId()  const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN particle_index_t maxElementId()  const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN particle_index_t
        minInitialTurnId() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN particle_index_t
        maxInitialTurnId() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN success_flag_t lastSuccessFlag() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN buffer_t* ptrParticlesBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN buffer_t const*
        ptrParticlesBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t* ptrCParticlesBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t const*
        ptrCParticlesBuffer() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

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

        SIXTRL_HOST_FN particle_index_t
        untilTurnElemByElem() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type numElemByElemTurns() const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN buffer_t* ptrOutputBuffer() SIXTRL_RESTRICT;
        SIXTRL_HOST_FN buffer_t const* ptrOutputBuffer() const SIXTRL_RESTRICT;

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
            bool const is_rolling ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN elem_by_elem_order_t
        elemByElemOrder() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN elem_by_elem_order_t
        defaultElemByElemOrder() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void setDefaultElemByElemOrder(
            elem_by_elem_order_t const order ) SIXTRL_NOEXCEPT;

        /* ================================================================ */

        template< class Derived >
        SIXTRL_HOST_FN Derived const* asDerivedTrackJob(
            arch_id_t const required_arch_id,
            bool const requires_exact_match = false ) const SIXTRL_NOEXCEPT;

        template< class Derived >
        SIXTRL_HOST_FN Derived* asDerivedTrackJob(
            arch_id_t const required_arch_id,
            bool const requires_exact_match = false ) SIXTRL_NOEXCEPT;

        protected:

        using el_by_el_conf_t = elem_by_elem_config_t;
        using ptr_output_buffer_t = std::unique_ptr< buffer_t >;
        using ptr_particles_addr_buffer_t = std::unique_ptr< buffer_t >;
        using ptr_elem_by_elem_config_t = std::unique_ptr< el_by_el_conf_t >;

        SIXTRL_HOST_FN static collect_flag_t UnsetCollectFlag(
            collect_flag_t const haystack,
            collect_flag_t const needle ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN TrackJobBaseNew( arch_id_t const arch_id,
            char const* SIXTRL_RESTRICT arch_str,
            char const* SIXTRL_RESTRICT config_str );

        SIXTRL_HOST_FN TrackJobBaseNew( TrackJobBaseNew const& other );
        SIXTRL_HOST_FN TrackJobBaseNew( TrackJobBaseNew&& other ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN TrackJobBaseNew& operator=(
            TrackJobBaseNew const& rhs );

        SIXTRL_HOST_FN TrackJobBaseNew& operator=(
            TrackJobBaseNew&& rhs ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN virtual void doClear();
        SIXTRL_HOST_FN virtual collect_flag_t doCollect(
            collect_flag_t const flags );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN virtual bool doPrepareParticlesStructures(
            c_buffer_t* SIXTRL_RESTRICT ptr_particles_buffer );

        SIXTRL_HOST_FN virtual bool doPrepareBeamElementsStructures(
            c_buffer_t* SIXTRL_RESTRICT ptr_beam_elem_buffer );

        SIXTRL_HOST_FN virtual bool doPrepareOutputStructures(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const until_turn_elem_by_elem );

        SIXTRL_HOST_FN virtual bool doAssignOutputBufferToBeamMonitors(
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT output_buffer );

        SIXTRL_HOST_FN virtual bool doReset(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const until_turn_elem_by_elem );

        SIXTRL_HOST_FN virtual bool doAssignNewOutputBuffer(
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN virtual status_t doFetchParticleAddresses();

        SIXTRL_HOST_FN virtual track_status_t doTrackUntilTurn(
            size_type const until_turn );

        SIXTRL_HOST_FN virtual track_status_t doTrackElemByElem(
            size_type const until_turn_elem_by_elem );

        SIXTRL_HOST_FN virtual track_status_t doTrackLine(
            size_type const beam_elements_begin_index,
            size_type const beam_elements_end_index, bool const finish_turn );

        /* ----------------------------------------------------------------- */

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

        SIXTRL_HOST_FN void doSetUntilTurnElemByElem(
            particle_index_t const until_turn_elem_by_elem ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetRequiresCollectFlag(
            bool const requires_collect_flag ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetBeamMonitorOutputEnabledFlag(
            bool const beam_monitor_flag ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetElemByElemOutputEnabledFlag(
            bool const elem_by_elem_flag ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        template< typename Iter >
        SIXTRL_HOST_FN void doSetParticleSetIndices( Iter begin, Iter end,
            const c_buffer_t *const SIXTRL_RESTRICT pbuffer = nullptr );

        SIXTRL_HOST_FN void doInitDefaultParticleSetIndices();

        /* ----------------------------------------------------------------- */

        template< typename BeMonitorIndexIter >
        SIXTRL_HOST_FN void doSetBeamMonitorIndices(
            BeMonitorIndexIter begin, BeMonitorIndexIter end );

        SIXTRL_HOST_FN void doInitDefaultBeamMonitorIndices();

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void doSetTotalNumParticles(
            num_particles_t const total_num_particles ) SIXTRL_NOEXCEPT;

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

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN buffer_t const*
        doGetPtrParticlesAddrBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN buffer_t* doGetPtrParticlesAddrBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doUpdateStoredParticlesAddrBuffer(
            ptr_particles_addr_buffer_t&& ptr_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetHasParticleAddressesFlag(
            bool const has_particle_addresses ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void doUpdateStoredOutputBuffer(
            ptr_output_buffer_t&& ptr_output_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doUpdateStoredElemByElemConfig(
            ptr_elem_by_elem_config_t&& ptr_config ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void doSetUsesControllerFlag(
            bool const uses_controller_flag ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetUsesArgumentsFlag(
            bool const arguments_flag ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void doSetLastSuccessFlag(
            success_flag_t const last_success_flag_value ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN success_flag_t const*
        doGetPtrLastSuccessFlag() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN success_flag_t*
        doGetPtrLastSuccessFlag() SIXTRL_NOEXCEPT;

        private:

        SIXTRL_HOST_FN void doClearBaseImpl() SIXTRL_NOEXCEPT;

        std::vector< size_type >        m_particle_set_indices;
        std::vector< size_type >        m_beam_monitor_indices;

        ptr_output_buffer_t             m_my_output_buffer;
        ptr_particles_addr_buffer_t     m_my_particles_addr_buffer;
        ptr_elem_by_elem_config_t       m_my_elem_by_elem_config;

        buffer_t*   SIXTRL_RESTRICT     m_ptr_particles_buffer;
        buffer_t*   SIXTRL_RESTRICT     m_ptr_beam_elem_buffer;
        buffer_t*   SIXTRL_RESTRICT     m_ptr_output_buffer;

        c_buffer_t* SIXTRL_RESTRICT     m_ptr_c_particles_buffer;
        c_buffer_t* SIXTRL_RESTRICT     m_ptr_c_beam_elem_buffer;
        c_buffer_t* SIXTRL_RESTRICT     m_ptr_c_output_buffer;

        size_type                       m_be_mon_output_buffer_offset;
        size_type                       m_elem_by_elem_output_offset;
        num_particles_t                 m_total_num_particles;

        elem_by_elem_order_t            m_default_elem_by_elem_order;

        particle_index_t                m_min_particle_id;
        particle_index_t                m_max_particle_id;

        particle_index_t                m_min_element_id;
        particle_index_t                m_max_element_id;

        particle_index_t                m_min_initial_turn_id;
        particle_index_t                m_max_initial_turn_id;
        particle_index_t                m_until_turn_elem_by_elem;

        collect_flag_t                  m_collect_flags;
        success_flag_t                  m_success_flag;

        bool                            m_default_elem_by_elem_rolling;
        bool                            m_has_beam_monitor_output;
        bool                            m_has_elem_by_elem_output;
        bool                            m_has_particle_addresses;

        bool                            m_requires_collect;
        bool                            m_uses_controller;
        bool                            m_uses_arguments;
    };

    SIXTRL_HOST_FN TrackJobBaseNew* TrackJobNew_create(
        char const* SIXTRL_RESTRICT arch_str,
        char const* SIXTRL_RESTRICT config_str );

    SIXTRL_HOST_FN TrackJobBaseNew* TrackJobNew_create(
        TrackJobBaseNew::arch_id_t const arch_id,
        char const* SIXTRL_RESTRICT config_str );

    SIXTRL_HOST_FN TrackJobBaseNew* TrackJobNew_new(
        char const* SIXTRL_RESTRICT arch_str,
        ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT beam_elemements_buffer,
        char const* SIXTRL_RESTRICT config_str );

    SIXTRL_HOST_FN TrackJobBaseNew* TrackJobNew_new(
        ::NS(arch_id_t) const arch_id,
        ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT beam_elemements_buffer,
        char const* SIXTRL_RESTRICT config_str );

    SIXTRL_HOST_FN TrackJobBaseNew* TrackJobNew_new(
        char const* SIXTRL_RESTRICT arch_str,
        ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT beam_elemements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        ::NS(buffer_size_t) const until_turn_elem_by_elem,
        char const* SIXTRL_RESTRICT config_str );

    SIXTRL_HOST_FN TrackJobBaseNew* TrackJobNew_new(
        ::NS(arch_id_t) const arch_id,
        ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT beam_elemements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        ::NS(buffer_size_t) const until_turn_elem_by_elem,
        char const* SIXTRL_RESTRICT config_str );

    SIXTRL_HOST_FN TrackJobBaseNew* TrackJobNew_new(
        char const* SIXTRL_RESTRICT arch_str,
        ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
        ::NS(buffer_size_t) const num_particle_sets,
        ::NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
        ::NS(Buffer)* SIXTRL_RESTRICT beam_elemements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        ::NS(buffer_size_t) const until_turn_elem_by_elem,
        char const* SIXTRL_RESTRICT config_str );

    SIXTRL_HOST_FN TrackJobBaseNew* TrackJobNew_new(
        ::NS(arch_id_t) const SIXTRL_RESTRICT arch_str,
        ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
        ::NS(buffer_size_t) const num_particle_sets,
        ::NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
        ::NS(Buffer)* SIXTRL_RESTRICT beam_elemements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        ::NS(buffer_size_t) const until_turn_elem_by_elem,
        const char *const SIXTRL_RESTRICT config_str );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_HOST_FN TrackJobBaseNew* TrackJobNew_create(
        SIXTRL_CXX_NAMESPACE::arch_id_t const arch_id,
        std::string const& SIXTRL_RESTRICT_REF config_str );

    SIXTRL_HOST_FN TrackJobBaseNew* TrackJobNew_create(
        std::string const& SIXTRL_RESTRICT_REF arch_str,
        std::string const& SIXTRL_RESTRICT_REF config_str );

    SIXTRL_HOST_FN TrackJobBaseNew* TrackJobNew_new(
        SIXTRL_CXX_NAMESPACE::arch_id_t const arch_id,
        Buffer& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer& SIXTRL_RESTRICT_REF beam_elemements_buffer,
        std::string const& SIXTRL_RESTRICT_REF config_str );

    SIXTRL_HOST_FN TrackJobBaseNew* TrackJobNew_new(
        std::string const& SIXTRL_RESTRICT_REF arch_str,
        Buffer& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer& SIXTRL_RESTRICT_REF beam_elemements_buffer,
        std::string const& SIXTRL_RESTRICT_REF config_str );

    SIXTRL_HOST_FN TrackJobBaseNew* TrackJobNew_new(
        SIXTRL_CXX_NAMESPACE::arch_id_t const arch_id,
        Buffer& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer& SIXTRL_RESTRICT_REF beam_elemements_buffer,
        Buffer* SIXTRL_RESTRICT output_buffer,
        Buffer::size_type const until_turn_elem_by_elem,
        std::string const& SIXTRL_RESTRICT_REF config_str );

    SIXTRL_HOST_FN TrackJobBaseNew* TrackJobNew_new(
        SIXTRL_CXX_NAMESPACE::arch_id_t const arch_id,
        Buffer& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer& SIXTRL_RESTRICT_REF beam_elemements_buffer,
        Buffer* SIXTRL_RESTRICT output_buffer,
        Buffer::size_type const until_turn_elem_by_elem,
        std::string const& SIXTRL_RESTRICT_REF config_str );

    SIXTRL_HOST_FN TrackJobBaseNew* TrackJobNew_new(
        std::string const& SIXTRL_RESTRICT_REF arch_str,
        Buffer& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer::size_type const num_particle_sets,
        Buffer::size_type const* SIXTRL_RESTRICT particle_set_indices_begin,
        Buffer& SIXTRL_RESTRICT_REF beam_elemements_buffer,
        Buffer* SIXTRL_RESTRICT output_buffer,
        Buffer::size_type const until_turn_elem_by_elem,
        std::string const& SIXTRL_RESTRICT_REF config_str );

    SIXTRL_HOST_FN TrackJobBaseNew* TrackJobNew_new(
        SIXTRL_CXX_NAMESPACE::arch_id_t const arch_id,
        Buffer& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer::size_type const num_particle_sets,
        Buffer::size_type const* SIXTRL_RESTRICT particle_set_indices_begin,
        Buffer& SIXTRL_RESTRICT_REF beam_elemements_buffer,
        Buffer* SIXTRL_RESTRICT output_buffer,
        Buffer::size_type const until_turn_elem_by_elem,
        std::string const& SIXTRL_RESTRICT_REF config_str );

    template< typename Iter >
    SIXTRL_HOST_FN TrackJobBaseNew* TrackJobNew_new(
        SIXTRL_CXX_NAMESPACE::arch_id_t const arch_id,
        Buffer& SIXTRL_RESTRICT_REF particles_buffer,
        Iter particle_set_indices_begin, Iter particle_set_indices_end,
        Buffer& SIXTRL_RESTRICT_REF beam_elemements_buffer,
        Buffer* SIXTRL_RESTRICT output_buffer,
        Buffer::size_type const until_turn_elem_by_elem,
        std::string const& SIXTRL_RESTRICT_REF config_str );

    template< typename Iter >
    SIXTRL_HOST_FN TrackJobBaseNew* TrackJobNew_new(
        std::string const& SIXTRL_RESTRICT_REF arch_str,
        Buffer& SIXTRL_RESTRICT_REF particles_buffer,
        Iter particle_set_indices_begin, Iter particle_set_indices_end,
        Buffer& SIXTRL_RESTRICT_REF beam_elemements_buffer,
        Buffer* SIXTRL_RESTRICT output_buffer,
        Buffer::size_type const until_turn_elem_by_elem,
        std::string const& SIXTRL_RESTRICT_REF config_str );

    template< typename Iter >
    SIXTRL_HOST_FN TrackJobBaseNew* TrackJobNew_new(
        SIXTRL_CXX_NAMESPACE::arch_id_t const arch_id,
        ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
        Iter particle_set_indices_begin, Iter particle_set_indices_end,
        ::NS(Buffer)* SIXTRL_RESTRICT beam_elemements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        ::NS(buffer_size_t) const until_turn_elem_by_elem,
        char const* SIXTRL_RESTRICT config_str );

    template< typename Iter >
    SIXTRL_HOST_FN TrackJobBaseNew* TrackJobNew_new(
        std::string const& SIXTRL_RESTRICT_REF arch_str,
        ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
        Iter particle_set_indices_begin, Iter particle_set_indices_end,
        ::NS(Buffer)* SIXTRL_RESTRICT beam_elemements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        ::NS(buffer_size_t) const until_turn_elem_by_elem,
        char const* SIXTRL_RESTRICT config_str );
}

typedef SIXTRL_CXX_NAMESPACE::TrackJobBaseNew NS(TrackJobBaseNew);

#else /* C++, Host */

typedef void NS(TrackJobBaseNew);

#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

/* ------------------------------------------------------------------------- */
/* ---- implementation of inline and template C++ methods/functions     ---- */
/* ------------------------------------------------------------------------- */

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_INLINE bool TrackJobBaseNew::usesController() const SIXTRL_NOEXCEPT
    {
        return this->m_uses_controller;
    }

    SIXTRL_INLINE bool TrackJobBaseNew::usesArguments() const SIXTRL_NOEXCEPT
    {
        return this->m_uses_arguments;
    }

    template< typename ParSetIndexIter  >
    SIXTRL_HOST_FN bool TrackJobBaseNew::reset(
        TrackJobBaseNew::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        ParSetIndexIter  particle_set_indices_begin,
        ParSetIndexIter  particle_set_indices_end,
        TrackJobBaseNew::buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        size_type const until_turn_elem_by_elem )
    {
        using c_buffer_t = TrackJobBaseNew::c_buffer_t;

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
            ptr_pb, ptr_eb, ptr_out, until_turn_elem_by_elem );

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
    SIXTRL_HOST_FN bool TrackJobBaseNew::reset(
        TrackJobBaseNew::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        ParSetIndexIter  particle_set_indices_begin,
        ParSetIndexIter  particle_set_indices_end,
        TrackJobBaseNew::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        TrackJobBaseNew::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobBaseNew::size_type const until_turn_elem_by_elem )
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
            ptr_output_buffer, until_turn_elem_by_elem );

        return success;
    }

    /* --------------------------------------------------------------------- */

    template< class Derived > Derived const* TrackJobBaseNew::asDerivedTrackJob(
        TrackJobBaseNew::arch_id_t const required_arch_id,
        bool requires_exact_match ) const SIXTRL_NOEXCEPT
    {
        Derived const* ptr_derived = nullptr;

        static_assert( std::is_base_of< TrackJobBaseNew, Derived >::value,
                       "asDerivedTrackJob< Derived > requires Dervied to be "
                       "derived from SIXTRL_CXX_NAMESPACE::TrackJobBaseNew" );

        if( ( ( !requires_exact_match ) &&
              ( this->isArchCompatibleWith( required_arch_id ) ) ) ||
            ( this->isArchIdenticalTo( required_arch_id ) ) )
        {
            ptr_derived = static_cast< Derived const* >( this );
        }

        return ptr_derived;
    }

    template< class Derived > Derived* TrackJobBaseNew::asDerivedTrackJob(
        TrackJobBaseNew::arch_id_t const required_arch_id,
        bool requires_exact_match ) SIXTRL_NOEXCEPT
    {
        return const_cast< Derived* >( static_cast< TrackJobBaseNew const& >(
            *this ).asDerivedTrackJob< Derived >(
                required_arch_id, requires_exact_match ) );
    }

    /* --------------------------------------------------------------------- */

    template< typename ParSetIndexIter >
    SIXTRL_HOST_FN void TrackJobBaseNew::doSetParticleSetIndices(
        ParSetIndexIter begin, ParSetIndexIter end,
        const TrackJobBaseNew::c_buffer_t *const SIXTRL_RESTRICT pbuffer )
    {
        using diff_t   = std::ptrdiff_t;
        using size_t   = TrackJobBaseNew::size_type;
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

    /* --------------------------------------------------------------------- */

    template< typename BeMonitorIndexIter >
    SIXTRL_HOST_FN void TrackJobBaseNew::doSetBeamMonitorIndices(
        BeMonitorIndexIter begin, BeMonitorIndexIter end )
    {
        using diff_t = std::ptrdiff_t;
        using size_t = TrackJobBaseNew::size_type;

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

    /* --------------------------------------------------------------------- */

    SIXTRL_INLINE bool TrackJobBaseNew::IsCollectFlagSet(
        TrackJobBaseNew::collect_flag_t const flag_set,
        TrackJobBaseNew::collect_flag_t const flag ) SIXTRL_NOEXCEPT
    {
        return ( ( flag_set & flag ) == flag );
    }

    SIXTRL_INLINE TrackJobBaseNew::collect_flag_t
    TrackJobBaseNew::UnsetCollectFlag(
        TrackJobBaseNew::collect_flag_t const flag_set,
        TrackJobBaseNew::collect_flag_t const flag ) SIXTRL_NOEXCEPT
    {
        return flag_set & ~flag;
    }

    /* ********************************************************************* */

    template< typename Iter > TrackJobBaseNew* TrackJobNew_new(
        SIXTRL_CXX_NAMESPACE::arch_id_t const arch_id,
        Buffer& SIXTRL_RESTRICT_REF pbuffer,
        Iter pset_indices_begin, Iter pset_indices_end,
        Buffer& SIXTRL_RESTRICT_REF belements_buffer,
        Buffer* SIXTRL_RESTRICT output_buffer,
        Buffer::size_type const until_turn_elem_by_elem,
        std::string const& SIXTRL_RESTRICT_REF config_str )
    {
        TrackJobBaseNew* ptr_track_job =
            SIXTRL_CXX_NAMESPACE::TrackJobNew_create( arch_id, config_str );

        if( ptr_track_job != nullptr )
        {
            if( !ptr_track_job->reset( pbuffer, pset_indices_begin,
                    pset_indices_end, belements_buffer, output_buffer,
                        until_turn_elem_by_elem ) )
            {
                delete ptr_track_job;
                ptr_track_job = nullptr;
            }
        }

        return ptr_track_job;
    }

     template< typename Iter > TrackJobBaseNew* TrackJobNew_new(
        std::string const& SIXTRL_RESTRICT_REF arch_str,
        Buffer& SIXTRL_RESTRICT_REF pbuffer,
        Iter pset_indices_begin, Iter pset_indices_end,
        Buffer& SIXTRL_RESTRICT_REF belements_buffer,
        Buffer* SIXTRL_RESTRICT output_buffer,
        Buffer::size_type const until_turn_elem_by_elem,
        std::string const& SIXTRL_RESTRICT_REF config_str )
    {
        return SIXTRL_CXX_NAMESPACE::TrackJobNew_new(
            SIXTRL_CXX_NAMESPACE::ArchInfo_arch_string_to_arch_id( arch_str ),
                pbuffer, pset_indices_begin, pset_indices_end, belements_buffer,
                    output_buffer, until_turn_elem_by_elem, config_str );
    }

    template< typename Iter > TrackJobBaseNew* TrackJobNew_new(
        SIXTRL_CXX_NAMESPACE::arch_id_t const arch_id,
        ::NS(Buffer)* SIXTRL_RESTRICT pbuffer,
        Iter pset_indices_begin, Iter pset_indices_end,
        ::NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        ::NS(buffer_size_t) const until_turn_elem_by_elem,
        char const* SIXTRL_RESTRICT config_str )
    {
        TrackJobBaseNew* ptr_track_job =
            SIXTRL_CXX_NAMESPACE::TrackJobNew_create( arch_id, config_str );

        if( ptr_track_job != nullptr )
        {
            if( !ptr_track_job->reset( pbuffer, pset_indices_begin,
                    pset_indices_end, belements_buffer, output_buffer,
                        until_turn_elem_by_elem ) )
            {
                delete ptr_track_job;
                ptr_track_job = nullptr;
            }
        }

        return ptr_track_job;
    }

    template< typename Iter > TrackJobBaseNew* TrackJobNew_new(
        std::string const& SIXTRL_RESTRICT_REF arch_str,
        ::NS(Buffer)* SIXTRL_RESTRICT pbuffer,
        Iter pset_indices_begin, Iter pset_indices_end,
        ::NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        ::NS(buffer_size_t) const until_turn_elem_by_elem,
        char const* SIXTRL_RESTRICT config_str )
    {
        return SIXTRL_CXX_NAMESPACE::TrackJobNew_new(
            SIXTRL_CXX_NAMESPACE::ArchInfo_arch_string_to_arch_id( arch_str ),
                pbuffer, pset_indices_begin, pset_indices_end, belements_buffer,
                    output_buffer, until_turn_elem_by_elem, config_str );
    }
}

#endif /* C++, Host */

#endif /* SIXTRACKLIB_SIXTRACKLIB_COMMON_TRACK_TRACK_JOB_BASE_HPP__ */

/*end: sixtracklib/common/track/track_job_base.hpp */