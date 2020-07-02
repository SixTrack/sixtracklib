#include "sixtracklib/common/track/track_job_base.h"

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/node_id.hpp"
#include "sixtracklib/common/control/arch_base.hpp"
#include "sixtracklib/common/track/definitions.h"
#include "sixtracklib/common/track/track_job_base.hpp"

namespace st = SIXTRL_CXX_NAMESPACE;

::NS(TrackJobBaseNew)* NS(TrackJobNew_create)(
    char const* SIXTRL_RESTRICT arch_str,
    char const* SIXTRL_RESTRICT config_str )
{
    return st::TrackJobNew_create( arch_str, config_str );
}

::NS(TrackJobBaseNew)* NS(TrackJobNew_new)(
    char const* SIXTRL_RESTRICT arch_str,
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    char const* SIXTRL_RESTRICT config_str )
{
    return st::TrackJobNew_new(
        arch_str, particles_buffer, beam_elem_buffer, config_str );
}

::NS(TrackJobBaseNew)* NS(TrackJobNew_new_with_output)(
    char const* SIXTRL_RESTRICT arch_str,
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    ::NS(buffer_size_t) const dump_elem_by_elem_turns,
    char const* SIXTRL_RESTRICT config_str )
{
    return st::TrackJobNew_new(
        arch_str, particles_buffer, beam_elem_buffer, output_buffer,
            dump_elem_by_elem_turns, config_str );
}

::NS(TrackJobBaseNew)* NS(TrackJobNew_new_detailed)(
    char const* SIXTRL_RESTRICT arch_str,
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(buffer_size_t) const num_particle_sets,
    ::NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    ::NS(buffer_size_t) const dump_elem_by_elem_turns,
    char const* SIXTRL_RESTRICT config_str )
{
    return st::TrackJobNew_new( arch_str, particles_buffer,
        num_particle_sets, pset_indices_begin, beam_elem_buffer, output_buffer,
            dump_elem_by_elem_turns, config_str );
}

/* ------------------------------------------------------------------------- */

void NS(TrackJobNew_delete)( ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    if( job != nullptr ) delete job;
}

::NS(track_status_t) NS(TrackJobNew_track_until)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const until_turn )
{
    return ( job != nullptr )
        ? job->trackUntil( until_turn ) : st::TRACK_STATUS_GENERAL_FAILURE;
}

::NS(track_status_t) NS(TrackJobNew_track_elem_by_elem)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const until_turn )
{
    return ( job != nullptr )
        ? job->trackElemByElem( until_turn ) : st::TRACK_STATUS_GENERAL_FAILURE;
}

::NS(track_status_t) NS(TrackJobNew_track_line)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const belem_begin_index,
    ::NS(buffer_size_t) const belem_end_index,
    bool const finish_turn )
{
    return ( job != nullptr )
        ? job->trackLine( belem_begin_index, belem_end_index, finish_turn )
        : st::TRACK_STATUS_GENERAL_FAILURE;
}

/* ------------------------------------------------------------------------- */

::NS(track_job_collect_flag_t) NS(TrackJobNew_collect)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->collect() : st::TRACK_JOB_IO_NONE;
}

::NS(track_job_collect_flag_t) NS(TrackJobNew_collect_detailed)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(track_job_collect_flag_t) const flags )
{
    return ( job != nullptr ) ? job->collect( flags ) : st::TRACK_JOB_IO_NONE;
}

::NS(arch_status_t) NS(TrackJobNew_collect_particles)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->collectParticles() : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t) NS(TrackJobNew_collect_beam_elements)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->collectBeamElements() : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t) NS(TrackJobNew_collect_output)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->collectOutput() : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t) NS(TrackJobNew_collect_debug_flag)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->collectDebugFlag() : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t) NS(TrackJobNew_collect_particles_addresses)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->collectParticlesAddresses() : st::ARCH_STATUS_GENERAL_FAILURE;
}

void NS(TrackJobNew_enable_collect_particles)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    if( job != nullptr ) job->enableCollectParticles();
}

void NS(TrackJobNew_disable_collect_particles)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    if( job != nullptr ) job->disableCollectParticles();
}

bool NS(TrackJobNew_is_collecting_particles)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->isCollectingParticles() : false;
}

void NS(TrackJobNew_enable_collect_beam_elements)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    if( job != nullptr ) job->enableCollectBeamElements();
}

void NS(TrackJobNew_disable_collect_beam_elements)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    if( job != nullptr ) job->disableCollectBeamElements();
}

bool NS(TrackJobNew_is_collecting_beam_elements)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->isCollectingBeamElements() : false;
}

void NS(TrackJobNew_enable_collect_output)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    if( job != nullptr ) job->enableCollectOutput();
}

void NS(TrackJobNew_disable_collect_output)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    if( job != nullptr ) job->disableCollectOutput();
}

bool NS(TrackJobNew_is_collecting_output)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->isCollectingOutput() : false;
}

::NS(track_job_collect_flag_t) NS(TrackJobNew_get_collect_flags)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->collectFlags() : st::TRACK_JOB_IO_DEFAULT_FLAGS;
}

void NS(TrackJobNew_set_collect_flags)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(track_job_collect_flag_t) const flags )
{
    if( job != nullptr ) job->setCollectFlags( flags );
}

bool NS(TrackJobNew_requires_collecting)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->requiresCollecting() : false;
}

/* ------------------------------------------------------------------------- */

::NS(track_job_push_flag_t) NS(TrackJobNew_push)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(track_job_push_flag_t) const flag )
{
    return ( job != nullptr ) ? job->push( flag ) : st::TRACK_JOB_IO_NONE;
}

::NS(track_job_push_flag_t) NS(TrackJobNew_push_particles)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->pushParticles() : st::TRACK_JOB_IO_NONE;
}

::NS(track_job_push_flag_t) NS(TrackJobNew_push_beam_elements)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->pushBeamElements() : st::TRACK_JOB_IO_NONE;
}

::NS(track_job_push_flag_t) NS(TrackJobNew_push_output)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->pushOutput() : st::TRACK_JOB_IO_NONE;
}

/* ------------------------------------------------------------------------- */

bool NS(TrackJobNew_can_fetch_particle_addresses)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( ( job != nullptr ) && ( job->canFetchParticleAddresses() ) );
}

bool NS(TrackJobNew_has_particle_addresses)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( ( job != nullptr ) && ( job->hasParticleAddresses() ) );
}

::NS(arch_status_t) NS(TrackJobNew_fetch_particle_addresses)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->fetchParticleAddresses()
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t) NS(TrackJobNew_clear_particle_addresses)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const particle_set_index )
{
    return ( job != nullptr )
        ? job->clearParticleAddresses( particle_set_index )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t) NS(TrackJobNew_clear_all_particle_addresses)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->clearAllParticleAddresses()
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(ParticlesAddr) const* NS(TrackJobNew_get_particle_addresses)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const particle_set_index )
{
    return ( job != nullptr )
        ? job->particleAddresses( particle_set_index ) : nullptr;
}

::NS(Buffer) const* NS(TrackJobNew_get_ptr_particle_addresses_buffer)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrCParticleAddressesBuffer() : nullptr;
}

/* ------------------------------------------------------------------------- */

bool NS(TrackJobNew_is_in_debug_mode)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( ( job != nullptr ) && ( job->isInDebugMode() ) );
}

::NS(arch_status_t) NS(TrackJobNew_enable_debug_mode)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->enableDebugMode()
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t) NS(TrackJobNew_disable_debug_mode)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->disableDebugMode()
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

/* ------------------------------------------------------------------------- */

void NS(TrackJobNew_clear)( NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    if( job != nullptr ) job->clear();
}

::NS(arch_status_t) NS(TrackJobNew_reset)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT ptr_output_buffer )
{
    return ( job != nullptr )
        ? job->reset( particles_buffer, beam_elem_buffer, ptr_output_buffer )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t) NS(TrackJobNew_reset_particle_set)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(buffer_size_t) const particle_set_index,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT output_buffer )
{
    return ( job != nullptr ) ? job->reset( particles_buffer,
            particle_set_index, beam_elem_buffer, output_buffer )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}


::NS(arch_status_t) NS(TrackJobNew_reset_with_output)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    ::NS(buffer_size_t) const dump_elem_by_elem_turns )
{
    return ( job != nullptr ) ? job->reset( particles_buffer, beam_elem_buffer,
                output_buffer, dump_elem_by_elem_turns )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t)  NS(TrackJobNew_reset_detailed)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(buffer_size_t) const num_particle_sets,
    ::NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    ::NS(buffer_size_t) const dump_elem_by_elem_turns )
{
    return ( job != nullptr ) ? job->reset( particles_buffer, num_particle_sets,
            particle_set_indices_begin, beam_elem_buffer, output_buffer,
                dump_elem_by_elem_turns )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t)  NS(TrackJobNew_select_particle_set)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const particle_set_index )
{
    return ( job != nullptr ) ? job->selectParticleSet( particle_set_index )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t) NS(TrackJobNew_assign_output_buffer)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(Buffer)* SIXTRL_RESTRICT ptr_output_buffer )
{
    return ( job != nullptr ) ? job->assignOutputBuffer( ptr_output_buffer )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

/* ------------------------------------------------------------------------- */

::NS(arch_id_t) NS(TrackJobNew_get_arch_id)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->archId() : st::ARCHITECTURE_ILLEGAL;
}

bool NS(TrackJobNew_has_arch_string)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( ( job != nullptr ) && ( job->hasArchStr() ) );
}

char const* NS(TrackJobNew_get_arch_string)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrArchStr() : nullptr;
}

bool NS(TrackJobNew_has_config_str)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( ( job != nullptr ) && ( job->hasConfigStr() ) );
}

char const* NS(TrackJobNew_get_config_str)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->configStr().c_str() : nullptr;
}

/* ------------------------------------------------------------------------- */

::NS(buffer_size_t) NS(TrackJobNew_get_num_particle_sets)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->numParticleSets() : st::buffer_size_t{ 0 };
}

::NS(buffer_size_t) const* NS(TrackJobNew_get_particle_set_indices_begin)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->particleSetIndicesBegin() : nullptr;
}

::NS(buffer_size_t) const* NS(TrackJobNew_get_particle_set_indices_end)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->particleSetIndicesEnd() : nullptr;
}

::NS(buffer_size_t) NS(TrackJobNew_get_particle_set_index)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const n )
{
    return ( job != nullptr )
        ? job->particleSetIndex( n ) : st::buffer_size_t{ 0 };
}

/* ------------------------------------------------------------------------- */

::NS(buffer_size_t) NS(TrackJobNew_get_total_num_of_particles)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->totalNumParticles() : st::buffer_size_t{ 0 };
}

::NS(particle_index_t) NS(TrackJobNew_get_min_particle_id)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    using index_t = ::NS(particle_index_t);
    return ( job != nullptr ) ? job->minParticleId() : index_t{ -1 };
}

::NS(particle_index_t) NS(TrackJobNew_get_max_particle_id)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    using index_t = ::NS(particle_index_t);
    return ( job != nullptr ) ? job->maxParticleId() : index_t{ -1 };
}

::NS(particle_index_t) NS(TrackJobNew_get_min_element_id)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    using index_t = ::NS(particle_index_t);
    return ( job != nullptr ) ? job->minElementId() : index_t{ -1 };
}

::NS(particle_index_t) NS(TrackJobNew_get_max_element_id)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    using index_t = ::NS(particle_index_t);
    return ( job != nullptr ) ? job->maxElementId() : index_t{ -1 };
}

::NS(particle_index_t) NS(TrackJobNew_get_min_initial_turn_id)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    using index_t = ::NS(particle_index_t);
    return ( job != nullptr ) ? job->minInitialTurnId() : index_t{ -1 };
}

::NS(particle_index_t) NS(TrackJobNew_get_max_initial_turn_id)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    using index_t = ::NS(particle_index_t);
    return ( job != nullptr ) ? job->maxInitialTurnId() : index_t{ -1 };
}

/* ------------------------------------------------------------------------- */

::NS(Buffer)* NS(TrackJobNew_get_particles_buffer)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrCParticlesBuffer() : nullptr;
}

::NS(Buffer) const* NS(TrackJobNew_get_const_particles_buffer)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrCParticlesBuffer() : nullptr;
}

::NS(Buffer)* NS(TrackJobNew_get_beam_elements_buffer)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrCBeamElementsBuffer() : nullptr;
}

::NS(Buffer) const* NS(TrackJobNew_get_const_beam_elements_buffer)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrCBeamElementsBuffer() : nullptr;
}

/* ------------------------------------------------------------------------- */

bool NS(TrackJobNew_has_output_buffer)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( ( job != nullptr ) && ( job->hasOutputBuffer() ) );
}

bool NS(TrackJobNew_owns_output_buffer)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( ( job != nullptr ) && ( job->ownsOutputBuffer() ) );
}

bool NS(TrackJobNew_has_elem_by_elem_output)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( ( job != nullptr ) && ( job->hasElemByElemOutput() ) );
}

bool NS(TrackJobNew_has_beam_monitor_output)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( ( job != nullptr ) && ( job->hasBeamMonitorOutput() ) );
}

::NS(buffer_size_t) NS(TrackJobNew_get_beam_monitor_output_buffer_offset)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    SIXTRL_ASSERT( job != nullptr );
    return job->beamMonitorsOutputBufferOffset();
}

::NS(buffer_size_t) NS(TrackJobNew_get_elem_by_elem_output_buffer_offset)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    SIXTRL_ASSERT( job != nullptr );
    return job->elemByElemOutputBufferOffset();
}

::NS(buffer_size_t) NS(TrackJobNew_get_num_elem_by_elem_turns)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->numElemByElemTurns() : st::buffer_size_t{ 0 };
}

::NS(Buffer)* NS(TrackJobNew_get_output_buffer)(
    NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrCOutputBuffer() : nullptr;
}

::NS(Buffer) const* NS(TrackJobNew_get_const_output_buffer)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrCOutputBuffer() : nullptr;
}

/* ------------------------------------------------------------------------- */

bool NS(TrackJobNew_has_beam_monitors)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( ( job != nullptr ) && ( job->hasBeamMonitors() ) );
}

::NS(buffer_size_t) NS(TrackJobNew_get_num_beam_monitors)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->numBeamMonitors() : st::buffer_size_t{ 0 };
}

::NS(buffer_size_t) const* NS(TrackJobNew_get_beam_monitor_indices_begin)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->beamMonitorIndicesBegin() : nullptr;
}

::NS(buffer_size_t) const* NS(TrackJobNew_get_beam_monitor_indices_end)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->beamMonitorIndicesEnd() : nullptr;
}

::NS(buffer_size_t) NS(TrackJobNew_get_beam_monitor_index)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const n )
{
    return ( job != nullptr )
        ? job->beamMonitorIndex( n ) : st::buffer_size_t{ 0 };
}

/* ------------------------------------------------------------------------- */

bool NS(TrackJobNew_has_elem_by_elem_config)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( ( job != nullptr ) && ( job->hasElemByElemConfig() ) );
}

NS(ElemByElemConfig) const* NS(TrackJobNew_get_elem_by_elem_config)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrElemByElemConfig() : nullptr;
}

bool NS(TrackJobNew_is_elem_by_elem_config_rolling)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( ( job != nullptr ) && ( job->elemByElemRolling() ) );
}

bool NS(TrackJobNew_get_default_elem_by_elem_config_rolling_flag)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( ( job != nullptr ) && ( job->defaultElemByElemRolling() ) );
}

void NS(TrackJobNew_set_default_elem_by_elem_config_rolling_flag)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job, bool const is_rolling_flag )
{
    if( job != nullptr ) job->setDefaultElemByElemRolling( is_rolling_flag );
}

::NS(elem_by_elem_order_t) NS(TrackJobNew_get_elem_by_elem_config_order)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->elemByElemOrder()
        : ::NS(ELEM_BY_ELEM_ORDER_INVALID);
}

::NS(elem_by_elem_order_t)
NS(TrackJobNew_get_default_elem_by_elem_config_order)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->defaultElemByElemOrder()
        : ::NS(ELEM_BY_ELEM_ORDER_INVALID);
}

void NS(TrackJobNew_set_default_elem_by_elem_config_order)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(elem_by_elem_order_t) const order )
{
    if( job != nullptr ) job->setDefaultElemByElemOrder( order );
}

/* ------------------------------------------------------------------------- */

bool NS(TrackJobNew_uses_controller)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( ( job != nullptr ) && ( job->usesController() ) );
}

bool NS(TrackJobNew_uses_arguments)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job )
{
    return ( ( job != nullptr ) && ( job->usesArguments() ) );
}

/* ------------------------------------------------------------------------- */

::NS(AssignAddressItem)* NS(TrackJobNew_add_assign_address_item)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    const ::NS(AssignAddressItem) *const SIXTRL_RESTRICT assign_item_to_add )
{
    st::AssignAddressItem const* ptr_cxx_item = reinterpret_cast<
        st::AssignAddressItem const* >( assign_item_to_add );

    return ( ( job != nullptr ) && ( ptr_cxx_item != nullptr ) )
        ? job->add_assign_address_item( *ptr_cxx_item ) : nullptr;
}

::NS(AssignAddressItem)* NS(TrackJobNew_add_assign_address_item_detailed)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(object_type_id_t) const dest_type_id,
    ::NS(buffer_size_t) const dest_buffer_id,
    ::NS(buffer_size_t) const dest_elem_index,
    ::NS(buffer_size_t) const dest_pointer_offset,
    ::NS(object_type_id_t) const src_type_id,
    ::NS(buffer_size_t) const src_buffer_id,
    ::NS(buffer_size_t) const src_elem_index,
    ::NS(buffer_size_t) const src_pointer_offset )
{
    return ( job != nullptr )
        ? job->add_assign_address_item(
            dest_type_id, dest_buffer_id, dest_elem_index, dest_pointer_offset,
            src_type_id, src_buffer_id, src_elem_index, src_pointer_offset )
        : nullptr;
}

::NS(arch_status_t) NS(TrackJobNew_remove_assign_address_item)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(AssignAddressItem)* SIXTRL_RESTRICT item_to_remove )
{
    st::AssignAddressItem const* ptr_cxx_item = reinterpret_cast<
        st::AssignAddressItem const* >( item_to_remove );

    return ( ( job != nullptr ) && ( ptr_cxx_item != nullptr ) )
        ? job->remove_assign_address_item( *ptr_cxx_item )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t) NS(TrackJobNew_remove_assign_address_item_by_key_and_index)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    const ::NS(TrackJobDestSrcBufferIds) *const SIXTRL_RESTRICT_REF key,
    ::NS(buffer_size_t) const index_of_item_to_remove )
{
    return ( ( job != nullptr ) && ( key != nullptr ) )
        ? job->remove_assign_address_item( *key, index_of_item_to_remove )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

bool NS(TrackJobNew_has_assign_address_item)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    const ::NS(AssignAddressItem) *const SIXTRL_RESTRICT item ) SIXTRL_NOEXCEPT
{
    st::AssignAddressItem const* ptr_cxx_item = reinterpret_cast<
        st::AssignAddressItem const* >( item );

    return ( ( job != nullptr ) && ( ptr_cxx_item != nullptr ) &&
             ( job->has_assign_address_item( *ptr_cxx_item ) ) );
}

bool NS(TrackJobNew_has_assign_item_by_index)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const dest_buffer_id,
    ::NS(buffer_size_t) const src_buffer_id,
    ::NS(buffer_size_t) const item_index ) SIXTRL_NOEXCEPT
{
    return ( ( job != nullptr ) && ( job->num_assign_items(
                dest_buffer_id, src_buffer_id ) > item_index ) );
}

bool NS(TrackJobNew_has_assign_address_item_detailed)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    ::NS(object_type_id_t) const dest_type_id,
    ::NS(buffer_size_t) const dest_buffer_id,
    ::NS(buffer_size_t) const dest_elem_index,
    ::NS(buffer_size_t) const dest_pointer_offset,
    ::NS(object_type_id_t) const src_type_id,
    ::NS(buffer_size_t) const src_buffer_id,
    ::NS(buffer_size_t) const src_elem_index,
    ::NS(buffer_size_t) const src_pointer_offset ) SIXTRL_NOEXCEPT
{
    return ( ( job != nullptr ) &&
        ( job->has_assign_address_item( dest_type_id, dest_buffer_id,
                dest_elem_index, dest_pointer_offset, src_type_id,
                    src_buffer_id, src_elem_index, src_pointer_offset ) ) );
}

::NS(buffer_size_t) NS(TrackJobNew_index_of_assign_address_item_detailed)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    ::NS(object_type_id_t) const dest_type_id,
    ::NS(buffer_size_t) const dest_buffer_id,
    ::NS(buffer_size_t) const dest_elem_index,
    ::NS(buffer_size_t) const dest_pointer_offset,
    ::NS(object_type_id_t) const src_type_id,
    ::NS(buffer_size_t) const src_buffer_id,
    ::NS(buffer_size_t) const src_elem_index,
    ::NS(buffer_size_t) const src_pointer_offset ) SIXTRL_NOEXCEPT
{
    return ( job != nullptr )
        ? job->index_of_assign_address_item(
            dest_type_id, dest_buffer_id, dest_elem_index, dest_pointer_offset,
            src_type_id, src_buffer_id, src_elem_index, src_pointer_offset )
        : ::NS(buffer_size_t){ 0 };
}

::NS(buffer_size_t) NS(TrackJobNew_index_of_assign_address_item)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    const ::NS(AssignAddressItem) *const SIXTRL_RESTRICT item ) SIXTRL_NOEXCEPT
{
    st::AssignAddressItem const* ptr_cxx_item = reinterpret_cast<
        st::AssignAddressItem const* >( item );

    return ( ( job != nullptr ) && ( ptr_cxx_item != nullptr ) )
        ? job->index_of_assign_address_item( *ptr_cxx_item )
        : ::NS(buffer_size_t){ 0 };
}

bool NS(TrackJobNew_has_assign_items)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const dest_buffer_id,
    ::NS(buffer_size_t) const src_buffer_id ) SIXTRL_NOEXCEPT
{
    return ( ( job != nullptr ) &&
             ( job->has_assign_items( dest_buffer_id, src_buffer_id ) ) );
}

::NS(buffer_size_t) NS(TrackJobNew_num_assign_items)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const dest_buffer_id,
    ::NS(buffer_size_t) const src_buffer_id ) SIXTRL_NOEXCEPT
{
    return ( job != nullptr )
        ? job->num_assign_items( dest_buffer_id, src_buffer_id )
        : ::NS(buffer_size_t){ 0 };
}

::NS(buffer_size_t) NS(TrackJobNew_total_num_assign_items)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job ) SIXTRL_NOEXCEPT
{
    return ( job != nullptr )
        ? job->total_num_assign_items() : ::NS(buffer_size_t){ 0 };
}

::NS(AssignAddressItem) const* NS(TrackJobNew_ptr_assign_address_item)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    const ::NS(AssignAddressItem) *const SIXTRL_RESTRICT item ) SIXTRL_NOEXCEPT
{
    ::NS(AssignAddressItem) const* ptr_item = nullptr;

    if( ( job != nullptr ) && ( item != nullptr ) )
    {
        st::AssignAddressItem const* _ptr = job->ptr_assign_address_item(
            *( reinterpret_cast< st::AssignAddressItem const* >( item ) ) );
        if( _ptr != nullptr ) ptr_item = _ptr->getCApiPtr();
    }

    return ptr_item;
}

::NS(AssignAddressItem) const* NS(TrackJobNew_ptr_assign_address_item_detailed)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    ::NS(object_type_id_t) const dest_type_id,
    ::NS(buffer_size_t) const dest_buffer_id,
    ::NS(buffer_size_t) const dest_elem_index,
    ::NS(buffer_size_t) const dest_pointer_offset,
    ::NS(object_type_id_t) const src_type_id,
    ::NS(buffer_size_t) const src_buffer_id,
    ::NS(buffer_size_t) const src_elem_index,
    ::NS(buffer_size_t) const src_pointer_offset ) SIXTRL_NOEXCEPT
{
    ::NS(AssignAddressItem) const* ptr_item = nullptr;
    if( job != nullptr )
    {
        st::TrackJobBaseNew::size_type const assign_address_item_index =
            job->index_of_assign_address_item( dest_type_id, dest_buffer_id,
                dest_elem_index, dest_pointer_offset, src_type_id,
                    src_buffer_id, src_elem_index, src_pointer_offset );

        ptr_item = job->ptr_assign_address_item(
            dest_buffer_id, src_buffer_id, assign_address_item_index );
    }

    return ptr_item;
}

::NS(AssignAddressItem) const* NS(TrackJobNew_ptr_assign_address_item_by_index)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const dest_buffer_id,
    ::NS(buffer_size_t) const src_buffer_id,
    ::NS(buffer_size_t) const assign_address_item_index ) SIXTRL_NOEXCEPT
{
    ::NS(AssignAddressItem) const* ptr_item = nullptr;
    if( job != nullptr )
    {
        st::AssignAddressItem const* _ptr = job->ptr_assign_address_item(
            dest_buffer_id, src_buffer_id, assign_address_item_index );
        if( _ptr != nullptr ) ptr_item = _ptr->getCApiPtr();
    }

    return ptr_item;
}

::NS(buffer_size_t)
NS(TrackJobNew_num_distinct_available_assign_address_items_dest_src_pairs)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job ) SIXTRL_NOEXCEPT
{
    return ( job != nullptr )
        ? job->num_distinct_available_assign_address_items_dest_src_pairs()
        : ::NS(buffer_size_t){ 0 };
}

::NS(buffer_size_t) NS(TrackJobNew_available_assign_address_items_dest_src_pairs)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const max_num_pairs,
    ::NS(TrackJobDestSrcBufferIds)* pairs_begin ) SIXTRL_NOEXCEPT
{
    return ( job != nullptr )
        ? job->available_assign_address_items_dest_src_pairs(
            max_num_pairs, pairs_begin )
        : ::NS(buffer_size_t){ 0 };
}

::NS(Buffer)* NS(TrackJobNew_buffer_by_buffer_id)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const buffer_id ) SIXTRL_NOEXCEPT
{
    return ( job != nullptr ) ? job->buffer_by_buffer_id( buffer_id ) : nullptr;
}

::NS(Buffer) const* NS(TrackJobNew_const_buffer_by_buffer_id)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const buffer_id ) SIXTRL_NOEXCEPT
{
    return ( job != nullptr ) ? job->buffer_by_buffer_id( buffer_id ) : nullptr;
}

bool NS(TrackJobNew_is_buffer_by_buffer_id)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const buffer_id ) SIXTRL_NOEXCEPT
{
    return ( ( job != nullptr ) &&
             ( job->is_buffer_by_buffer_id)( buffer_id ) );
}

bool NS(TrackJobNew_is_raw_memory_by_buffer_id)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const buffer_id ) SIXTRL_NOEXCEPT
{
    return ( ( job != nullptr ) &&
             ( job->is_raw_memory_by_buffer_id( buffer_id ) ) );
}

SIXTRL_BUFFER_OBJ_ARGPTR_DEC ::NS(Object) const*
NS(TrackJobNew_assign_items_begin)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const dest_buffer_id,
    ::NS(buffer_size_t) const src_buffer_id ) SIXTRL_NOEXCEPT
{
    return ( job != nullptr )
        ? job->assign_items_begin( dest_buffer_id, src_buffer_id ) : nullptr;
}

SIXTRL_BUFFER_OBJ_ARGPTR_DEC ::NS(Object) const*
NS(TrackJobNew_assign_items_end)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const dest_buffer_id,
    ::NS(buffer_size_t) const src_buffer_id ) SIXTRL_NOEXCEPT
{
    return ( job != nullptr )
        ? job->assign_items_end( dest_buffer_id, src_buffer_id ) : nullptr;
}

::NS(TrackJobDestSrcBufferIds) const*
NS(TrackJobNew_assign_item_dest_src_begin)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job ) SIXTRL_NOEXCEPT
{
    return ( job != nullptr ) ? job->assign_item_dest_src_begin() : nullptr;
}

::NS(TrackJobDestSrcBufferIds) const*
NS(TrackJobNew_assign_item_dest_src_end)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job ) SIXTRL_NOEXCEPT
{
    return ( job != nullptr ) ? job->assign_item_dest_src_end() : nullptr;
}

::NS(arch_status_t) NS(TrackJobNew_commit_address_assignments)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->commit_address_assignments() : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t) NS(TrackJobNew_assign_all_addresses)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->assign_all_addresses() : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t) NS(TrackJobNew_assign_addresses)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const dest_buffer_id,
    ::NS(buffer_size_t) const src_buffer_id )
{
    return ( job != nullptr )
        ? job->assign_addresses( dest_buffer_id, src_buffer_id )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

/* ------------------------------------------------------------------------- */

::NS(arch_size_t) NS(TrackJobNew_stored_buffers_capacity)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job ) SIXTRL_NOEXCEPT
{
    return ( job != nullptr )
        ? job->stored_buffers_capacity() : ::NS(arch_size_t){ 0 };
}

::NS(arch_status_t) NS(TrackJobNew_reserve_stored_buffers_capacity)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(arch_size_t) const capacity )
{
    return ( job != nullptr )
        ? job->reserve_stored_buffers_capacity( capacity )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

bool NS(TrackJobNew_has_stored_buffers)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job ) SIXTRL_NOEXCEPT
{
    return ( ( job != nullptr ) && ( job->has_stored_buffers() ) );
}

::NS(arch_size_t) NS(TrackJobNew_num_stored_buffers)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job ) SIXTRL_NOEXCEPT
{
    return ( job != nullptr )
        ? job->num_stored_buffers() : ::NS(arch_size_t){ 0 };
}

::NS(arch_size_t) NS(TrackJobNew_min_stored_buffer_id)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job ) SIXTRL_NOEXCEPT
{
    return ( job != nullptr )
        ? job->min_stored_buffer_id() : st::ARCH_ILLEGAL_BUFFER_ID;
}

::NS(arch_size_t) NS(TrackJobNew_max_stored_buffer_id)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job ) SIXTRL_NOEXCEPT
{
    return ( job != nullptr )
        ? job->max_stored_buffer_id() : st::ARCH_ILLEGAL_BUFFER_ID;
}

::NS(arch_size_t) NS(TrackJobNew_create_stored_buffer)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const buffer_capacity )
{
    return ( job != nullptr ) ? job->add_stored_buffer( buffer_capacity )
        : st::ARCH_ILLEGAL_BUFFER_ID;
}

::NS(arch_size_t) NS(TrackJobNew_add_stored_buffer)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(Buffer)* SIXTRL_RESTRICT buffer, bool const take_ownership,
    bool const delete_ptr_after_move )
{
    return ( job != nullptr )
        ? job->add_stored_buffer( buffer, take_ownership, delete_ptr_after_move )
        : st::ARCH_ILLEGAL_BUFFER_ID;
}

bool NS(TrackJobNew_owns_stored_buffer)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    ::NS(arch_size_t) const buffer_id ) SIXTRL_NOEXCEPT
{
    return ( ( job != nullptr ) && ( job->owns_stored_buffer( buffer_id ) ) );
}

::NS(arch_status_t) NS(TrackJobNew_remove_stored_buffer)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(arch_size_t) const buffer_index )
{
    return ( job != nullptr )
        ? job->remove_stored_buffer( buffer_index )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(Buffer)* NS(TrackJobNew_stored_buffer)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(arch_size_t) const buffer_id ) SIXTRL_NOEXCEPT
{
    return ( job != nullptr ) ? job->ptr_stored_buffer( buffer_id ) : nullptr;
}

::NS(Buffer) const* NS(TrackJobNew_const_stored_buffer)(
    const ::NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    ::NS(arch_size_t) const buffer_id ) SIXTRL_NOEXCEPT
{
    return ( job != nullptr ) ? job->ptr_stored_buffer( buffer_id ) : nullptr;
}

::NS(arch_status_t) NS(TrackJobNew_push_stored_buffer)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(arch_size_t) const buffer_id )
{
    return ( job != nullptr )
        ? job->push_stored_buffer( buffer_id )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t) NS(TrackJobNew_collect_stored_buffer)(
    ::NS(TrackJobBaseNew)* SIXTRL_RESTRICT job,
    ::NS(arch_size_t) const buffer_id )
{
    return ( job != nullptr )
        ? job->collect_stored_buffer( buffer_id )
        : st::ARCH_STATUS_GENERAL_FAILURE;
}
#endif /* C++, Host */
