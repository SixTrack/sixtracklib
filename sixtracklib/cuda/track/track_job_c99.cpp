#include "sixtracklib/cuda/track_job.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/track/track_job_base.h"
#include "sixtracklib/common/track/track_job_base.hpp"

#include "sixtracklib/cuda/definitions.h"
#include "sixtracklib/cuda/controller.h"
#include "sixtracklib/cuda/argument.h"
#include "sixtracklib/cuda/track_job.hpp"

namespace st = SIXTRL_CXX_NAMESPACE;

::NS(ctrl_size_t) NS(CudaTrackJob_get_num_available_nodes)()
{
    return st::CudaTrackJob::NumAvailableNodes();
}

::NS(ctrl_size_t) NS(CudaTrackJob_get_available_node_ids_list)(
    ::NS(ctrl_size_t)  const max_num_node_ids,
    ::NS(NodeId)* SIXTRL_RESTRICT node_ids_begin )
{
    return st::CudaTrackJob::GetAvailableNodeIdsList(
        max_num_node_ids, node_ids_begin );
}

::NS(ctrl_size_t) NS(CudaTrackJob_get_available_node_indices_list)(
    ::NS(ctrl_size_t)  const max_num_node_indices,
    ::NS(node_index_t)* SIXTRL_RESTRICT node_indices_begin )
{
    return st::CudaTrackJob::GetAvailableNodeIndicesList(
        max_num_node_indices, node_indices_begin );
}

/* ------------------------------------------------------------------------- */


::NS(CudaTrackJob)* NS(CudaTrackJob_create)()
{
    return new st::CudaTrackJob;
}

::NS(CudaTrackJob)* NS(CudaTrackJob_new_from_config_str)(
    char const* SIXTRL_RESTRICT config_str )
{
    return new st::CudaTrackJob( config_str );
}

::NS(CudaTrackJob)* NS(CudaTrackJob_new)(
    char const* SIXTRL_RESTRICT node_id_str,
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer )
{
    return new st::CudaTrackJob(
        node_id_str, particles_buffer, beam_elements_buffer );
}

::NS(CudaTrackJob)* NS(CudaTrackJob_new_with_output)(
    char const* SIXTRL_RESTRICT node_id_str,
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    ::NS(buffer_size_t) const until_turn_elem_by_elem )
{
    return new st::CudaTrackJob( node_id_str, particles_buffer,
        beam_elements_buffer, output_buffer, until_turn_elem_by_elem );
}

::NS(CudaTrackJob)* NS(CudaTrackJob_new_detailed)(
    char const* SIXTRL_RESTRICT node_id_str,
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(buffer_size_t) const num_psets,
    ::NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    ::NS(buffer_size_t) const until_turn_elem_by_elem,
    char const* SIXTRL_RESTRICT config_str )
{
    return new st::CudaTrackJob( node_id_str, particles_buffer, num_psets,
        pset_indices_begin, beam_elements_buffer, output_buffer,
            until_turn_elem_by_elem, config_str );
}

/* ------------------------------------------------------------------------- */

bool NS(CudaTrackJob_has_controller)(
    const ::NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job )
{
    return ( ( track_job != nullptr ) &&
             ( track_job->hasCudaController() ) );
}

NS(CudaController)* NS(CudaTrackJob_get_ptr_controller)(
    ::NS(CudaTrackJob)* SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr ) ? track_job->ptrCudaController() : nullptr;
}

NS(CudaController) const* NS(CudaTrackJob_get_ptr_const_controller)(
    const ::NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr ) ? track_job->ptrCudaController() : nullptr;
}

/* ------------------------------------------------------------------------- */

bool NS(CudaTrackJob_has_particles_arg)(
    const ::NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job )
{
    return ( ( track_job != nullptr ) && ( track_job->hasCudaParticlesArg() ) );
}

NS(CudaArgument)* NS(CudaTrackJob_get_ptr_particles_arg)(
    ::NS(CudaTrackJob)* SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr )
        ? track_job->ptrCudaParticlesArg() : nullptr;
}

NS(CudaArgument) const* NS(CudaTrackJob_get_ptr_const_particles_arg)(
    const ::NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr )
        ? track_job->ptrCudaParticlesArg() : nullptr;
}

/* ------------------------------------------------------------------------- */

bool NS(CudaTrackJob_has_beam_elements_arg)(
    const ::NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job )
{
    return ( ( track_job != nullptr ) && ( track_job->hasCudaParticlesArg() ) );
}

NS(CudaArgument)* NS(CudaTrackJob_get_ptr_beam_elements_arg)(
    ::NS(CudaTrackJob)* SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr )
        ? track_job->ptrCudaBeamElementsArg() : nullptr;
}

NS(CudaArgument) const* NS(CudaTrackJob_get_ptr_const_beam_elements_arg)(
    const ::NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr )
        ? track_job->ptrCudaBeamElementsArg() : nullptr;
}

/* ------------------------------------------------------------------------- */

bool NS(CudaTrackJob_has_output_arg)(
    const ::NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job )
{
    return ( ( track_job != nullptr ) &&
             ( track_job->hasCudaOutputArg() ) );
}

NS(CudaArgument)* NS(CudaTrackJob_get_ptr_output_arg)(
    ::NS(CudaTrackJob)* SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr )
        ? track_job->ptrCudaOutputArg() : nullptr;
}

NS(CudaArgument) const* NS(CudaTrackJob_get_ptr_const_output_arg)(
    const ::NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr )
        ? track_job->ptrCudaOutputArg() : nullptr;
}

/* ------------------------------------------------------------------------- */

bool NS(CudaTrackJob_has_elem_by_elem_config_arg)(
    const ::NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job )
{
    return ( ( track_job != nullptr ) &&
             ( track_job->hasCudaElemByElemConfigArg() ) );
}

NS(CudaArgument)* NS(CudaTrackJob_get_ptr_elem_by_elem_config_arg)(
    ::NS(CudaTrackJob)* SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr )
        ? track_job->ptrCudaElemByElemConfigArg() : nullptr;
}

NS(CudaArgument) const* NS(CudaTrackJob_get_ptr_const_elem_by_elem_config_arg)(
    const ::NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr )
        ? track_job->ptrCudaElemByElemConfigArg() : nullptr;
}

/* ------------------------------------------------------------------------- */

bool NS(CudaTrackJob_has_debug_register_arg)(
    const ::NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job )
{
    return ( ( track_job != nullptr ) &&
             ( track_job->hasCudaDebugRegisterArg() ) );
}

NS(CudaArgument)* NS(CudaTrackJob_get_ptr_debug_register_arg)(
    ::NS(CudaTrackJob)* SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr )
        ? track_job->ptrCudaDebugRegisterArg() : nullptr;
}

NS(CudaArgument) const* NS(CudaTrackJob_get_ptr_const_debug_register_arg)(
    const ::NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr )
        ? track_job->ptrCudaDebugRegisterArg() : nullptr;
}

/* ------------------------------------------------------------------------- */

bool NS(CudaTrackJob_has_particles_addr_arg)(
    const ::NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job )
{
    return ( ( track_job != nullptr ) &&
             ( track_job->hasCudaParticlesAddrArg() ) );
}

NS(CudaArgument)* NS(CudaTrackJob_get_ptr_particles_addr_arg)(
    ::NS(CudaTrackJob)* SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr )
        ? track_job->ptrCudaParticlesAddrArg() : nullptr;
}

NS(CudaArgument) const* NS(CudaTrackJob_get_ptr_const_particles_addr_arg)(
    const ::NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr )
        ? track_job->ptrCudaParticlesAddrArg() : nullptr;
}

/* end: sixtracklib/cuda/track/track_job_c99.cpp */
