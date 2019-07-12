#include "sixtracklib/common/track/track_job_cpu.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/track/track_job_base.h"
#include "sixtracklib/common/track/track_job_cpu.hpp"

namespace st = SIXTRL_CXX_NAMESPACE;

::NS(CpuTrackJob)* NS(CpuTrackJob_create)()
{
    return new st::CpuTrackJob( nullptr );
}

::NS(CpuTrackJob)* NS(CpuTrackJob_new_from_config_str)(
    char const* SIXTRL_RESTRICT config_str )
{
    return new st::CpuTrackJob( config_str );
}

::NS(CpuTrackJob)* NS(CpuTrackJob_new)(
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer )
{
    return new st::CpuTrackJob( particles_buffer, beam_elements_buffer );
}

::NS(CpuTrackJob)* NS(CpuTrackJob_new_with_output)(
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    ::NS(buffer_size_t) const until_turn_elem_by_elem )
{
    return new st::CpuTrackJob( particles_buffer, beam_elements_buffer,
                                output_buffer, until_turn_elem_by_elem );
}

::NS(CpuTrackJob)* NS(CpuTrackJob_new_detailed)(
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(buffer_size_t) const num_psets,
    ::NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    ::NS(buffer_size_t) const until_turn_elem_by_elem,
    char const* SIXTRL_RESTRICT config_str )
{
    return new st::CpuTrackJob( particles_buffer, num_psets,
        pset_indices_begin, beam_elements_buffer, output_buffer,
            until_turn_elem_by_elem, config_str );
}

/* end: sixtracklib/common/track/track_job_cpu_c99.cpp */
