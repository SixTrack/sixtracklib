#include "sixtracklib/opencl/make_track_job.h"

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <utility>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/modules.h"
#include "sixtracklib/common/internal/track_job_base.h"
#include "sixtracklib/opencl/track_job_cl.h"

namespace st = SIXTRL_CXX_NAMESPACE;

namespace SIXTRL_CXX_NAMESPACE
{
    st::TrackJobBase* TrackJobCl_create( 
        char const* SIXTRL_RESTRICT device_id_str, 
        char const* SIXTRL_RESTRICT config_str )
    {
        return new st::TrackJobCl( device_id_str, config_str );
    }
    
    st::TrackJobBase* TrackJobCl_create( 
        std::string const& SIXTRL_RESTRICT_REF device_id_str,
        Buffer& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer::size_type const num_particle_sets,
        Buffer::size_type const* SIXTRL_RESTRICT pset_indices_begin,
        Buffer& SIXTRL_RESTRICT_REF beam_elemements_buffer,
        Buffer* SIXTRL_RESTRICT output_buffer,
        Buffer::size_type const dump_elem_by_elem_turns,
        std::string const& SIXTRL_RESTRICT_REF config_str )
    {
        return new st::TrackJobCl( device_id_str, particles_buffer,
            pset_indices_begin, pset_indices_begin + num_particle_sets, 
                beam_elemements_buffer, output_buffer, 
                    dump_elem_by_elem_turns, config_str );
    }
    
    st::TrackJobBase* TrackJobCl_create( 
        char const* SIXTRL_RESTRICT device_id_str, 
        ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
        ::NS(buffer_size_t) const num_particle_sets,
        ::NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
        ::NS(Buffer)* SIXTRL_RESTRICT belemements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        ::NS(buffer_size_t) const dump_elem_by_elem_turns,
        char const* SIXTRL_RESTRICT config_str )
    {
        return new st::TrackJobCl( device_id_str, particles_buffer,
                    num_particle_sets, pset_indices_begin, belemements_buffer,
                        output_buffer, dump_elem_by_elem_turns, config_str );
    }
}

::NS(TrackJobBase)* NS(TrackJobCl_create)(
    char const* SIXTRL_RESTRICT device_id_str, 
    char const* SIXTRL_RESTRICT config_str )
{
    return new st::TrackJobCl( device_id_str, config_str );
}

::NS(TrackJobBase)* NS(TrackJobCl_create_detailed)( 
        char const* SIXTRL_RESTRICT device_id_str, 
        ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
        ::NS(buffer_size_t) const num_particle_sets,
        ::NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
        ::NS(Buffer)* SIXTRL_RESTRICT belemements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        ::NS(buffer_size_t) const dump_elem_by_elem_turns,
        char const* SIXTRL_RESTRICT config_str )
{
    return new st::TrackJobCl( device_id_str, particles_buffer,
                num_particle_sets, pset_indices_begin, belemements_buffer,
                    output_buffer, dump_elem_by_elem_turns, config_str );
}

/* end: sixtracklib/opencl/internal/make_track_job.cpp */
