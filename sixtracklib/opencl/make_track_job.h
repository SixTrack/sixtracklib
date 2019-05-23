#ifndef SIXTRL_SIXTRACKLIB_OPENCL_MAKE_TRACK_JOB_H__
#define SIXTRL_SIXTRACKLIB_OPENCL_MAKE_TRACK_JOB_H__

#if !defined( SIXTRL_TRACK_JOB_CL_ID )
    #define   SIXTRL_TRACK_JOB_CL_ID 2
#endif /* !defined( SIXTRL_TRACK_JOB_CL_ID ) */

#if !defined( SIXTRL_TRACK_JOB_CL_STR )
    #define   SIXTRL_TRACK_JOB_CL_STR "opencl"
#endif /* !defined( SIXTRL_TRACK_JOB_CL_STR ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/generated/modules.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/internal/track_job_base.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    TrackJobBase* TrackJobCl_create( 
        char const* SIXTRL_RESTRICT device_id_str, 
        char const* SIXTRL_RESTRICT config_str );
    
    TrackJobBase* TrackJobCl_create( 
        std::string const& SIXTRL_RESTRICT_REF arch_str,
        Buffer& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer::size_type const num_particle_sets,
        Buffer::size_type const* SIXTRL_RESTRICT particle_set_indices_begin,
        Buffer& SIXTRL_RESTRICT_REF beam_elemements_buffer,
        Buffer* SIXTRL_RESTRICT output_buffer,
        Buffer::size_type const dump_elem_by_elem_turns,
        std::string const& SIXTRL_RESTRICT_REF config_str );
    
    TrackJobBase* TrackJobCl_create( 
        char const* SIXTRL_RESTRICT device_id_str, 
        ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
        ::NS(buffer_size_t) const num_particle_sets,
        ::NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
        ::NS(Buffer)* SIXTRL_RESTRICT belemements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        ::NS(buffer_size_t) const dump_elem_by_elem_turns,
        char const* SIXTRL_RESTRICT config_str );
}

NS(TrackJobBase)* NS(TrackJobCl_create)(
    char const* SIXTRL_RESTRICT device_id_str, 
    char const* SIXTRL_RESTRICT config_str );

NS(TrackJobBase)* NS(TrackJobCl_create_detailed)(
    char const* SIXTRL_RESTRICT device_id_str, 
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(buffer_size_t) const num_particle_sets,
    ::NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
    ::NS(Buffer)* SIXTRL_RESTRICT belemements_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    ::NS(buffer_size_t) const dump_elem_by_elem_turns,
    char const* SIXTRL_RESTRICT config_str );

#endif /* defined( __cplusplus ) */

#endif /* SIXTRL_SIXTRACKLIB_OPENCL_MAKE_TRACK_JOB_H__ */

/* end: sixtracklib/opencl/make_track_job.h */
