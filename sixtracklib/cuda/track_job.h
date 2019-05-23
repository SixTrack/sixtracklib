#ifndef SIXTRACKLIB_CUDA_TRACK_JOB_H__
#define SIXTRACKLIB_CUDA_TRACK_JOB_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/track/track_job_base.h"
    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/controller.h"
    #include "sixtracklib/cuda/argument.h"
    #include "sixtracklib/cuda/track_job.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, Host */

#if !defined( _GPUCODE )

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaTrackJob)* NS(CudaTrackJob_create)( void );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaTrackJob)*
NS(CudaTrackJob_new_from_config_str)( char const* SIXTRL_RESTRICT config_str );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaTrackJob)*
NS(CudaTrackJob_new)(
    char const* SIXTRL_RESTRICT node_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaTrackJob)*
NS(CudaTrackJob_new_with_output)(
    char const* SIXTRL_RESTRICT node_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaTrackJob)*
NS(CudaTrackJob_new_detailed)(
    char const* SIXTRL_RESTRICT node_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_psets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    char const* SIXTRL_RESTRICT config_str );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(CudaTrackJob_has_controller)(
    const NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaController)* 
NS(CudaTrackJob_get_ptr_controller)(
    NS(CudaTrackJob)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaController) const* 
NS(CudaTrackJob_get_ptr_const_controller)(
    const NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job );
    

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(CudaTrackJob_has_particles_arg)(
    const NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job );
    
SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaArgument)*
NS(CudaTrackJob_get_ptr_particles_arg)(
    NS(CudaTrackJob)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaArgument) const*
NS(CudaTrackJob_get_ptr_const_particles_arg)(
    const NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(CudaTrackJob_has_beam_elements_arg)(
    const NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job );
    
SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaArgument)*
NS(CudaTrackJob_get_ptr_beam_elements_arg)(
    NS(CudaTrackJob)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaArgument) const*
NS(CudaTrackJob_get_ptr_const_beam_elements_arg)(
    const NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(CudaTrackJob_has_output_arg)(
    const NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job );
    
SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaArgument)*
NS(CudaTrackJob_get_ptr_output_arg)(
    NS(CudaTrackJob)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaArgument) const*
NS(CudaTrackJob_get_ptr_const_output_arg)(
    const NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(CudaTrackJob_has_elem_by_elem_config_arg)(
    const NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job );
    
SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaArgument)*
NS(CudaTrackJob_get_ptr_elem_by_elem_config_arg)(
    NS(CudaTrackJob)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaArgument) const*
NS(CudaTrackJob_get_ptr_const_elem_by_elem_config_arg)(
    const NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(CudaTrackJob_has_debug_register_arg)(
    const NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job );
    
SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaArgument)*
NS(CudaTrackJob_get_ptr_debug_register_arg)(
    NS(CudaTrackJob)* SIXTRL_RESTRICT track_job );
    
SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaArgument) const*
NS(CudaTrackJob_get_ptr_const_debug_register_arg)(
    const NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(CudaTrackJob_has_particles_addr_arg)(
    const NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job );
    
SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaArgument)*
NS(CudaTrackJob_get_ptr_particles_addr_arg)(
    NS(CudaTrackJob)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaArgument) const*
NS(CudaTrackJob_get_ptr_const_particles_addr_arg)(
    const NS(CudaTrackJob) *const SIXTRL_RESTRICT track_job );

/* ------------------------------------------------------------------------- */

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_TRACK_JOB_H__ */

/* end: sixtracklib/cuda/track_job.h */
