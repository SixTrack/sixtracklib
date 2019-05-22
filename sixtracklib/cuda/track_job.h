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

SIXTRL_EXTERN SIXTRL_HOST_FN cuda_controller_t& cudaController();
SIXTRL_EXTERN SIXTRL_HOST_FN cuda_controller_t const& cudaController() const;

SIXTRL_EXTERN SIXTRL_HOST_FN cuda_controller_t* ptrCudaController() SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN cuda_controller_t const*
ptrCudaController() const SIXTRL_NOEXCEPT;

/* ----------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool hasCudaParticlesArg() const SIXTRL_NOEXCEPT;
SIXTRL_EXTERN SIXTRL_HOST_FN cuda_argument_t& cudaParticlesArg();
SIXTRL_EXTERN SIXTRL_HOST_FN cuda_argument_t const& cudaParticlesArg() const;

SIXTRL_EXTERN SIXTRL_HOST_FN cuda_argument_t*
ptrCudaParticlesArg() SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN cuda_argument_t const*
ptrCudaParticlesArg() const SIXTRL_NOEXCEPT;

/* ----------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool hasCudaBeamElementsArg() const SIXTRL_NOEXCEPT;
SIXTRL_EXTERN SIXTRL_HOST_FN cuda_argument_t& cudaBeamElementsArg();
SIXTRL_EXTERN SIXTRL_HOST_FN cuda_argument_t const& cudaBeamElementsArg() const;

SIXTRL_EXTERN SIXTRL_HOST_FN cuda_argument_t const*
ptrCudaBeamElementsArg() const SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN cuda_argument_t*
ptrCudaBeamElementsArg() SIXTRL_NOEXCEPT;

/* ----------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool hasCudaOutputArg() const SIXTRL_NOEXCEPT;
SIXTRL_EXTERN SIXTRL_HOST_FN cuda_argument_t& cudaOutputArg();
SIXTRL_EXTERN SIXTRL_HOST_FN cuda_argument_t const& cudaOutputArg() const;

SIXTRL_EXTERN SIXTRL_HOST_FN cuda_argument_t* ptrCudaOutputArg() SIXTRL_NOEXCEPT;
SIXTRL_EXTERN SIXTRL_HOST_FN cuda_argument_t const*
ptrCudaOutputArg() const SIXTRL_NOEXCEPT;

/* ----------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool hasCudaElemByElemConfigArg() const SIXTRL_NOEXCEPT;
SIXTRL_EXTERN SIXTRL_HOST_FN cuda_argument_t& cudaElemByElemConfigArg();
SIXTRL_EXTERN SIXTRL_HOST_FN cuda_argument_t const& cudaElemByElemConfigArg() const;

SIXTRL_EXTERN SIXTRL_HOST_FN cuda_argument_t const*
ptrCudaElemByElemConfigArg() const SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN cuda_argument_t*
ptrCudaElemByElemConfigArg() SIXTRL_NOEXCEPT;

/* ----------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool hasCudaDebugRegisterArg() const SIXTRL_NOEXCEPT;
SIXTRL_EXTERN SIXTRL_HOST_FN cuda_argument_t& cudaDebugRegisterArg();
SIXTRL_EXTERN SIXTRL_HOST_FN cuda_argument_t const& cudaDebugRegisterArg() const;

SIXTRL_EXTERN SIXTRL_HOST_FN cuda_argument_t const*
ptrCudaDebugRegisterArg() const SIXTRL_NOEXCEPT;

SIXTRL_HOST_FN cuda_argument_t*
ptrCudaDebugRegisterArg() SIXTRL_NOEXCEPT;

/* ----------------------------------------------------------------- */

SIXTRL_HOST_FN bool hasCudaParticlesAddrArg() const SIXTRL_NOEXCEPT;
SIXTRL_HOST_FN cuda_argument_t const& cudaParticlesAddrArg() const;
SIXTRL_HOST_FN cuda_argument_t& cudaParticlesAddrArg();

SIXTRL_HOST_FN cuda_argument_t const*
ptrCudaParticlesAddrArg() const SIXTRL_NOEXCEPT;

SIXTRL_HOST_FN cuda_argument_t*
ptrCudaParticlesAddrArg() SIXTRL_NOEXCEPT;


#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_TRACK_JOB_H__ */

/* end: sixtracklib/cuda/track_job.h */
