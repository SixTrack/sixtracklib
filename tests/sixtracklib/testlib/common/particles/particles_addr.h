#ifndef SIXTRACK_TESTLIB_COMMON_PARTICLES_PARTICLES_ADDR_C99_H__
#define SIXTRACK_TESTLIB_COMMON_PARTICLES_PARTICLES_ADDR_C99_H__

#include <stddef.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/controller_base.h"
#include "sixtracklib/common/control/argument_base.h"

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++ */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TestParticlesAddr_prepare_buffers)(
    NS(Buffer)* SIXTRL_RESTRICT paddr_buffer,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_elements,
    NS(buffer_size_t) const min_num_particles,
    NS(buffer_size_t) const max_num_particles,
    double const probablity_for_non_particles,
    NS(buffer_size_t) const initial_seed_value );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TestParticlesAddr_verify_structure)(
    const NS(Buffer) *const SIXTRL_RESTRICT paddr_buffer,
    const NS(Buffer) *const SIXTRL_RESTRICT particles_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TestParticlesAddr_verify_addresses)(
    const NS(Buffer) *const SIXTRL_RESTRICT paddr_buffer,
    const NS(Buffer) *const SIXTRL_RESTRICT particles_buffer );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TestParticlesAddr_prepare_ctrl_args_test)(
    NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    NS(ArgumentBase)* SIXTRL_RESTRICT paddr_arg,
    NS(Buffer)* SIXTRL_RESTRICT paddr_buffer,
    NS(ArgumentBase)* SIXTRL_RESTRICT particles_arg,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(ArgumentBase)* SIXTRL_RESTRICT result_arg );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TestParticlesAddr_evaluate_ctrl_args_test)(
    NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    NS(ArgumentBase)* SIXTRL_RESTRICT paddr_arg,
    NS(Buffer)* SIXTRL_RESTRICT paddr_buffer,
    NS(ArgumentBase)* SIXTRL_RESTRICT particles_arg,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(ArgumentBase)* SIXTRL_RESTRICT result_arg );

/* ------------------------------------------------------------------------- */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++ */

#endif /* SIXTRACK_TESTLIB_COMMON_PARTICLES_PARTICLES_ADDR_C99_H__ */

/* end: tests/sixtracklib/testlib/common/particles/particles_addr.h */
