#ifndef SIXTRACKLIB_TESTLIB_COMMON_TRACK_TRACK_PARTICLES_CTRL_ARG_HELPER_H__
#define SIXTRACKLIB_TESTLIB_COMMON_TRACK_TRACK_PARTICLES_CTRL_ARG_HELPER_H__

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/particles/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/argument_base.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN
NS(arch_status_t) NS(TestTrackCtrlArg_prepare_ctrl_arg_tracking)(
    NS(ArgumentBase)* SIXTRL_RESTRICT particles_arg,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(ArgumentBase)* SIXTRL_RESTRICT beam_elements_arg,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(ArgumentBase)* SIXTRL_RESTRICT result_arg );

SIXTRL_EXTERN SIXTRL_HOST_FN
NS(arch_status_t) NS(TestTrackCtrlArg_prepare_ctrl_arg_elem_by_elem_tracking)(
    NS(ArgumentBase)* SIXTRL_RESTRICT particles_arg,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(ArgumentBase)* SIXTRL_RESTRICT beam_elements_arg,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(ArgumentBase)* SIXTRL_RESTRICT output_arg,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(ArgumentBase)* SIXTRL_RESTRICT elem_by_elem_conf_arg,
    NS(ElemByElemConfig)* SIXTRL_RESTRICT elem_by_elem_config,
    NS(ArgumentBase)* SIXTRL_RESTRICT result_arg );


SIXTRL_EXTERN SIXTRL_HOST_FN
NS(arch_status_t) NS(TestTrackCtrlArg_evaulate_ctrl_arg_tracking)(
    NS(ArgumentBase)* SIXTRL_RESTRICT particles_arg,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    const NS(Buffer) *const SIXTRL_RESTRICT cmp_particles_buffer,
    NS(particle_real_t) const abs_tolerance,
    NS(ArgumentBase)* SIXTRL_RESTRICT result_arg );

SIXTRL_EXTERN SIXTRL_HOST_FN
NS(arch_status_t) NS(TestTrackCtrlArg_evaluate_ctrl_arg_tracking_all)(
    NS(ArgumentBase)* SIXTRL_RESTRICT particles_arg,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    const NS(Buffer) *const SIXTRL_RESTRICT cmp_particles_buffer,
    NS(particle_real_t) const abs_tolerance,
    NS(ArgumentBase)* SIXTRL_RESTRICT result_arg );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_TESTLIB_COMMON_TRACK_TRACK_PARTICLES_CTRL_ARG_HELPER_H__ */
/* end: tests/sixtracklib/testlib/common/track/track_particles_cpu.h */
