#ifndef SIXTRACKLIB_TESTLIB_COMMON_OUTPUT_ASSIGN_ELEM_BY_ELEM_CTRL_ARG_C99_H__
#define SIXTRACKLIB_TESTLIB_COMMON_OUTPUT_ASSIGN_ELEM_BY_ELEM_CTRL_ARG_C99_H__

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/particles/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/argument_base.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"
#include "sixtracklib/common/output/output_buffer.h"

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TestBeamMonitorCtrlArg_prepare_assign_elem_by_elem_output_buffer)(
    NS(ArgumentBase)* SIXTRL_RESTRICT elem_by_elem_config_arg,
    NS(ElemByElemConfig)* SIXTRL_RESTRICT elem_by_elem_config,
    NS(ArgumentBase)* SIXTRL_RESTRICT output_arg,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const output_buffer_index_offset,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    NS(ArgumentBase)* SIXTRL_RESTRICT result_arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TestBeamMonitorCtrlArg_evaluate_assign_elem_by_elem_output_buffer)(
    NS(ArgumentBase)* SIXTRL_RESTRICT elem_by_elem_config_arg,
    NS(ElemByElemConfi)* SIXTRL_RESTRICT elem_by_elem_config,
    NS(ArgumentBase)* SIXTRL_RESTRICT output_arg,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    const NS(Buffer) *const SIXTRL_RESTRICT cmp_output_buffer,
    NS(buffer_size_t) const output_buffer_index_offset,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    bool const compare_buffer_content,
    NS(particle_real_t) const abs_tolerance,
    NS(ArgumentBase)* SIXTRL_RESTRICT result_arg );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /*SIXTRACKLIB_TESTLIB_COMMON_OUTPUT_ASSIGN_ELEM_BY_ELEM_CTRL_ARG_C99_H__*/

/* tests/sixtracklib/testlib/common/output/assign_be_monitor_ctrl_arg.h */
