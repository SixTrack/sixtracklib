#ifndef SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_H__
#define SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_H__

#include "sixtracklib/common/generated/namespace.h"
#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/modules.h"

#include "sixtracklib/testlib/common/generic_buffer_obj.h"
#include "sixtracklib/testlib/common/gpu_kernel.h"
#include "sixtracklib/testlib/common/random.h"
#include "sixtracklib/testlib/common/time.h"
#include "sixtracklib/testlib/common/particles/particles.h"
#include "sixtracklib/testlib/common/particles/particles_addr.h"

#include "sixtracklib/testlib/common/output/assign_be_monitor_ctrl_arg.h"
#include "sixtracklib/testlib/common/output/assign_elem_by_elem_ctrl_arg.h"

#include "sixtracklib/testlib/common/track/track_job_setup.h"
#include "sixtracklib/testlib/common/track/track_particles_cpu.h"
#include "sixtracklib/testlib/common/track/track_particles_ctrl_arg.h"

#include "sixtracklib/testlib/common/buffer.h"
#include "sixtracklib/testlib/common/beam_elements/beam_elements.h"
#include "sixtracklib/testlib/common/path.h"
#include "sixtracklib/testlib/testdata/track_testdata.h"
#include "sixtracklib/testlib/testdata/testdata_files.h"


#if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
           ( SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1 )

#endif /* OpenCL */


#if defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) && \
           ( SIXTRACKLIB_ENABLE_MODULE_CUDA == 1 )

#include "sixtracklib/testlib/cuda/cuda_beam_elements_kernel_c_wrapper.h"
#include "sixtracklib/testlib/cuda/cuda_particles_kernel_c_wrapper.h"

#endif /* Cuda */

/* ------------------------------------------------------------------------- */

#endif /* SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_H__ */

/* end: tests/sixtracklib/testlib.h */
