#ifndef SIXTRACKLIB_SIXTRACKLIB_H__
#define SIXTRACKLIB_SIXTRACKLIB_H__

#if !defined( __NAMESPACE )
    #define __NAMESPACE st_
    #define __SIXTRACKLIB_SIXTRACKLIB_UNDEF_NAMESPACE 1
#endif /* !defined( __NAMESPACE ) */

/* ------------------------------------------------------------------------- */

#include "sixtracklib/common/generated/namespace.h"
#include "sixtracklib/common/generated/config.h"
#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/generated/modules.h"
#include "sixtracklib/common/buffer/mem_pool.h"
#include "sixtracklib/common/buffer/managed_buffer_minimal.h"
#include "sixtracklib/common/buffer/managed_buffer_remap.h"
#include "sixtracklib/common/buffer/managed_buffer.h"
#include "sixtracklib/common/buffer/buffer_type.h"
#include "sixtracklib/common/buffer/buffer_object.h"
#include "sixtracklib/common/buffer/buffer_garbage.h"
#include "sixtracklib/common/buffer/buffer_generic.h"
#include "sixtracklib/common/be_beambeam/be_beambeam4d.h"
#include "sixtracklib/common/be_beambeam/be_beambeam6d.h"
#include "sixtracklib/common/be_beambeam/track.h"
#include "sixtracklib/common/be_beambeam/faddeeva_cern.h"
#include "sixtracklib/common/be_beambeam/gauss_fields.h"
#include "sixtracklib/common/be_cavity/be_cavity.h"
#include "sixtracklib/common/be_cavity/track.h"
#include "sixtracklib/common/be_drift/be_drift.h"
#include "sixtracklib/common/be_drift/track.h"
#include "sixtracklib/common/be_monitor/be_monitor.h"
#include "sixtracklib/common/be_monitor/output_buffer.h"
#include "sixtracklib/common/be_monitor/track.h"
#include "sixtracklib/common/be_multipole/be_multipole.h"
#include "sixtracklib/common/be_multipole/track.h"
#include "sixtracklib/common/be_srotation/be_srotation.h"
#include "sixtracklib/common/be_srotation/track.h"
#include "sixtracklib/common/be_xyshift/be_xyshift.h"
#include "sixtracklib/common/be_xyshift/track.h"
#include "sixtracklib/common/context/definitions.h"
#include "sixtracklib/common/context/argument_base.h"
#include "sixtracklib/common/context/compute_arch.h"
#include "sixtracklib/common/context/context_base.h"
#include "sixtracklib/common/context/context_base_with_nodes.h"
#include "sixtracklib/common/context.h"
#include "sixtracklib/common/internal/track_job_base.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"
#include "sixtracklib/common/output/elem_by_elem_output_buffer.h"
#include "sixtracklib/common/output/output_buffer.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/constants.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/track_job.h"
#include "sixtracklib/common/track_job_cpu.h"
#include "sixtracklib/common/track.h"


/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
           ( SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1 )

    #include "sixtracklib/opencl/cl.h"
    #include "sixtracklib/opencl/argument.h"
    #include "sixtracklib/opencl/context.h"
    #include "sixtracklib/opencl/track_job_cl.h"

#endif /* defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) && \
           ( SIXTRACKLIB_ENABLE_MODULE_CUDA == 1 )

    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/internal/argument_base.h"
    #include "sixtracklib/cuda/argument.h"
    #include "sixtracklib/cuda/internal/context_base.h"
    #include "sixtracklib/cuda/context.h"
    #include "sixtracklib/cuda/track_particles_kernel_c_wrapper.h"

#endif /* defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) */


/* ------------------------------------------------------------------------- */

#endif /* SIXTRACKLIB_SIXTRACKLIB_H__ */

/* end: sixtracklib/sixtracklib.h */
