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
#include "sixtracklib/common/be_beamfields/be_beamfields.h"
#include "sixtracklib/common/be_beamfields/track.h"
#include "sixtracklib/common/be_beamfields/faddeeva_cern.h"
#include "sixtracklib/common/be_beamfields/gauss_fields.h"
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
#include "sixtracklib/common/be_limit/be_limit_rect.h"
#include "sixtracklib/common/be_limit/be_limit_ellipse.h"
#include "sixtracklib/common/be_limit/be_limit_rect_ellipse.h"
#include "sixtracklib/common/be_limit/track.h"
#include "sixtracklib/common/be_dipedge/be_dipedge.h"
#include "sixtracklib/common/be_dipedge/track.h"
#include "sixtracklib/common/be_rfmultipole/be_rfmultipole.h"
#include "sixtracklib/common/be_rfmultipole/track.h"
#include "sixtracklib/common/context/compute_arch.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/argument_base.h"
#include "sixtracklib/common/control/controller_base.h"
#include "sixtracklib/common/control/node_controller_base.h"
#include "sixtracklib/common/controller.h"
#include "sixtracklib/common/internal/track_job_base.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"
#include "sixtracklib/common/output/elem_by_elem_output_buffer.h"
#include "sixtracklib/common/output/output_buffer.h"
#include "sixtracklib/common/track/definitions.h"
#include "sixtracklib/common/track/track.h"
#include "sixtracklib/common/track/track_kernel_impl.h"
#include "sixtracklib/common/track/track_kernel_opt_impl.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/constants.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/track_job.h"
#include "sixtracklib/common/track_job_cpu.h"

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
    #include "sixtracklib/cuda/control/kernel_config.h"
    #include "sixtracklib/cuda/argument.h"
    #include "sixtracklib/cuda/controller.h"
    #include "sixtracklib/cuda/track_particles_kernel_c_wrapper.h"

#endif /* defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) */


/* ------------------------------------------------------------------------- */

#endif /* SIXTRACKLIB_SIXTRACKLIB_H__ */

/* end: sixtracklib/sixtracklib.h */
