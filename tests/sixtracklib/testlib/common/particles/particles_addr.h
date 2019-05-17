#ifndef SIXTRACK_TESTLIB_COMMON_PARTICLES_PARTICLES_ADDR_CXX_HPP__
#define SIXTRACK_TESTLIB_COMMON_PARTICLES_PARTICLES_ADDR_CXX_HPP__

#include <stddef.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/buffer.h"

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++ */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t
NS(TestParticlesAddr_prepare_buffers)(
    NS(Buffer)* SIXTRL_RESTRICT paddr_buffer,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_elements,
    NS(buffer_size_t) const min_num_particles,
    NS(buffer_size_t) const max_num_particles,
    double const probablity_for_non_particles );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TestParticlesAddr_verify_structure)(
    const NS(Buffer) *const SIXTRL_RESTRICT paddr_buffer,
    const NS(Buffer) *const SIXTRL_RESTRICT particles_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TestParticlesAddr_verify_addresses)(
    const NS(Buffer) *const SIXTRL_RESTRICT paddr_buffer,
    const NS(Buffer) *const SIXTRL_RESTRICT particles_buffer );

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++ */

#endif /* SIXTRACK_TESTLIB_COMMON_PARTICLES_PARTICLES_ADDR_CXX_HPP__ */

/* end: tests/sixtracklib/testlib/common/particles/particles_addr.hpp */
