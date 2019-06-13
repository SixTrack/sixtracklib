#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/kernels/track_particles.cuh"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>

    #include <cuda_runtime.h>
    #include <cuda.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/internal/particles_defines.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/track.h"
    #include "sixtracklib/cuda/cuda_tools.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__global__ void NS(Track_particles_line)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT belems,
    uint64_t const line_begin_idx,
    uint64_t const line_end_idx,
    bool const finish_turn )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object)*        obj_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*  obj_const_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC     NS(Particles)*     ptr_particles_t;

    buf_size_t const slot_size = ( buf_size_t )8u;

    size_t particle_index = NS(Cuda_get_1d_thread_id_in_kernel)();
    size_t const stride   = NS(Cuda_get_1d_thread_stride_in_kernel)();

    ptr_particles_t particles = NS(BufferIndex_get_particles)(
        NS(ManagedBuffer_get_objects_index_begin)( pbuffer, slot_size ) );

    size_t const nparticles = NS(Particles_get_num_of_particles)( particles );

    obj_const_iter_t line_begin =
        NS(ManagedBuffer_get_const_objects_index_begin)( belems, slot_size );

    obj_const_iter_t line_end = line_begin + line_end_idx;
    line_begin = line_begin + line_begin_idx;

    NS(Track_subset_of_particles_line)( particles, particle_index,
        nparticles, stride, line_begin, line_end, finish_turn );

    return;
}


/* end */
