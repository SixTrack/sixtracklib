#ifndef SIXTRL_SIXTRACKLIB_OPENCL_TRACK_H__
#define SIXTRL_SIXTRACKLIB_OPENCL_TRACK_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#if !defined( _GPUCODE )

struct NS(Buffer);

SIXTRL_HOST_FN struct NS(Buffer)* NS(TrackCL)(
    char const* SIXTRL_RESTRICT device_id_str,
    struct NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    struct NS(Buffer)* SIXTRL_RESTRICT beam_elements,
    struct NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    int const until_turn, int const elem_by_elem_turns );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRL_SIXTRACKLIB_OPENCL_TRACK_H__*/

/* end: sixtracklib/opencl/track.h */
