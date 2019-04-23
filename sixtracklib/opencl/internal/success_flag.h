#ifndef SIXTRACKLIB_OPENCL_INTERNAL_SUCCESS_FLAG_H__
#define SIXTRACKLIB_OPENCL_INTERNAL_SUCCESS_FLAG_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

typedef SIXTRL_INT32_T NS(opencl_success_flag_t);

#if defined( __OPENCL_VERSION__ )
    #if __OPENCL_VERSION__ == CL_VERSION_1_0 && \
        defined( cl_khr_int32_extended_atomics )
        #pragma OPENCL EXTENSION cl_khr_int32_extended_atomics : enable
    #endif /* OpenCL 1.x cl_khr_int32_extended_atomics available */

    SIXTRL_STATIC SIXTRL_DEVICE_FN
    void NS(OpenCl1x_collect_success_flag_value)( SIXTRL_BUFFER_DATAPTR_DEC
        NS(opencl_success_flag_t)* SIXTRL_RESTRICT ptr_success_flag,
        NS(opencl_success_flag_t) const local_success_flag );

    /* --------------------------------------------------------------------- */

    SIXTRL_INLINE void NS(OpenCl1x_collect_success_flag_value)(
        SIXTRL_BUFFER_DATAPTR_DEC NS(opencl_success_flag_t)*
            SIXTRL_RESTRICT ptr_success_flag,
        NS(opencl_success_flag_t) const local_success_flag )
    {
        if( ( ptr_success_flag != SIXTRL_NULLPTR ) &&
            ( local_success_flag > ( NS(opencl_success_flag_t) )0u ) )
        {
            #if  __OPENCL_VERSION__ == CL_VERSION_1_0 && \
                defined( cl_khr_int32_extended_atomics )

                atom_or( ptr_success_flag, local_success_flag );

            #elif __OPENCL_VERSION__ > CL_VERSION_1_0
                /* Above OpenCL 1.0, atomic_or should always be available */
                atomic_or( ptr_success_flag, local_success_flag );

            #endif /* !defined( cl_khr_int32_extended_atomics ) */
        }

        return;
    }
#endif /* defined( __OPENCL_VERSION__ ) */

#endif /* SIXTRACKLIB_OPENCL_INTERNAL_SUCCESS_FLAG_H__ */
/* end: sixtracklib/opencl/internal/success_flag.h */
