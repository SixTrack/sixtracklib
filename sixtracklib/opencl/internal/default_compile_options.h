#ifndef SIXTRACKLIB_OPENCL_INTERNAL_DEFAULT_COMPILE_OPTIONS_H__
#define SIXTRACKLIB_OPENCL_INTERNAL_DEFAULT_COMPILE_OPTIONS_H__

    #if defined( _GPUCODE ) && defined( __OPENCL_VERSION__ )

        #if !defined( SIXTRL_ARGPTR_DEC )
            #define SIXTRL_ARGPTR_DEC __private
        #endif /* !defined( SIXTRL_ARGPTR_DEC ) */

        #if !defined( SIXTRL_DATAPTR_DEC )
            #define SIXTRL_DATAPTR_DEC __global
        #endif /* !defined( SIXTRL_DATAPTR_DEC ) */

        #if !defined( SIXTRL_BUFFER_ARGPTR_DEC )
            #define SIXTRL_BUFFER_ARGPTR_DEC __global
        #endif /* SIXTRL_BUFFER_ARGPTR_DEC */

        #if !defined( SIXTRL_BUFFER_DATAPTR_DEC )
            #define SIXTRL_BUFFER_DATAPTR_DEC __global
        #endif /* SIXTRL_BUFFER_DATAPTR_DEC */

        #if !defined( DSIXTRL_BUFFER_OBJ_ARGPTR_DEC )
            #define DSIXTRL_BUFFER_OBJ_ARGPTR_DEC __global
        #endif /* !defined( DSIXTRL_BUFFER_OBJ_ARGPTR_DEC ) */

        #if !defined( DSIXTRL_BUFFER_OBJ_DATAPTR_DEC )
            #define DSIXTRL_BUFFER_OBJ_DATAPTR_DEC __global
        #endif /* !defined( DSIXTRL_BUFFER_OBJ_DATAPTR_DEC ) */

        #if !defined( SIXTRL_BE_ARGPTR_DEC )
            #define SIXTRL_BE_ARGPTR_DEC __global
        #endif /* !defined( SIXTRL_BE_ARGPTR_DEC ) */

        #if !defined( SIXTRL_BE_DATAPTR_DEC )
            #define SIXTRL_BE_DATAPTR_DEC __global
        #endif /* !defined( SIXTRL_BE_DATAPTR_DEC ) */

        #if !defined( SIXTRL_PARTICLE_ARGPTR_DEC )
            #define SIXTRL_PARTICLE_ARGPTR_DEC __global
        #endif /* !defined( SIXTRL_PARTICLE_ARGPTR_DEC ) */

        #if !defined( SIXTRL_PARTICLE_DATAPTR_DEC )
            #define SIXTRL_PARTICLE_DATAPTR_DEC __global
        #endif /* !defined( SIXTRL_PARTICLE_DATAPTR_DEC ) */

    #endif /* defined( _GPUCODE ) && defined( __OPENCL_VERSION__ ) */

#endif /* SIXTRACKLIB_OPENCL_INTERNAL_DEFAULT_COMPILE_OPTIONS_H__ */

/* end: sixtracklib/opencl/internal/default_compile_options.h */
