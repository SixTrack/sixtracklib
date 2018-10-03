#ifndef SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_TEST_GPU_KERNELTOOLS_H__
#define SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_TEST_GPU_KERNELTOOLS_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_SIZE_T NS(File_get_size)(
    const char *const path_to_file );

char* NS(File_read_into_string)( const char *const path_to_file,
    char* buffer_begin, SIXTRL_SIZE_T const max_num_chars );

bool NS(File_exists)( const char *const path_to_file );

/* ------------------------------------------------------------------------- */

char** NS(GpuKernel_create_file_list)( char* SIXTRL_RESTRICT filenames,
    SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_num_of_files,
    char const* SIXTRL_RESTRICT prefix, char const* SIXTRL_RESTRICT separator );

void NS(GpuKernel_free_file_list)(
    char** SIXTRL_RESTRICT paths_to_kernel_files,
    SIXTRL_SIZE_T const num_of_files );

char* NS(GpuKernel_collect_source_string)(
    char** file_list, SIXTRL_SIZE_T const num_of_files,
    SIXTRL_SIZE_T const max_line_length, SIXTRL_SIZE_T* SIXTRL_RESTRICT lines_offset );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_TEST_GPU_KERNELTOOLS_H__ */

/* end: tests/sixtracklib/testlib/gpu_kernel_tools.h */
