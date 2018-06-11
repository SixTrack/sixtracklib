#ifndef SIXTRACKLIB_OPENCL_TESTS_TEST_OPENCL_TOOLS_H__
#define SIXTRACKLIB_OPENCL_TESTS_TEST_OPENCL_TOOLS_H__

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/blocks.h"

struct NS(OpenCLEnv);

void handle_cmd_line_arguments( int argc, char* argv[], 
    const struct NS(OpenCLEnv) *const ocl_env, char device_id_str[], 
    NS(block_num_elements_t)* ptr_num_particles, 
    NS(block_num_elements_t)* ptr_num_elements, 
    NS(block_size_t)* ptr_num_turns );

#endif /* SIXTRACKLIB_OPENCL_TESTS_TEST_OPENCL_TOOLS_H__ */

/* end: sixtracklib/opencl/tests/test_opencl_tools.h */
