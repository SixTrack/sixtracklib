#ifndef SIXTRACKLIB_OPENCL_CL_ENVIRONMENT_H__
#define SIXTRACKLIB_OPENCL_CL_ENVIRONMENT_H__

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif 
   
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
    
#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/impl/particles_impl.h"
#include "sixtracklib/common/beam_elements.h"

struct NS(OpenCLEnv);
struct NS(OpenCLEnvNodeDevice);
    
typedef struct NS(OpenCLEnvNodeDevice)
{
    char id_str[ 16 ];
    
    char*  platform_name;
    char*  device_name;
    char*  extensions;
    
    SIXTRL_SIZE_T env_platform_index;
    SIXTRL_SIZE_T env_device_index;
    
    struct NS(OpenCLEnvNodeDevice)* ptr_next;
    struct NS(OpenCLEnv)*           ptr_environment;
}
NS(OpenCLEnvNodeDevice);

struct NS(OpenCLEnvNodeDevice)* NS(OpenCLEnvNodeDevice_preset)(
    struct NS(OpenCLEnvNodeDevice)* nodes );

struct NS(OpenCLEnvNodeDevice)* 
    NS(OpenCLEnvNodeDevice_init)( SIXTRL_SIZE_T const num_of_devices );

struct NS(OpenCLEnvNodeDevice)* NS(OpenCLEnvNodeDevice_get_next)(
    struct NS(OpenCLEnvNodeDevice)* current_node );

void NS(OpenCLEnvNodeDevice_free)( 
    NS(OpenCLEnvNodeDevice)* SIXTRL_RESTRICT p );
    
typedef struct NS(OpenCLEnv)
{
    cl_platform_id*    platforms;
    size_t             num_platforms;
    
    cl_device_id*      devices;
        
    cl_context         context;
    cl_program         program;
    cl_kernel          kernel;
    cl_command_queue   queue;
    cl_mem             particles_buffer;
    cl_mem             beam_elem_info_buffer;
    cl_mem             beam_elem_data_buffer;
    cl_mem             elem_by_elem_buffer;
    cl_mem             turn_by_turn_buffer;    
        
    uint64_t           ressources_flags;
    
    struct NS(OpenCLEnvNodeDevice)* nodes;
    struct NS(OpenCLEnvNodeDevice)* default_node;
    struct NS(OpenCLEnvNodeDevice)* selected_nodes;
    
    size_t             num_selected_nodes;
    size_t             num_devices;
    
    SIXTRL_UINT64_T    num_beam_elements;
    SIXTRL_UINT64_T    num_particles;
    SIXTRL_UINT64_T    num_turns;
    
    char*              current_id_str;
    char*              current_kernel_function;
    char*              kernel_source;
    
    bool               is_ready;
}
NS(OpenCLEnv);

NS(OpenCLEnv)* NS(OpenCLEnv_init)();

NS(OpenCLEnvNodeDevice) const* NS(OpenCLEnv_get_node_devices)(
    const NS(OpenCLEnv) *const SIXTRL_RESTRICT ocl_env );

size_t NS(OpenCLEnv_get_num_node_devices)( 
    const NS(OpenCLEnv) *const SIXTRL_RESTRICT ocl_env );

void NS(OpenCLEnv_free)( NS(OpenCLEnv)* ocl_env );

const char *const NS(OpenCLEnv_get_current_kernel_function)( 
    const NS(OpenCLEnv) *const SIXTRL_RESTRICT ocl_env );

bool NS(OpenCLEnv_is_ready)( 
    const NS(OpenCLEnv) *const SIXTRL_RESTRICT ocl_env );

bool NS(OpenCLEnv_prepare)( struct NS(OpenCLEnv)* ocl_env, 
    char const* node_device_id, char const* kernel_function_name, 
    char* kernel_source_files, char const* compile_options,
    SIXTRL_SIZE_T const num_turns,
    NS(Particles) const* SIXTRL_RESTRICT particles,
    NS(BeamElements) const* SIXTRL_RESTRICT beam_elements );

bool NS(OpenCLEnv_track_particles)( 
    struct NS(OpenCLEnv)* ocl_env, 
    NS(Particles)*   SIXTRL_RESTRICT particles, 
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements );
   
                       

#endif /* SIXTRACKLIB_OPENCL_CL_ENVIRONMENT_H__ */

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

/* end: sixtracklib/opencl/ocl_environment.h */
