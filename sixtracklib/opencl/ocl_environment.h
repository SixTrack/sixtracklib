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
    cl_mem             blocks_buffer;
    cl_mem             elem_by_elem_buffer;
    cl_mem             turn_by_turn_buffer;    
        
    uint64_t           ressources_flags;
    
    struct NS(OpenCLEnvNodeDevice)* nodes;
    struct NS(OpenCLEnvNodeDevice)* default_node;
    
    size_t             num_devices;
    
    char*              current_id_str;
    char*              current_kernel_function;
    char*              kernel_source;
    
    bool               is_ready;
}
NS(OpenCLEnv);

struct NS(Particles);
struct NS(Drift);
struct NS(BeamElementInfo);
struct NS(Drift);

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
    const struct NS(Particles) *const SIXTRL_RESTRICT particles,
    const struct NS(BeamElementInfo) *const SIXTRL_RESTRICT beam_elements, 
    SIXTRL_SIZE_T const num_beam_elements );

bool NS(OpenCLEnv_track_drift)( 
    NS(OpenCLEnv)* SIXTRL_RESTRICT ocl_env, 
    struct NS(Particles)* SIXTRL_RESTRICT particles, 
    const struct NS(Drift) *const SIXTRL_RESTRICT drift );
   
char** NS(OpenCLEnv_build_kernel_files_list)(
    char* filenames, SIXTRL_SIZE_T* ptr_num_of_files, 
    char const* prefix, char const* separator );

void NS(OpenCLEnv_free_kernel_files_list)( char** paths_to_kernel_files );

#endif /* SIXTRACKLIB_OPENCL_CL_ENVIRONMENT_H__ */

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

/* end: sixtracklib/opencl/ocl_environment.h */
