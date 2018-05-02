#include "sixtracklib/opencl/ocl_environment.h"

#include <assert.h>
#include <ctype.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif 

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/namespace_begin.h"
#include "sixtracklib/_impl/path.h"
#include "sixtracklib/common/details/gpu_kernel_tools.h"
#include "sixtracklib/common/impl/block_info_impl.h"
#include "sixtracklib/common/impl/particles_impl.h"
#include "sixtracklib/common/impl/be_drift_impl.h"
#include "sixtracklib/common/beam_elements.h"

/* ------------------------------------------------------------------------- */

extern struct NS(OpenCLEnvNodeDevice)* NS(OpenCLEnvNodeDevice_preset)(
    struct NS(OpenCLEnvNodeDevice)* nodes );

extern struct NS(OpenCLEnvNodeDevice)* NS(OpenCLEnvNodeDevice_init)( 
    SIXTRL_SIZE_T const num_of_devices );

extern struct NS(OpenCLEnvNodeDevice)* NS(OpenCLEnvNodeDevice_get_next)(
    struct NS(OpenCLEnvNodeDevice)* current_node );

extern void NS(OpenCLEnvNodeDevice_free)( 
    struct NS(OpenCLEnvNodeDevice)* SIXTRL_RESTRICT p );

/* ------------------------------------------------------------------------- */

extern NS(OpenCLEnv)* NS(OpenCLEnv_init)();

extern void NS(OpenCLEnv_free)( NS(OpenCLEnv)* ocl_env );

extern NS(OpenCLEnvNodeDevice) const* NS(OpenCLEnv_get_node_devices)(
    const NS(OpenCLEnv) *const SIXTRL_RESTRICT ocl_env );

extern size_t NS(OpenCLEnv_get_num_node_devices)( 
    const NS(OpenCLEnv) *const SIXTRL_RESTRICT ocl_env );

extern const char *const NS(OpenCLEnv_get_current_kernel_function)( 
    const NS(OpenCLEnv) *const SIXTRL_RESTRICT ocl_env );

extern bool NS(OpenCLEnv_is_ready)( 
    const NS(OpenCLEnv) *const SIXTRL_RESTRICT ocl_env );

extern bool NS(OpenCLEnv_prepare)( struct NS(OpenCLEnv)* ocl_env, 
    char const* node_device_id, char const* kernel_function_name, 
    char* kernel_source_files, char const* compile_options,
    SIXTRL_SIZE_T const num_turns,
    NS(Particles) const* SIXTRL_RESTRICT particles,
    NS(BeamElements) const* SIXTRL_RESTRICT beam_elements );

extern bool NS(OpenCLEnv_track_particles)( 
    struct NS(OpenCLEnv)* ocl_env, 
    NS(Particles)*   SIXTRL_RESTRICT particles, 
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

static uint64_t const NS(HAS_PLATFORMS)              = UINT64_C( 0x00000001 );
static uint64_t const NS(HAS_DEVICES)                = UINT64_C( 0x00000002 );
static uint64_t const NS(HAS_NODES)                  = UINT64_C( 0x00000004 );
static uint64_t const NS(HAS_CONTEXT)                = UINT64_C( 0x00000008 );
static uint64_t const NS(HAS_PROGRAM)                = UINT64_C( 0x00000010 );
static uint64_t const NS(HAS_KERNEL)                 = UINT64_C( 0x00000020 );
static uint64_t const NS(HAS_QUEUE)                  = UINT64_C( 0x00000040 );
                                                     
static uint64_t const NS(HAS_PARTICLES_BUFFER)       = UINT64_C( 0x00000080 );
static uint64_t const NS(HAS_BEAM_ELEMS_BUFFERS)     = UINT64_C( 0x00000100 );
static uint64_t const NS(HAS_E_BY_E_BUFFER)          = UINT64_C( 0x00000200 );
static uint64_t const NS(HAS_T_BY_T_BUFFER)          = UINT64_C( 0x00000400 );
                                                     
static uint64_t const NS(HAS_CURRENT_ID_STR)         = UINT64_C( 0x00000800 );
static uint64_t const NS(HAS_CURRENT_KERNEL_FN)      = UINT64_C( 0x00001000 );
static uint64_t const NS(HAS_CURRENT_KERNEL_SOURCE ) = UINT64_C( 0x00002000 );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC char* NS(OpenCLEnv_get_string_attribute_from_platform_info)(
    cl_platform_id platform, cl_platform_info attribute, 
    char** ptr_to_string, SIXTRL_SIZE_T* ptr_to_string_max_size );

SIXTRL_STATIC char* NS(OpenCLEnv_get_string_attribute_from_device_info)(
    cl_device_id const device, cl_device_info const attribute, 
    char** ptr_to_string, SIXTRL_SIZE_T* string_max_size );

SIXTRL_STATIC cl_platform_id* NS(OpenCLEnv_get_valid_platforms)( 
    SIXTRL_SIZE_T* ptr_num_valid_platforms, 
    SIXTRL_SIZE_T* ptr_num_of_potentially_valid_devices );

SIXTRL_STATIC cl_device_id* NS(OpenCLEnv_get_valid_devices)( 
    cl_platform_id* platforms, SIXTRL_SIZE_T const num_platforms,
    cl_device_type const device_type, SIXTRL_SIZE_T* ptr_num_valid_devices );

SIXTRL_STATIC NS(OpenCLEnv)* NS(OpenCLEnv_preset)( NS(OpenCLEnv)* ocl_env );

SIXTRL_STATIC void NS(OpenCLEnv_reset_kernel)( NS(OpenCLEnv)* ocl_env );

/* ------------------------------------------------------------------------- */

NS(OpenCLEnvNodeDevice)* NS(OpenCLEnvNodeDevice_preset)(
    NS(OpenCLEnvNodeDevice)* nodes )
{
    if( nodes != 0 )
    {
        memset( &nodes->id_str[ 0 ], ( int )'\0', 16 );
        
        nodes->platform_name      = 0;
        nodes->device_name        = 0;
        nodes->extensions         = 0;
        
        nodes->env_platform_index = 0;
        nodes->env_device_index   = 0;
        
        nodes->ptr_next           = 0;
        nodes->ptr_environment    = 0;
    }
    
    return nodes;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(OpenCLEnvNodeDevice)* 
    NS(OpenCLEnvNodeDevice_init)( SIXTRL_SIZE_T const num_of_devices )
{
    typedef NS(OpenCLEnvNodeDevice) node_t;
    
    node_t* first_node = 0;
    
    static SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;
    
    if( num_of_devices > ZERO_SIZE )
    {
        SIXTRL_SIZE_T  ii = ZERO_SIZE;
        
        node_t* node = NS(OpenCLEnvNodeDevice_preset)( 
            ( node_t* )malloc( sizeof( node_t ) ) );
        
        first_node = node;
                
        for( ii = ( SIXTRL_SIZE_T )1u ; ii < num_of_devices ; ++ii )
        {
            if( node == 0 )
            {
                NS(OpenCLEnvNodeDevice_free)( first_node );
                first_node = 0;
                break;
            }
            
            node->ptr_next = NS(OpenCLEnvNodeDevice_preset)(
                ( node_t* )malloc( sizeof( node_t ) ) );
            
            node = node->ptr_next;
        }
    }
    
    return first_node;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(OpenCLEnvNodeDevice)* NS(OpenCLEnvNodeDevice_get_next)(
    NS(OpenCLEnvNodeDevice)* current_node )
{
    return ( current_node != 0 ) ? current_node->ptr_next : 0;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

void NS(OpenCLEnvNodeDevice_free)( 
    NS(OpenCLEnvNodeDevice)* SIXTRL_RESTRICT p )
{
    typedef NS(OpenCLEnvNodeDevice) node_t;
    
    while( ( p != 0 ) && ( p->ptr_next != 0 ) )
    {
        node_t* next = p->ptr_next;
        
        free( p->platform_name );
        free( p->device_name );
        free( p->extensions );        
        free( p );
        
        NS(OpenCLEnvNodeDevice_preset)( p );
        p = next;
    }
    
    return;
}

/* -------------------------------------------------------------------------- */

NS(OpenCLEnv)* NS(OpenCLEnv_preset)( NS(OpenCLEnv)* ocl_env )
{
    static SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;
    
    if( ocl_env != 0 )
    {
        ocl_env->platforms = 0;
        ocl_env->num_platforms = ZERO_SIZE;
        
        ocl_env->devices = 0;        
        ocl_env->ressources_flags = UINT64_C( 0 );
        
        ocl_env->nodes                      = 0;
        ocl_env->default_node               = 0;
        ocl_env->selected_nodes             = 0;
        
        ocl_env->num_selected_nodes         = ZERO_SIZE;
        ocl_env->num_devices                = ZERO_SIZE;
        
        ocl_env->num_beam_elements          = ( SIXTRL_UINT64_T )0u;
        ocl_env->num_particles              = ( SIXTRL_UINT64_T )0u;
        ocl_env->num_turns                  = ( SIXTRL_UINT64_T )0u;
        
        ocl_env->current_id_str             = 0;
        ocl_env->current_kernel_function    = 0;
        ocl_env->kernel_source              = 0;
        ocl_env->is_ready                   = false;
    }
    
    return ocl_env;
}

NS(OpenCLEnv)* NS(OpenCLEnv_init)()
{
    typedef NS(OpenCLEnv) ocl_env_t;
    
    static SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;
    bool success = false;
    
    ocl_env_t* env = NS(OpenCLEnv_preset)( 
        ( ocl_env_t* )malloc( sizeof( ocl_env_t ) ) );
    
    env->ressources_flags = UINT64_C( 0 );
    env->platforms = NS(OpenCLEnv_get_valid_platforms)( 
        &env->num_platforms, &env->num_devices );
    
    if( ( env->platforms != 0 ) && ( env->num_platforms > ZERO_SIZE ) )
    {
        env->ressources_flags |= NS(HAS_PLATFORMS);
        success = true;
    }
    
    if( success )
    {
        env->devices = NS(OpenCLEnv_get_valid_devices)( 
            env->platforms, env->num_platforms, CL_DEVICE_TYPE_ALL, 
            &env->num_devices );
        
        if( ( env->devices != 0 ) && ( env->num_devices > ZERO_SIZE ) )
        {
            env->ressources_flags |= NS(HAS_DEVICES);
        }
        else
        {
            success = false;
        }
    }
    
    if( success )
    {
        typedef NS(OpenCLEnvNodeDevice) node_t;
        node_t* next_node = 0;
        
        SIXTRL_SIZE_T ii = ( SIXTRL_SIZE_T )0u;
        
        SIXTRL_SIZE_T temp_str_len = 10240u;
        char* temp_str = ( char* )malloc( sizeof( char ) * temp_str_len );
        
        assert( ( env->platforms != 0 ) && ( env->num_platforms > ZERO_SIZE ) &&
                ( env->devices   != 0 ) && ( env->num_devices   > ZERO_SIZE ) );
        
        env->nodes = NS(OpenCLEnvNodeDevice_init)( env->num_devices );        
        next_node  = env->nodes;
        
        while( next_node != 0 )
        {
            cl_platform_id platform;
            
            cl_int ret = clGetDeviceInfo( env->devices[ ii ], 
                CL_DEVICE_PLATFORM, sizeof( platform ), &platform, 0 );
            
            success &= ( ret == CL_SUCCESS );
            
            if( env->default_node == 0 )
            {
                if( ( env->num_platforms == ( SIXTRL_SIZE_T )1u ) &&
                    ( env->num_devices   == ( SIXTRL_SIZE_T )1u ) )
                {
                    env->default_node = next_node;
                }
                else
                {
                    cl_device_type device_type;
                    
                    ret = clGetDeviceInfo( env->devices[ ii ], CL_DEVICE_TYPE, 
                        sizeof( device_type ), &device_type, 0 );
                
                    if( ret == CL_SUCCESS )
                    {
                        if( ( device_type & CL_DEVICE_TYPE_DEFAULT ) == 
                            CL_DEVICE_TYPE_DEFAULT )
                        {
                            env->default_node = next_node;
                        }
                    }
                    else
                    {
                        success = false;
                    }
                }
            }
            
            next_node->platform_name = 
                NS(OpenCLEnv_get_string_attribute_from_platform_info)(
                    platform, CL_PLATFORM_NAME, &temp_str, &temp_str_len );
                
            success &= ( next_node->platform_name != 0 );
            
            next_node->device_name = 
                NS(OpenCLEnv_get_string_attribute_from_device_info)(
                    env->devices[ ii ], CL_DEVICE_NAME, &temp_str, 
                        &temp_str_len );
                
            success &= ( next_node->device_name != 0 );
            
            next_node->extensions = 
                NS(OpenCLEnv_get_string_attribute_from_device_info)(
                    env->devices[ ii ], CL_DEVICE_EXTENSIONS, &temp_str, 
                        &temp_str_len );
                
            success &= ( next_node->extensions != 0 );
            
            if( success )
            {
                bool platform_found = false;
                SIXTRL_SIZE_T jj = ZERO_SIZE;
                
                next_node->ptr_environment = env;
                next_node->env_device_index = ii;
                            
                while( jj < env->num_platforms )
                {
                    if( env->platforms[ jj ] == platform )
                    {
                        next_node->env_platform_index = jj;
                        platform_found = true;
                        break;
                    }
                    
                    ++jj;
                }
                
                success &= platform_found;
            }
            
            if( success )
            {
                assert( temp_str != 0 );
                
                memset( temp_str, ( int )'\0', temp_str_len );
                sprintf( temp_str, "%lu.%lu", 
                         next_node->env_platform_index, 
                         next_node->env_device_index );
                
                strncpy( next_node->id_str, temp_str, 16 );                
            }
            
            ++ii;
            next_node = next_node->ptr_next;
        }
        
        assert( ( !success ) || ( ii == env->num_devices ) );
        
        if( success )
        {
            env->ressources_flags |= NS(HAS_NODES);
        }
        
        free( temp_str );
        temp_str = 0;
    }
    
    if( !success )
    {
        NS(OpenCLEnv_free)( env );
        free( env );
        env = 0;
    }
    
    return env;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

void NS(OpenCLEnv_free)( NS(OpenCLEnv)* ocl_env )
{
    if( ocl_env != 0 )
    {
        uint64_t flags = UINT64_C( 0 );
        
        NS(OpenCLEnv_reset_kernel)( ocl_env );
        
        flags = ocl_env->ressources_flags;
        
        if( ( flags & NS(HAS_NODES ) ) == NS(HAS_NODES ) )
        {
            if( ocl_env->num_selected_nodes > ( SIXTRL_SIZE_T )1u )
            {
                free( ocl_env->selected_nodes );                
            }
            
            ocl_env->selected_nodes = 0;
            ocl_env->num_selected_nodes = ( SIXTRL_SIZE_T )0u;
            
            NS(OpenCLEnvNodeDevice_free)( ocl_env->nodes );
            flags &= ~( NS(HAS_NODES ) );
        }
        
        if( ( flags & NS(HAS_DEVICES) ) == NS(HAS_DEVICES) )
        {
            free( ocl_env->devices );
            ocl_env->devices = 0;
            flags &= ~( NS(HAS_DEVICES) );
        }
        
        if( ( flags & NS(HAS_PLATFORMS) ) == NS(HAS_PLATFORMS) )
        {
            free( ocl_env->platforms );
            ocl_env->platforms = 0;
            flags &= ~( NS(HAS_PLATFORMS) );
        }
        
        assert( flags == UINT64_C( 0 ) );
        
        NS(OpenCLEnv_preset)( ocl_env );        
    }
    
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(OpenCLEnvNodeDevice) const* NS(OpenCLEnv_get_node_devices)(
    const NS(OpenCLEnv) *const SIXTRL_RESTRICT ocl_env )
{
    return ( ocl_env != 0 ) ? ocl_env->nodes : 0;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

size_t NS(OpenCLEnv_get_num_node_devices)( 
    const NS(OpenCLEnv) *const SIXTRL_RESTRICT ocl_env )
{
    return ( ( ocl_env != 0 ) && ( ocl_env->nodes != 0 ) )
        ? ocl_env->num_devices : ( size_t )0u;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

const char *const NS(OpenCLEnv_get_current_kernel_function)( 
    const NS(OpenCLEnv) *const SIXTRL_RESTRICT ocl_env )
{
    return ( ocl_env != 0 ) ? ( ocl_env->current_kernel_function ) : 0;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(OpenCLEnv_is_ready)( 
    const NS(OpenCLEnv) *const SIXTRL_RESTRICT ocl_env )
{
    return ( ocl_env != 0 ) ? ocl_env->is_ready : false;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(OpenCLEnv_prepare)( NS(OpenCLEnv)* ocl_env, 
    char const* node_device_id, char const* kernel_function_name, 
    char* kernel_source_files, char const* compile_options,
    SIXTRL_SIZE_T const num_turns,
    NS(Particles) const* SIXTRL_RESTRICT particles,
    NS(BeamElements) const* SIXTRL_RESTRICT beam_elements )
{
    bool success = false;
    
    static SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;
    
    ( void )particles;
    
    if( ( ocl_env != 0 ) && 
        ( num_turns > 0u ) &&
        /* ( particles != 0 ) && */
        ( NS(BeamElements_get_num_of_blocks)( beam_elements ) > 
            ( NS(block_num_elements_t ) )0u ) && 
        ( NS(BeamElements_get_const_ptr_data_begin)( beam_elements ) != 0 ) &&
        ( NS(BeamElements_get_const_block_infos_begin)( beam_elements ) != 0 ) &&
        ( node_device_id != 0 ) && ( kernel_function_name != 0 ) && 
        ( kernel_source_files != 0 ) )
    {
        success = false;
        
        if( ocl_env->ressources_flags > (
            NS(HAS_PLATFORMS) | NS(HAS_DEVICES) | NS(HAS_NODES) ) )
        {
            NS(OpenCLEnv_reset_kernel)( ocl_env );
        }
        
        if( !ocl_env->is_ready )
        {
            typedef NS(OpenCLEnvNodeDevice) node_t;  
            node_t* node = 0;
            
            SIXTRL_SIZE_T num_kernel_files = ZERO_SIZE;
            SIXTRL_SIZE_T* source_line_offsets = 0;
                        
            char** path_kernel_files = NS(GpuKernel_create_file_list)(
                kernel_source_files, &num_kernel_files, 
                    NS(PATH_TO_BASE_DIR), "," );
                
            if( ( path_kernel_files != 0 ) && ( num_kernel_files > ZERO_SIZE ) )
            {
                source_line_offsets = ( SIXTRL_SIZE_T* )malloc( 
                    sizeof( SIXTRL_SIZE_T ) * num_kernel_files );
                
                ocl_env->kernel_source = NS(GpuKernel_collect_source_string)(
                    path_kernel_files, num_kernel_files, 1024u, source_line_offsets );
                
                if( ocl_env->kernel_source != 0 )
                {
                    ocl_env->ressources_flags |= NS(HAS_CURRENT_KERNEL_SOURCE);
                    success = true;
                }
            }
            
            if( success )
            {                
                node = ocl_env->nodes;
                success = false;
                
                while( node != 0 )
                {
                    if( ( strcmp( node->id_str, node_device_id ) == 0 ) &&
                        ( node->ptr_environment == ocl_env ) &&
                        ( node->env_platform_index < ocl_env->num_platforms ) &&
                        ( node->env_device_index < ocl_env->num_devices ) )
                    {
                        break;
                    }
                    
                    node = node->ptr_next;
                }
                
                if( node != 0 )
                {
                    SIXTRL_SIZE_T const id_str_len = strlen( node->id_str );
                    
                    ocl_env->selected_nodes = node;
                    ocl_env->num_selected_nodes = ( SIXTRL_SIZE_T )1u;
                    
                    ocl_env->current_id_str = ( char* )malloc( 
                        sizeof( char ) * ( id_str_len + 1 ) );
                        
                    strncpy( ocl_env->current_id_str, node->id_str, id_str_len );
                    ocl_env->ressources_flags |= NS(HAS_CURRENT_ID_STR);
                    success = true;
                }
            }
            
            if( success )
            {
                cl_int ret;
                
                cl_platform_id platform = 
                    ocl_env->platforms[ node->env_platform_index ];
                
                cl_device_id device = 
                    ocl_env->devices[ node->env_device_index ];
                    
                cl_context_properties const prop[] = 
                {
                    CL_CONTEXT_PLATFORM, ( cl_context_properties )platform, 0
                };
                
                ocl_env->context = clCreateContext( prop, 1, &device, 0, 0, &ret );
                
                if( ret != CL_SUCCESS )
                {
                    success = false;
                }
                
                if( success ) ocl_env->ressources_flags |= NS(HAS_CONTEXT);
            }
            
            if( success )
            {
                cl_int ret = CL_FALSE;
                
                ocl_env->program = clCreateProgramWithSource(
                    ocl_env->context, 1, ( const char** )&ocl_env->kernel_source, 
                        0, &ret );
            
                success &= ( ret == CL_SUCCESS );
            }
            
            if( success )
            {
                SIXTRL_SIZE_T const DEFAULT_LOC_COMP_OPTIONS_LEN = 10240u;
                
                cl_int ret = CL_FALSE;                
                cl_device_id device = ocl_env->devices[ node->env_device_index ];
                
                char* local_compile_options = 0;
                    
                SIXTRL_SIZE_T comp_options_len = ( compile_options != 0 ) 
                        ? ( strlen( compile_options ) + 1u )
                        : DEFAULT_LOC_COMP_OPTIONS_LEN;
                      
                success = false;
                        
                if( comp_options_len < DEFAULT_LOC_COMP_OPTIONS_LEN )
                {
                    comp_options_len = DEFAULT_LOC_COMP_OPTIONS_LEN;
                }
                    
                local_compile_options = ( char* )malloc( 
                    comp_options_len * sizeof( char ) );
                    
                assert(  local_compile_options != 0 );
                memset(  local_compile_options, ( int )'\0', comp_options_len );
                
                if( compile_options != 0 )
                {
                    strncpy( local_compile_options, compile_options, 
                             comp_options_len );
                }
                
                assert( ( ocl_env->selected_nodes != 0 ) &&
                        ( ocl_env->num_selected_nodes == 1u ) );
                
                if( ( ocl_env->selected_nodes->extensions != 0 ) && 
                    ( strstr( ocl_env->selected_nodes->extensions, 
                              "cl_khr_fp64" ) != 0 ) )
                {
                    char const enable_fp64_flag[] = " -D SIXTRL_CL_ENABLE_FP64=1 ";
                    
                    strncat( local_compile_options, enable_fp64_flag, 
                             comp_options_len - 
                             ( 1 + strlen( local_compile_options ) ) );
                }
                
                ret = clBuildProgram( ocl_env->program, 1, &device, 
                                      compile_options, 0, 0 );
                
                if( ret == CL_SUCCESS )
                {
                    success = true;
                    ocl_env->ressources_flags |= NS(HAS_PROGRAM);
                }
                else if( ret == CL_BUILD_PROGRAM_FAILURE )
                {
                    SIXTRL_SIZE_T log_len = ZERO_SIZE;
                    char* log_buffer = 0;
                    
                    clGetProgramBuildInfo( ocl_env->program, device, 
                        CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
                    
                    if( log_len > ZERO_SIZE )
                    {
                        #if !defined( NDEBUG )
                        FILE* fp = fopen( "./out.dump", "wb" );
                        if( fp != 0 )
                        {
                            fputs( ocl_env->kernel_source, fp );
                            fflush( fp );
                            fclose( fp );
                            fp = 0;
                        }
                        #endif /* !defined( NDEBUG ) */
                        
                        SIXTRL_SIZE_T ii = ZERO_SIZE;
                        
                        if( ( path_kernel_files != 0 ) && 
                            ( source_line_offsets != 0 ) )
                        {
                            for( ; ii < num_kernel_files ; ++ii )
                            {
                                printf( "file #%02lu : line_offset = %6lu : %s\r\n",
                                        ii, source_line_offsets[ ii ], 
                                        path_kernel_files[ ii ] );
                            }
                        }
                        
                        log_buffer = ( char* )malloc( 
                            sizeof( char ) * ( log_len + 1 ) );
                        
                        assert( log_buffer != 0 );
                    
                        ret = clGetProgramBuildInfo( ocl_env->program, device, 
                            CL_PROGRAM_BUILD_LOG, log_len, log_buffer, NULL );
                        
                        assert( ret == CL_SUCCESS );
                        
                        printf( "\r\n***************************************************\r\n"
                                "OpenCL Program Build Error: \r\n"
                                "***************************************************\r\n"
                                "%s\r\n\r\n"
                                "***************************************************\r\n"
                                "End of OpenCL Program Build Error\r\n"
                                "***************************************************\r\n",
                                log_buffer );
                        
                        free( log_buffer );
                        log_buffer = 0;
                    }
                }
                
                free( local_compile_options );
                local_compile_options = 0;
            }
            
            if( success )
            {
                cl_int ret = CL_FALSE;
                
                SIXTRL_SIZE_T const kernel_fn_name_len = (SIXTRL_SIZE_T )1u + 
                    strlen( kernel_function_name );
                    
                success = false;
                    
                ocl_env->current_kernel_function = ( char* )malloc(
                    sizeof( char ) * kernel_fn_name_len );
                
                if( ocl_env->current_kernel_function != 0 )
                {
                    memset( ocl_env->current_kernel_function, ( int )'\0',
                            sizeof( char ) * kernel_fn_name_len );
                    
                    strncpy( ocl_env->current_kernel_function, 
                             kernel_function_name, kernel_fn_name_len );
                    
                    ocl_env->kernel = clCreateKernel( 
                        ocl_env->program, ocl_env->current_kernel_function, &ret );
                    
                    if( ret == CL_SUCCESS )
                    {
                        ocl_env->ressources_flags |= NS(HAS_KERNEL);
                        ocl_env->ressources_flags |= NS(HAS_CURRENT_KERNEL_FN);
                        
                        success = true;
                    }
                }
            }
            
            if( success )
            {
                cl_int ret = CL_FALSE;
                success    = false;
                
                ocl_env->queue = clCreateCommandQueue( 
                    ocl_env->context, ocl_env->devices[ node->env_device_index ], 
                        0, &ret );
                
                
                
                if( ret == CL_SUCCESS )
                {
                    ocl_env->ressources_flags |= NS(HAS_QUEUE );
                    success = true;
                }
            }
            
            if( success )
            {
                cl_int ret_info = CL_FALSE;
                cl_int ret_data = CL_FALSE;
                
                SIXTRL_SIZE_T const num_elem  = 
                    NS(BeamElements_get_num_of_blocks)( beam_elements );
                
                SIXTRL_SIZE_T const info_size = sizeof( NS(BlockInfo) ) * num_elem;
                
                SIXTRL_SIZE_T const data_size = 
                    NS(BlockInfo_get_total_storage_size)(
                        NS(BeamElements_get_const_block_infos_begin)( beam_elements ),
                        num_elem );
                
                success = false;
                
                ocl_env->num_turns = num_turns;
                ocl_env->num_beam_elements = num_elem;
                ocl_env->num_particles = ( SIXTRL_SIZE_T )1u; /* HACK!!!! */
                
                ocl_env->beam_elem_info_buffer = clCreateBuffer(
                    ocl_env->context, CL_MEM_READ_WRITE, info_size, 0, &ret_info );
                
                ocl_env->beam_elem_data_buffer = clCreateBuffer(
                    ocl_env->context, CL_MEM_READ_WRITE, data_size, 0, &ret_data );
                
                if( ( ret_data == CL_SUCCESS ) && ( ret_info == CL_SUCCESS ) )
                {
                    int cl_ret = CL_FALSE;
                    
                    ocl_env->ressources_flags |= NS(HAS_BEAM_ELEMS_BUFFERS);
                    
                    cl_ret  = clEnqueueWriteBuffer( ocl_env->queue, 
                        ocl_env->beam_elem_info_buffer, CL_TRUE, 0u, info_size,
                            beam_elements->info_begin, 0u, 0, 0 );
                    
                    cl_ret |= clEnqueueWriteBuffer( ocl_env->queue,
                        ocl_env->beam_elem_data_buffer, CL_TRUE, 0u, data_size,
                            beam_elements->data_begin, 0u, 0, 0 );
                    
                    if( cl_ret == CL_SUCCESS )
                    {
                        SIXTRL_SIZE_T const U64_SIZE = sizeof( SIXTRL_UINT64_T );
                        
                        cl_ret |= clSetKernelArg( ocl_env->kernel, 0, 
                              U64_SIZE, &ocl_env->num_turns );
                        
                        cl_ret |= clSetKernelArg( ocl_env->kernel, 1, 
                              U64_SIZE, &ocl_env->num_beam_elements );
                        
                        cl_ret |= clSetKernelArg( ocl_env->kernel, 2, 
                              U64_SIZE, &ocl_env->num_particles );
                        
                        cl_ret |= clSetKernelArg( ocl_env->kernel, 3, 
                              sizeof( cl_mem ), &ocl_env->beam_elem_info_buffer );
                        
                        cl_ret |= clSetKernelArg( ocl_env->kernel, 4, 
                              sizeof( cl_mem ), &ocl_env->beam_elem_data_buffer );
                        
                        success = ( cl_ret == CL_SUCCESS );
                    }
                }                
            }
            
            if( success ) ocl_env->is_ready = true;
            
            NS(GpuKernel_free_file_list)( 
                    path_kernel_files, num_kernel_files );
            
            path_kernel_files = 0;
            
            free( source_line_offsets );
            source_line_offsets = 0;
        }
    }
    
    return success;
}

bool NS(OpenCLEnv_track_particles)( 
    struct NS(OpenCLEnv)* ocl_env, 
    NS(Particles)*   SIXTRL_RESTRICT particles, 
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements )
{
    bool success = false;
    
    ( void )particles;
    
    if( ( ocl_env != 0 ) && ( ocl_env->is_ready ) &&
        ( beam_elements != 0 ) && 
        ( NS(BeamElements_get_num_of_blocks)( beam_elements) > 0 ) &&
        ( NS(BeamElements_get_num_of_blocks)( beam_elements) == 
            ocl_env->num_beam_elements ) )
    {
        size_t work_units_per_kernel = ( size_t )ocl_env->num_particles;
        
        cl_event finished_kernel;
        
        cl_int ret = clEnqueueNDRangeKernel(
            ocl_env->queue, ocl_env->kernel, 1u, 0, &work_units_per_kernel,
            0, 0, 0, &finished_kernel );
        
        if( ret == CL_SUCCESS )
        {
            ret = clWaitForEvents( 1u, &finished_kernel );
            
            success = ( ret == CL_SUCCESS );            
            clReleaseEvent( finished_kernel );                       
        }
    }
        
    return success;
}


/* -------------------------------------------------------------------------- */

void NS(OpenCLEnv_reset_kernel)( NS(OpenCLEnv)* ocl_env )
{
    if( ocl_env != 0 ) 
    {
        uint64_t flags = ocl_env->ressources_flags;
        
        ocl_env->is_ready = false;
        
        if( NS(HAS_CURRENT_ID_STR) == ( flags & NS(HAS_CURRENT_ID_STR) ) )
        {
            free( ocl_env->current_id_str );
            ocl_env->current_id_str = 0;
            flags &= ~( NS(HAS_CURRENT_ID_STR) );
        }
        
        if( NS(HAS_CURRENT_KERNEL_FN) == ( flags & NS(HAS_CURRENT_KERNEL_FN) ) )
        {
            free( ocl_env->current_kernel_function );
            ocl_env->current_kernel_function = 0;
            flags &= ~( NS(HAS_CURRENT_KERNEL_FN) );
        }
        
        if( NS(HAS_CURRENT_KERNEL_SOURCE) == 
            ( flags & NS(HAS_CURRENT_KERNEL_SOURCE ) ) )
        {
            free( ocl_env->kernel_source );
            ocl_env->kernel_source = 0;
            flags &= ~( NS(HAS_CURRENT_KERNEL_SOURCE) );
        }
        
        if( NS(HAS_QUEUE) == ( NS(HAS_QUEUE) & flags ) )
        {
            clReleaseCommandQueue(ocl_env->queue );
            flags &= ~( NS( HAS_QUEUE ) ); 
        }
        
        if( NS(HAS_KERNEL) == ( NS(HAS_KERNEL) & flags ) )
        {
            clReleaseKernel( ocl_env->kernel );
            flags &= ~( NS( HAS_KERNEL ) );
        }
        
        if( NS(HAS_PROGRAM) == ( NS(HAS_PROGRAM) & flags ) )
        {
            clReleaseProgram( ocl_env->program );
            flags &= ~( NS( HAS_PROGRAM ) );
        }
        
        if( NS(HAS_CONTEXT) == ( NS(HAS_CONTEXT) & flags ) )
        {
            clReleaseContext( ocl_env->context );
            flags &= ~( NS( HAS_CONTEXT ) );
        }
        
        if( NS(HAS_E_BY_E_BUFFER) == ( NS(HAS_E_BY_E_BUFFER) & flags ) )
        {
            clReleaseMemObject( ocl_env->elem_by_elem_buffer );
            flags &= ~( NS(HAS_E_BY_E_BUFFER) );
        }
        
        if( NS(HAS_T_BY_T_BUFFER) == ( NS(HAS_T_BY_T_BUFFER) & flags ) )
        {
            clReleaseMemObject( ocl_env->turn_by_turn_buffer );
            flags &= ~( NS(HAS_T_BY_T_BUFFER) );
        }
        
        if( NS(HAS_BEAM_ELEMS_BUFFERS) == ( NS(HAS_BEAM_ELEMS_BUFFERS) & flags ) )
        {
            clReleaseMemObject( ocl_env->beam_elem_info_buffer );
            clReleaseMemObject( ocl_env->beam_elem_data_buffer );
            flags &= ~( NS(HAS_BEAM_ELEMS_BUFFERS) );            
        }
            
        if( NS(HAS_PARTICLES_BUFFER) == ( NS(HAS_PARTICLES_BUFFER) & flags ) )
        {
            clReleaseMemObject( ocl_env->particles_buffer );
            flags &= ~( NS(HAS_PARTICLES_BUFFER) );            
        }
        
        ocl_env->ressources_flags = flags;
    }
    
    return;
}

/* -------------------------------------------------------------------------- */

cl_platform_id* NS(OpenCLEnv_get_valid_platforms)( 
    SIXTRL_SIZE_T* ptr_num_valid_platforms, 
    SIXTRL_SIZE_T* ptr_num_of_potentially_valid_devices )
{
    static cl_uint const MAX_NUM_PLATFORMS        = ( cl_uint )10u;    
    static cl_uint const MAX_DEVICES_PER_PLATFORM = ( cl_uint )10u;
    static SIXTRL_SIZE_T const ZERO = ( SIXTRL_SIZE_T )0u;
        
    cl_platform_id* valid_platforms = 0;
    
    cl_uint temp_num_platforms = ZERO;        
    cl_int ret = clGetPlatformIDs( MAX_NUM_PLATFORMS, 0, &temp_num_platforms );
            
    if( ( ret == CL_SUCCESS ) && ( temp_num_platforms > ZERO ) &&
        ( ptr_num_valid_platforms != 0 ) )
    {
        SIXTRL_SIZE_T num_of_potentially_valid_devices = ZERO;
        SIXTRL_SIZE_T num_valid_platforms = ZERO;
        cl_uint* temp_num_devices = 0;
        
        cl_platform_id* temp_platforms = ( cl_platform_id* )malloc(
            sizeof( cl_platform_id ) * temp_num_platforms );
        
        if( temp_platforms == 0 ) return valid_platforms;
        
        ret = clGetPlatformIDs( temp_num_platforms, temp_platforms, 0 );
        
        if( ret == CL_SUCCESS )
        {
            cl_uint ii = ZERO;
            
            temp_num_devices = 
                ( cl_uint* )malloc( sizeof( cl_uint ) * temp_num_platforms );
            
            if( temp_num_devices == 0 )
            {
                free( temp_platforms );
                temp_platforms = 0;
                
                return valid_platforms;
            }
        
            for( ; ii < temp_num_platforms ; ++ii )
            {
                ret = clGetDeviceIDs( temp_platforms[ ii ], CL_DEVICE_TYPE_ALL, 
                    MAX_DEVICES_PER_PLATFORM, 0, &temp_num_devices[ ii ] );
                
                if( ( ret == CL_SUCCESS ) && ( temp_num_devices[ ii ] > ZERO ) )
                {
                    num_of_potentially_valid_devices += temp_num_devices[ ii ];
                    ++num_valid_platforms;                    
                }
                else
                {
                    temp_num_devices[ ii ] = ZERO;
                }
            }
            
            if( num_valid_platforms > ZERO )
            {
                valid_platforms = ( cl_platform_id* )malloc(
                    sizeof( cl_platform_id ) * num_valid_platforms );
            }
            
            if( ( valid_platforms != 0 ) && ( temp_num_devices != 0 ) )
            {
                SIXTRL_SIZE_T jj = ZERO;
                
                for( ii = ZERO ; ii < temp_num_platforms ; ++ii )
                {
                    if( temp_num_devices[ ii ] > ZERO )
                    {
                        valid_platforms[ jj++ ] = temp_platforms[ ii ];
                    }
                }
                
                if( jj == num_valid_platforms )
                {
                    if( ptr_num_of_potentially_valid_devices != 0 )
                    {
                        *ptr_num_of_potentially_valid_devices =
                             num_of_potentially_valid_devices;
                    }
                    
                    *ptr_num_valid_platforms = num_valid_platforms;
                }
                else
                {
                    *ptr_num_valid_platforms = ZERO;
                    
                    if( ptr_num_of_potentially_valid_devices != 0 )
                    {
                        *ptr_num_of_potentially_valid_devices = ZERO;
                    }
                    
                    free( valid_platforms );
                    valid_platforms = 0;                    
                }
                
                free( temp_num_devices );
                temp_num_devices = 0;
            }
        }
        
        free( temp_platforms );
        temp_platforms = 0;
    }
    
    return valid_platforms;
}

cl_device_id* NS(OpenCLEnv_get_valid_devices)( 
    cl_platform_id* platforms, SIXTRL_SIZE_T const num_platforms, 
    cl_device_type const device_type, SIXTRL_SIZE_T* ptr_num_valid_devices )
{
    static SIXTRL_SIZE_T const ZERO = ( SIXTRL_SIZE_T )0u;
    
    cl_device_id* valid_devices = 0;
    
    if( ( platforms != 0 ) && ( num_platforms > ZERO ) &&
        ( ptr_num_valid_devices != 0 ) )
    {
        static SIXTRL_SIZE_T const MAX_DEVS_PER_PLATFORM = ( SIXTRL_SIZE_T )10u;
        
        SIXTRL_SIZE_T temp_num_devices  = ( ptr_num_valid_devices != 0 ) 
                ? *( ptr_num_valid_devices ) : ZERO;
                
        SIXTRL_SIZE_T ii = ZERO;
        
        if( temp_num_devices == ZERO )
        {
            for( ; ii < num_platforms ; ++ii )
            {
                cl_uint num_devs = ZERO;
                cl_int ret = clGetDeviceIDs( platforms[ ii ], 
                    device_type, MAX_DEVS_PER_PLATFORM, 0, &num_devs );
                
                if( ( ret == CL_SUCCESS ) && ( num_devs > ZERO ) )
                {
                    temp_num_devices += num_devs;
                }
            }
        }
        
        if( temp_num_devices > ZERO )
        {
            SIXTRL_SIZE_T devices_queried  = ZERO;
            SIXTRL_SIZE_T devices_to_query = temp_num_devices;
            
            cl_device_id* temp_devices = ( cl_device_id* )malloc( 
                sizeof( cl_device_id ) * temp_num_devices );
            
            if( temp_devices == 0 ) return valid_devices;
            
            for( ii = ZERO ; ii < num_platforms ; ++ii )
            {
                cl_uint devices_found = ZERO;
                cl_int const ret = clGetDeviceIDs( platforms[ ii ], 
                    device_type, devices_to_query, 
                    &temp_devices[ devices_queried ], &devices_found );
                
                if( ret == CL_SUCCESS )
                {
                    if( ( devices_found > ZERO ) && 
                        ( ( devices_found + devices_queried ) <=
                           temp_num_devices ) )
                    {
                        devices_queried += devices_found;                        
                    }
                    else if( devices_found > ZERO )                        
                    {
                        break;
                    }
                }
            }
            
            if( devices_queried > ZERO )
            {
                valid_devices = ( cl_device_id* )malloc( 
                    sizeof( cl_device_id ) * devices_queried );
                
                if( valid_devices != 0 )
                {
                    for( ii = ZERO ; ii < devices_queried ; ++ii )
                    {
                        valid_devices[ ii ] = temp_devices[ ii ];
                    }
                    
                    *ptr_num_valid_devices = devices_queried;
                }
            }
            
            free( temp_devices );
            temp_devices = 0;
        }
    }
        
    return valid_devices;
}

char* NS(OpenCLEnv_get_string_attribute_from_platform_info)(
    cl_platform_id platform, cl_platform_info attribute, 
    char** ptr_to_string, SIXTRL_SIZE_T* ptr_to_string_max_size )
{
    char* str = 0;
    
    if( ( ptr_to_string_max_size != 0 ) && ( ptr_to_string != 0 ) )
    {
        static SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;
        static SIXTRL_SIZE_T const MIN_VAL_BUFFER_SIZE = 
            ( SIXTRL_SIZE_T )10240u;
        
        SIXTRL_SIZE_T attr_bytes = sizeof( char );
        
        cl_int ret = CL_FALSE;
            
        if( *ptr_to_string == 0 )
        {
            SIXTRL_SIZE_T const len = 
                ( *ptr_to_string_max_size > MIN_VAL_BUFFER_SIZE ) 
                    ? ( *ptr_to_string_max_size ) : ( MIN_VAL_BUFFER_SIZE );
                    
            SIXTRL_SIZE_T const num_bytes = sizeof( char ) * len;
                    
            *ptr_to_string = ( char* )malloc( num_bytes );
            *ptr_to_string_max_size = len;
        }        
        else if( ( *ptr_to_string != 0 ) && 
                 ( *ptr_to_string_max_size < MIN_VAL_BUFFER_SIZE ) )
        {
            SIXTRL_SIZE_T const num_bytes = MIN_VAL_BUFFER_SIZE * sizeof( char );            
            char* new_string = ( char* )malloc( num_bytes );
            
            
            free( *ptr_to_string );            
            *ptr_to_string = new_string;
            *ptr_to_string_max_size = MIN_VAL_BUFFER_SIZE;
        }
        
        assert( ( *ptr_to_string != 0 ) && 
                ( *ptr_to_string_max_size >= MIN_VAL_BUFFER_SIZE ) );
        
        attr_bytes *= ( *ptr_to_string_max_size );
        memset( *ptr_to_string, ( int )'\0', attr_bytes );
        
        ret = clGetPlatformInfo( platform, attribute, attr_bytes, 
                                 *ptr_to_string, 0 );
        
        if( ret == CL_SUCCESS )
        {
            SIXTRL_SIZE_T out_str_length = strlen( *ptr_to_string );
            
            if( out_str_length > ZERO_SIZE )
            {
                ++out_str_length;
                
                str = ( char* )malloc( out_str_length * sizeof( char ) );
                
                if( str != 0 )
                {
                    strncpy( str, *ptr_to_string, out_str_length );
                }
            }            
        }
    }
    
    return str;
}


char* NS(OpenCLEnv_get_string_attribute_from_device_info)(
    cl_device_id device, cl_device_info attribute, 
    char** ptr_to_string, SIXTRL_SIZE_T* ptr_to_string_max_size )
{
    char* str = 0;
    
    if( ( ptr_to_string_max_size != 0 ) && ( ptr_to_string != 0 ) )
    {
        static SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;
        static SIXTRL_SIZE_T const MIN_VAL_BUFFER_SIZE = 
            ( SIXTRL_SIZE_T )10240u;
        
        SIXTRL_SIZE_T attr_bytes = sizeof( char );
        
        cl_int ret = CL_FALSE;
            
        if( *ptr_to_string == 0 )
        {
            SIXTRL_SIZE_T const len = 
                ( *ptr_to_string_max_size > MIN_VAL_BUFFER_SIZE ) 
                    ? ( *ptr_to_string_max_size ) : ( MIN_VAL_BUFFER_SIZE );
                    
            SIXTRL_SIZE_T const num_bytes = sizeof( char ) * len;
                    
            *ptr_to_string = ( char* )malloc( num_bytes );
            *ptr_to_string_max_size = len;
        }        
        else if( ( *ptr_to_string != 0 ) && 
                 ( *ptr_to_string_max_size < MIN_VAL_BUFFER_SIZE ) )
        {
            SIXTRL_SIZE_T const num_bytes = MIN_VAL_BUFFER_SIZE * sizeof( char );            
            char* new_string = ( char* )malloc( num_bytes );
            
            
            free( *ptr_to_string );            
            *ptr_to_string = new_string;
            *ptr_to_string_max_size = MIN_VAL_BUFFER_SIZE;
        }
        
        assert( ( *ptr_to_string != 0 ) && 
                ( *ptr_to_string_max_size >= MIN_VAL_BUFFER_SIZE ) );
        
        attr_bytes *= ( *ptr_to_string_max_size );
        memset( *ptr_to_string, ( int )'\0', attr_bytes );
        
        ret = clGetDeviceInfo( device, attribute, attr_bytes, *ptr_to_string, 0 );
        
        if( ret == CL_SUCCESS )
        {
            SIXTRL_SIZE_T out_str_length = strlen( *ptr_to_string );
            
            if( out_str_length > ZERO_SIZE )
            {
                ++out_str_length;
                
                str = ( char* )malloc( out_str_length * sizeof( char ) );
                
                if( str != 0 )
                {
                    strncpy( str, *ptr_to_string, out_str_length );
                }
            }            
        }
    }
    
    return str;
}

/* end: sixtracklib/opencl/details/ocl_environment.c */
