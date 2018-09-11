#ifndef SIXTRACKLIB_OPENCL_IMPL_MANAGED_BUFFER_REMAP_KERNEL_OPENCL_CL__
#define SIXTRACKLIB_OPENCL_IMPL_MANAGED_BUFFER_REMAP_KERNEL_OPENCL_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/buffer_defines.h"
    #include "sixtracklib/common/impl/managed_buffer_minimal.h"
    #include "sixtracklib/common/impl/managed_buffer_remap.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#pragma OPENCL_EXTENSION cl_khr_int32_extended_atomics

__kernel void NS(ManagedBuffer_remap_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC int* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t) buf_size_t;

    size_t const global_id           = get_global_id( 0 );
    size_t const gid_to_remap_buffer = ( size_t )0u;

    if( gid_to_remap_buffer == global_id )
    {
        long int success_flag = ( buffer_begin != SIXTRL_NULLPTR ) ? 0 : -1;

        buf_size_t const slot_size = ( buf_size_t )8u;

        if( 0 != NS(ManagedBuffer_remap)( buffer_begin, slot_size ) )
        {
            success_flag |= -2;
        }

        if( ptr_success_flag != SIXTRL_NULLPTR )
        {
            atomic_or( ptr_success_flag, success_flag );
        }
    }

    return;
}

__kernel void NS(ManagedBuffer_remap_io_buffers_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT in_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC int* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t) buf_size_t;

    size_t const global_id = get_global_id( 0 );
    size_t const gid_to_remap_input_buffer  = ( size_t )0u;
    size_t const gid_to_remap_output_buffer = ( get_global_size( 0 ) > 1u )
        ? ( gid_to_remap_input_buffer + 1u )
        : ( gid_to_remap_input_buffer );

    if( global_id <= gid_to_remap_output_buffer )
    {
        long int success_flag = ( long int )0;
        buf_size_t const slot_size = ( buf_size_t )8u;

        if( global_id == gid_to_remap_input_buffer )
        {
            if( ( in_buffer_begin != SIXTRL_NULLPTR ) &&
                ( in_buffer_begin != out_buffer_begin ) )
            {
                printf( "\r\nDEVICE: remap_kernel -> before remapping :: in_buffer\r\n" );
                NS(ManagedBuffer_print_header)( in_buffer_begin, slot_size );

                if( 0 != NS(ManagedBuffer_remap)( in_buffer_begin, slot_size ) )
                {
                    success_flag |= -2;
                }
                else
                {
                    printf( "\r\nDEVICE: remap_kernel -> after remapping :: in_buffer\r\n" );
                    NS(ManagedBuffer_print_header)( in_buffer_begin, slot_size );

                    __global NS(Object) const* ptr_g_obj =
                        NS(ManagedBuffer_get_const_objects_index_begin)(
                            in_buffer_begin, slot_size );

                    NS(Object) obj_info = *ptr_g_obj;

                    printf( "obj->begin_addr: %20lu | %20lu\r\n",
                            ( unsigned long )ptr_g_obj->begin_addr,
                            ( unsigned long )obj_info.begin_addr );

                    __global NS(Particles) const* ptr_g_particles =
                        ( __global NS(Particles) const* )( obj_info.begin_addr );

                    printf( "num_particles: %20lu\r\n",
                            ptr_g_particles->num_particles );

                    printf( "&q0[ 0 ]: %20lu\r\n",
                            ( unsigned long )&ptr_g_particles->q0[ 0 ] );

                    printf( "sizeof() = %lu\r\n", sizeof( unsigned long ) );

                    NS(Particles) particles = *ptr_g_particles;

                    //SIXTRL_REAL_T const q0 = particles->q0[ 0 ];

                    printf( "\r\nDEVICE: remap_kernel -> printout of particles after remapping :: in_buffer\r\n" );
                    printf( "base_addr : %20lu \r\n"
                            "addr q0   : %20lu \r\n",
                            ( unsigned long )ptr_g_particles,
                            ( unsigned long )ptr_g_particles->q0 );
                }
            }
            else
            {
                success_flag |= -1;
            }

//             printf( "\r\nDEVICE: remap_kernel -> after mapping :: in_buffer\r\n" );
//             NS(ManagedBuffer_print_header)( in_buffer_begin, slot_size );
        }

        if( global_id == gid_to_remap_output_buffer )
        {
            if( ( out_buffer_begin != SIXTRL_NULLPTR ) &&
                ( out_buffer_begin != in_buffer_begin ) )
            {
//                 printf( "\r\nDEVICE: remap_kernel -> before mapping :: out_buffer\r\n" );
//                 NS(ManagedBuffer_print_header)( out_buffer_begin, slot_size );

                if( 0 != NS(ManagedBuffer_remap)(
                        out_buffer_begin, slot_size ) )
                {
                    success_flag |= -4;
                }
//                 else
//                 {
//                     SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* obj =
//                         NS(ManagedBuffer_get_const_objects_index_begin)(
//                             out_buffer_begin, slot_size );

//                     printf("\r\nDEVICE: remap_kernel -> printout of first obj after remapping :: out_buffer\r\n" );
//                     NS(Object_print_slots)( obj, 25u );
//                     printf( "\r\n" );
//                 }
            }
            else
            {
                success_flag |= -1;
            }

//             printf( "\r\nDEVICE: remap_kernel -> after mapping :: out_buffer\r\n" );
//             NS(ManagedBuffer_print_header)( out_buffer_begin, slot_size );
        }

        if( ptr_success_flag != SIXTRL_NULLPTR )
        {
            atomic_or( ptr_success_flag, success_flag );
        }
    }

    return;
}

#endif /* SIXTRACKLIB_OPENCL_IMPL_MANAGED_BUFFER_REMAP_KERNEL_OPENCL_CL__ */

/* end: sixtracklib/opencl/impl/managed_buffer_remap_kernel.cl */
