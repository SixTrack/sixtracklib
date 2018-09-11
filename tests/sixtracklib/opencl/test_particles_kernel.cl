#ifndef TESTS_SIXTRACKLIB_OPENCL_TEST_PARTICLES_KERNEL_OPENCL_CL__
#define TESTS_SIXTRACKLIB_OPENCL_TEST_PARTICLES_KERNEL_OPENCL_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/buffer_defines.h"
    #include "sixtracklib/common/impl/managed_buffer_minimal.h"
    #include "sixtracklib/common/impl/managed_buffer_remap.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#pragma OPENCL_EXTENSION cl_khr_int32_extended_atomics

__kernel void NS(Particles_copy_buffer_opencl)(
    __global unsigned char const* SIXTRL_RESTRICT in_buffer_begin,
    __global unsigned char* SIXTRL_RESTRICT out_buffer_begin,
    __global int* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef __global NS(Object)*       obj_iter_t;
    typedef __global NS(Object) const* obj_const_iter_t;

    typedef __global NS(Particles) const* ptr_const_particles_t;
    typedef __global NS(Particles)*       ptr_particles_t;

    int success_flag = ( int )0u;
    buf_size_t const slot_size = ( buf_size_t )8u;

    if( ( !NS(ManagedBuffer_needs_remapping(  in_buffer_begin,  slot_size ) ) ) &&
        ( !NS(ManagedBuffer_needs_remapping( out_buffer_begin,  slot_size ) ) ) &&
        (  NS(ManagedBuffer_get_num_objects)( out_buffer_begin, slot_size ) ==
           NS(ManagedBuffer_get_num_objects)(  in_buffer_begin, slot_size ) ) )
    {
        obj_const_iter_t in_obj_it  = NS(ManagedBuffer_get_const_objects_index_begin)(
            in_buffer_begin, slot_size );

        obj_const_iter_t in_obj_end = NS(ManagedBuffer_get_const_objects_index_end)(
            in_buffer_begin, slot_size );

        obj_iter_t out_obj_it = NS(ManagedBuffer_get_objects_index_begin)(
            out_buffer_begin, slot_size );

        buf_size_t obj_begin_index = ( buf_size_t )0u;

        size_t const global_id = get_global_id( 0 );

        if( global_id == 0u )
        {
            printf( "\r\nDEVICE: copy_kernel :: in_buffer\r\n" );
            NS(ManagedBuffer_print_header)( in_buffer_begin, slot_size );

            printf( "\r\nDEVICE: copy_kernel :: out_buffer\r\n" );
            NS(ManagedBuffer_print_header)( out_buffer_begin, slot_size );
            printf( "\r\n" );
        }

        for( ; in_obj_it != in_obj_end ; ++in_obj_it, ++out_obj_it )
        {
            ptr_const_particles_t in_particles = ( ptr_const_particles_t )(
                uintptr_t )NS(Object_get_begin_addr)( in_obj_it );

            ptr_particles_t out_particles = ( ptr_particles_t )(
                uintptr_t )NS(Object_get_begin_addr)( out_obj_it );

            buf_size_t const nn =
                NS(Particles_get_num_of_particles)( in_particles );

            buf_size_t const obj_end_index = obj_begin_index + nn;

            if( global_id == 0u )
            {
                printf( " particle_id: %8lu | "
                    "in_particles addr: 0x%016lu | "
                    "in_particles->s addr: 0x%016lu || "
                    "out_particles addr: 0x%016lu | "
                    "out_particles->s addr: 0x%016lu\r\n",
                    ( unsigned long )global_id,
                    ( unsigned long )in_particles,
                    ( unsigned long )in_particles->s,
                    ( unsigned long )out_particles,
                    ( unsigned long )out_particles->s );
            }

            if( nn == NS(Particles_get_num_of_particles)( out_particles ) )
            {
                if( ( global_id >= obj_begin_index ) &&
                    ( global_id <  obj_end_index ) )
                {
                    buf_size_t const particle_id = global_id - obj_begin_index;

                    if( particle_id < nn )
                    {
                        SIXTRL_REAL_T const s = in_particles->s[ particle_id ];
                        out_particles->s[ particle_id ] = s;
                    }
                    else
                    {
                        success_flag |= -1;
                    }

                    break;
                }
            }
            else
            {
                success_flag |= -2;
            }
        }
    }
    else
    {
        if( NS(ManagedBuffer_needs_remapping(  in_buffer_begin, slot_size ) ) )
        {
            success_flag |= -4;
        }

        if( NS(ManagedBuffer_needs_remapping( out_buffer_begin, slot_size ) ) )
        {
            success_flag |= -8;
        }

        if(  NS(ManagedBuffer_get_num_objects)( out_buffer_begin, slot_size ) !=
             NS(ManagedBuffer_get_num_objects)(  in_buffer_begin, slot_size ) )
        {
            success_flag |= -16;
        }
    }

    if( ( success_flag != 0 ) && ( ptr_success_flag != SIXTRL_NULLPTR ) )
    {
        atomic_or( ptr_success_flag, success_flag );
    }

    return;
}

#endif /* TESTS_SIXTRACKLIB_OPENCL_TEST_PARTICLES_KERNEL_OPENCL_CL__ */

/* end: tests/sixtracklib/opencl/test_particles_kernel.cl */
