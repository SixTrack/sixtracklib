#include "sixtracklib/cuda/wrappers/track_job_wrappers.h"
#include "sixtracklib/cuda/control/kernel_config.h"

#include <cuda_runtime_api.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer/buffer_type.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/argument_base.h"
#include "sixtracklib/common/particles/definitions.h"
#include "sixtracklib/common/track/definitions.h"

#include "sixtracklib/cuda/definitions.h"
#include "sixtracklib/cuda/argument.h"
#include "sixtracklib/cuda/control/kernel_config.h"

#include "sixtracklib/cuda/kernels/extract_particles_addr.cuh"
#include "sixtracklib/cuda/kernels/be_monitors_assign_out_buffer.cuh"
#include "sixtracklib/cuda/kernels/elem_by_elem_assign_out_buffer.cuh"
#include "sixtracklib/cuda/kernels/assign_address_item.cuh"
#include "sixtracklib/cuda/kernels/track_particles.cuh"

void NS(Track_particles_until_turn_cuda_wrapper)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT particles_arg,
    NS(buffer_size_t) const pset_index,
    NS(CudaArgument)* SIXTRL_RESTRICT beam_elem_arg,
    NS(buffer_size_t) const until_turn,
    NS(CudaArgument)* SIXTRL_RESTRICT dbg_register_arg )
{
    dim3 const* ptr_blocks =
        NS(CudaKernelConfig_get_ptr_const_blocks)( kernel_config );

    bool const is_finished = NS(KernelConfig_needs_update)( kernel_config );

    dim3 const* ptr_threads =
        NS(CudaKernelConfig_get_ptr_const_threads_per_block)( kernel_config );

    if( dbg_register_arg == SIXTRL_NULLPTR )
    {
        SIXTRL_ASSERT( ptr_blocks  != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( ptr_threads != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( !NS(KernelConfig_needs_update)( kernel_config ) );
        SIXTRL_ASSERT( NS(KernelConfig_get_arch_id)( kernel_config ) ==
            NS(ARCHITECTURE_CUDA) );

        SIXTRL_ASSERT( particles_arg != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Argument_get_arch_id)( particles_arg ) ==
                   NS(ARCHITECTURE_CUDA) );

        SIXTRL_ASSERT( NS(Argument_uses_cobjects_buffer)( particles_arg ) );
        SIXTRL_ASSERT( NS(Argument_get_const_cobjects_buffer)(
            particles_arg ) != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( beam_elem_arg != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Argument_get_arch_id)( beam_elem_arg ) ==
                   NS(ARCHITECTURE_CUDA) );

        SIXTRL_ASSERT( NS(Argument_uses_cobjects_buffer)( beam_elem_arg ) );
        SIXTRL_ASSERT( NS(Argument_get_const_cobjects_buffer)(
            beam_elem_arg ) != SIXTRL_NULLPTR );

        SIXTRL_ASSERT(
            NS(Argument_get_cobjects_buffer_slot_size)( particles_arg ) ==
            NS(Argument_get_cobjects_buffer_slot_size)( beam_elem_arg ) );

        NS(Track_particles_until_turn_cuda)<<< *ptr_blocks, *ptr_threads >>>(
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                particles_arg ), pset_index,
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                beam_elem_arg ), until_turn,
            NS(Argument_get_cobjects_buffer_slot_size)( particles_arg ) );
    }
    else if( ( ptr_blocks != SIXTRL_NULLPTR ) &&
        ( ptr_threads != SIXTRL_NULLPTR ) &&
        ( NS(KernelConfig_get_arch_id)( kernel_config ) ==
                NS(ARCHITECTURE_CUDA) ) &&
        ( !NS(KernelConfig_needs_update)( kernel_config ) ) )
    {
        NS(Track_particles_until_turn_cuda_debug)<<< *ptr_blocks, *ptr_threads >>>(
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                particles_arg ), pset_index,
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                beam_elem_arg ), until_turn,
            NS(Argument_get_cobjects_buffer_slot_size)( particles_arg ),
            NS(CudaArgument_get_cuda_arg_buffer_as_debugging_register_begin)(
                dbg_register_arg ) );
    }
}

void NS(Track_particles_elem_by_elem_until_turn_cuda_wrapper)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT particles_arg,
    NS(buffer_size_t) const pset_index,
    NS(CudaArgument)* SIXTRL_RESTRICT beam_elem_arg,
    NS(CudaArgument)* SIXTRL_RESTRICT config_buffer_arg,
    NS(buffer_size_t) const elem_by_elem_config_index,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    NS(CudaArgument)* SIXTRL_RESTRICT dbg_register_arg )
{
    dim3 const* ptr_blocks = NS(CudaKernelConfig_get_ptr_const_blocks)(
        kernel_config );

    dim3 const* ptr_threads =
        NS(CudaKernelConfig_get_ptr_const_threads_per_block)( kernel_config );

    if( dbg_register_arg == SIXTRL_NULLPTR )
    {
        SIXTRL_ASSERT( ptr_blocks  != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( ptr_threads != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( !NS(KernelConfig_needs_update)( kernel_config ) );
        SIXTRL_ASSERT( NS(KernelConfig_get_arch_id)( kernel_config ) ==
            NS(ARCHITECTURE_CUDA) );

        SIXTRL_ASSERT( particles_arg != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Argument_get_arch_id)( particles_arg ) ==
                   NS(ARCHITECTURE_CUDA) );

        SIXTRL_ASSERT( NS(Argument_uses_cobjects_buffer)( particles_arg ) );
        SIXTRL_ASSERT( NS(Argument_get_const_cobjects_buffer)(
            particles_arg ) != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( beam_elem_arg != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Argument_get_arch_id)( beam_elem_arg ) ==
                   NS(ARCHITECTURE_CUDA) );

        SIXTRL_ASSERT( NS(Argument_uses_cobjects_buffer)( beam_elem_arg ) );
        SIXTRL_ASSERT( NS(Argument_get_const_cobjects_buffer)(
            beam_elem_arg ) != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( config_buffer_arg != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Argument_get_arch_id)( config_buffer_arg ) ==
                   NS(ARCHITECTURE_CUDA) );

        SIXTRL_ASSERT( NS(Argument_uses_cobjects_buffer)( config_buffer_arg ) );
        SIXTRL_ASSERT( NS(Argument_get_const_cobjects_buffer)(
            config_buffer_arg ) != SIXTRL_NULLPTR );

        SIXTRL_ASSERT(
            NS(Argument_get_cobjects_buffer_slot_size)( particles_arg ) ==
            NS(Argument_get_cobjects_buffer_slot_size)( beam_elem_arg ) );

        SIXTRL_ASSERT(
            NS(Argument_get_cobjects_buffer_slot_size)( config_buffer_arg ) ==
            NS(Argument_get_cobjects_buffer_slot_size)( particles_arg ) );

        NS(Track_track_elem_by_elem_until_turn_cuda)<<< *ptr_blocks, *ptr_threads >>>(
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                particles_arg ), pset_index,
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                beam_elem_arg ),
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                config_buffer_arg ), elem_by_elem_config_index,
            until_turn_elem_by_elem,
            NS(Argument_get_cobjects_buffer_slot_size)( particles_arg ) );
    }
    else if( ( ptr_blocks != SIXTRL_NULLPTR ) &&
        ( ptr_threads != SIXTRL_NULLPTR ) &&
        ( NS(KernelConfig_get_arch_id)( kernel_config ) ==
                NS(ARCHITECTURE_CUDA) ) &&
        ( !NS(KernelConfig_needs_update)( kernel_config ) ) )
    {
        NS(Track_track_elem_by_elem_until_turn_cuda_debug)<<< *ptr_blocks, *ptr_threads >>>(
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                particles_arg ), pset_index,
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                beam_elem_arg ),
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                config_buffer_arg ), elem_by_elem_config_index,
            until_turn_elem_by_elem,
            NS(Argument_get_cobjects_buffer_slot_size)( particles_arg ),
            NS(CudaArgument_get_cuda_arg_buffer_as_debugging_register_begin)(
                dbg_register_arg ) );
    }
}

void NS(Track_particles_line_cuda_wrapper)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT particles_arg,
    NS(buffer_size_t) const pset_index,
    NS(CudaArgument)* SIXTRL_RESTRICT beam_elem_arg,
    NS(buffer_size_t) const be_begin_idx,
    NS(buffer_size_t) const be_end_idx,
    bool const finish_turn,
    NS(CudaArgument)* SIXTRL_RESTRICT dbg_register_arg )
{
    dim3 const* ptr_blocks = NS(CudaKernelConfig_get_ptr_const_blocks)(
        kernel_config );

    dim3 const* ptr_threads =
        NS(CudaKernelConfig_get_ptr_const_threads_per_block)( kernel_config );

    if( dbg_register_arg == SIXTRL_NULLPTR )
    {
        SIXTRL_ASSERT( ptr_blocks  != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( ptr_threads != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( !NS(KernelConfig_needs_update)( kernel_config ) );
        SIXTRL_ASSERT( NS(KernelConfig_get_arch_id)( kernel_config ) ==
            NS(ARCHITECTURE_CUDA) );

        SIXTRL_ASSERT( particles_arg != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Argument_get_arch_id)( particles_arg ) ==
                   NS(ARCHITECTURE_CUDA) );

        SIXTRL_ASSERT( NS(Argument_uses_cobjects_buffer)( particles_arg ) );
        SIXTRL_ASSERT( NS(Argument_get_const_cobjects_buffer)(
            particles_arg ) != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( beam_elem_arg != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Argument_get_arch_id)( beam_elem_arg ) ==
                   NS(ARCHITECTURE_CUDA) );

        SIXTRL_ASSERT( NS(Argument_uses_cobjects_buffer)( beam_elem_arg ) );
        SIXTRL_ASSERT( NS(Argument_get_const_cobjects_buffer)(
            beam_elem_arg ) != SIXTRL_NULLPTR );

        SIXTRL_ASSERT(
            NS(Argument_get_cobjects_buffer_slot_size)( particles_arg ) ==
            NS(Argument_get_cobjects_buffer_slot_size)( beam_elem_arg ) );

        NS(Track_particles_line_cuda)<<< *ptr_blocks, *ptr_threads >>>(
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                particles_arg ), pset_index,
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                beam_elem_arg ),
                be_begin_idx, be_end_idx, finish_turn,
            NS(Argument_get_cobjects_buffer_slot_size)( particles_arg ) );
    }
    else if( ( ptr_blocks != SIXTRL_NULLPTR ) &&
        ( ptr_threads != SIXTRL_NULLPTR ) &&
        ( NS(KernelConfig_get_arch_id)( kernel_config ) ==
                NS(ARCHITECTURE_CUDA) ) &&
        ( !NS(KernelConfig_needs_update)( kernel_config ) ) )
    {
        NS(Track_particles_line_cuda_debug)<<< *ptr_blocks, *ptr_threads >>>(
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                particles_arg ), pset_index,
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                beam_elem_arg ),
            be_begin_idx, be_end_idx, finish_turn,
            NS(Argument_get_cobjects_buffer_slot_size)( particles_arg ),
            NS(CudaArgument_get_cuda_arg_buffer_as_debugging_register_begin)(
                dbg_register_arg ) );
    }

    return;
}

void NS(BeamMonitor_assign_out_buffer_from_offset_cuda_wrapper)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT beam_elem_arg,
    NS(CudaArgument)* SIXTRL_RESTRICT output_arg,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t) const out_buffer_offset_index,
    NS(CudaArgument)* SIXTRL_RESTRICT dbg_register_arg )
{
    dim3 const* ptr_blocks = NS(CudaKernelConfig_get_ptr_const_blocks)(
        kernel_config );

    dim3 const* ptr_threads =
        NS(CudaKernelConfig_get_ptr_const_threads_per_block)( kernel_config );

    if( dbg_register_arg == SIXTRL_NULLPTR )
    {
        SIXTRL_ASSERT( ptr_blocks  != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( ptr_threads != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( !NS(KernelConfig_needs_update)( kernel_config ) );
        SIXTRL_ASSERT( NS(KernelConfig_get_arch_id)( kernel_config ) ==
            NS(ARCHITECTURE_CUDA) );

        SIXTRL_ASSERT( output_arg != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Argument_get_arch_id)( output_arg ) ==
                   NS(ARCHITECTURE_CUDA) );

        SIXTRL_ASSERT( NS(Argument_uses_cobjects_buffer)( output_arg ) );
        SIXTRL_ASSERT( NS(Argument_get_const_cobjects_buffer)(
            output_arg ) != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( beam_elem_arg != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Argument_get_arch_id)( beam_elem_arg ) ==
                   NS(ARCHITECTURE_CUDA) );

        SIXTRL_ASSERT( NS(Argument_uses_cobjects_buffer)( beam_elem_arg ) );
        SIXTRL_ASSERT( NS(Argument_get_const_cobjects_buffer)(
            beam_elem_arg ) != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( min_turn_id >= ( NS(particle_index_t) )0u );
        SIXTRL_ASSERT( out_buffer_offset_index <
            NS(Buffer_get_num_of_objects)(
                NS(Argument_get_const_cobjects_buffer)( output_arg ) ) );

        SIXTRL_ASSERT( NS(Argument_get_cobjects_buffer_slot_size)( output_arg )
            > ( NS(buffer_size_t) )0u );

        NS(BeamMonitor_assign_out_buffer_from_offset_cuda)<<<
            *ptr_blocks, *ptr_threads >>>(
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                beam_elem_arg ),
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                output_arg ), min_turn_id, out_buffer_offset_index,
            NS(Argument_get_cobjects_buffer_slot_size)( output_arg ) );


    }
    else if( ( ptr_blocks != SIXTRL_NULLPTR ) &&
        ( ptr_threads != SIXTRL_NULLPTR ) &&
        ( NS(KernelConfig_get_arch_id)( kernel_config ) ==
                NS(ARCHITECTURE_CUDA) ) &&
        ( !NS(KernelConfig_needs_update)( kernel_config ) ) )
    {
        NS(BeamMonitor_assign_out_buffer_from_offset_cuda_debug)<<<
            *ptr_blocks, *ptr_threads >>>(
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                beam_elem_arg ),
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                output_arg ), min_turn_id, out_buffer_offset_index,
            NS(Argument_get_cobjects_buffer_slot_size)( output_arg ),
            NS(CudaArgument_get_cuda_arg_buffer_as_debugging_register_begin)(
                dbg_register_arg ) );
    }
}

void NS(ElemByElemConfig_assign_out_buffer_from_offset_cuda_wrapper)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT config_buffer_arg,
    NS(buffer_size_t) const elem_by_elem_config_index,
    NS(CudaArgument)* SIXTRL_RESTRICT output_arg,
    NS(buffer_size_t) const out_buffer_offset_index,
    NS(CudaArgument)* SIXTRL_RESTRICT dbg_register_arg )
{
    dim3 const* ptr_blocks = NS(CudaKernelConfig_get_ptr_const_blocks)(
        kernel_config );

    dim3 const* ptr_threads =
        NS(CudaKernelConfig_get_ptr_const_threads_per_block)( kernel_config );

    if( dbg_register_arg != SIXTRL_NULLPTR )
    {
        SIXTRL_ASSERT( ptr_blocks  != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( ptr_threads != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( !NS(KernelConfig_needs_update)( kernel_config ) );
        SIXTRL_ASSERT( NS(KernelConfig_get_arch_id)( kernel_config ) ==
            NS(ARCHITECTURE_CUDA) );

        SIXTRL_ASSERT( output_arg != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Argument_get_arch_id)( output_arg ) ==
                   NS(ARCHITECTURE_CUDA) );

        SIXTRL_ASSERT( NS(Argument_uses_cobjects_buffer)( output_arg ) );
        SIXTRL_ASSERT( NS(Argument_get_const_cobjects_buffer)(
            output_arg ) != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( config_buffer_arg != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Argument_get_arch_id)( config_buffer_arg ) ==
                   NS(ARCHITECTURE_CUDA) );

        SIXTRL_ASSERT( NS(Argument_uses_cobjects_buffer)( config_buffer_arg ) );
        SIXTRL_ASSERT( NS(Argument_get_const_cobjects_buffer)(
            config_buffer_arg ) != SIXTRL_NULLPTR );

        SIXTRL_ASSERT(
            NS(Argument_get_cobjects_buffer_slot_size)( config_buffer_arg ) ==
            NS(Argument_get_cobjects_buffer_slot_size)( output_arg ) );

        SIXTRL_ASSERT( out_buffer_offset_index <
            NS(Buffer_get_num_of_objects)(
                NS(Argument_get_const_cobjects_buffer)( output_arg ) ) );

        SIXTRL_ASSERT( NS(Argument_get_cobjects_buffer_slot_size)( output_arg )
            > ( NS(buffer_size_t) )0u );

        NS(ElemByElemConfig_assign_out_buffer_from_offset_cuda)<<<
            *ptr_blocks, *ptr_threads >>>(
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                config_buffer_arg ), elem_by_elem_config_index,
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                output_arg ), out_buffer_offset_index,
            NS(Argument_get_cobjects_buffer_slot_size)( output_arg ) );


    }
    else if( ( ptr_blocks != SIXTRL_NULLPTR ) &&
        ( ptr_threads != SIXTRL_NULLPTR ) &&
        ( NS(KernelConfig_get_arch_id)( kernel_config ) ==
                NS(ARCHITECTURE_CUDA) ) &&
        ( !NS(KernelConfig_needs_update)( kernel_config ) ) )
    {
        NS(ElemByElemConfig_assign_out_buffer_from_offset_cuda_debug)<<<
            *ptr_blocks, *ptr_threads >>>(
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                config_buffer_arg ), elem_by_elem_config_index,
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                output_arg ), out_buffer_offset_index,
            NS(Argument_get_cobjects_buffer_slot_size)( output_arg ),
            NS(CudaArgument_get_cuda_arg_buffer_as_debugging_register_begin)(
                dbg_register_arg ) );
    }
}

void NS(AssignAddressItem_process_managed_buffer_cuda_wrapper)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT assign_buffer_arg,
    NS(CudaArgument)* SIXTRL_RESTRICT dest_buffer_arg,
    NS(buffer_size_t) const dest_buffer_id,
    NS(CudaArgument)* SIXTRL_RESTRICT src_buffer_arg,
    NS(buffer_size_t) const src_buffer_id )
{
    dim3 const* ptr_blocks =
        NS(CudaKernelConfig_get_ptr_const_blocks)( kernel_config );

    dim3 const* ptr_threads =
            NS(CudaKernelConfig_get_ptr_const_threads_per_block)( kernel_config );

    NS(buffer_size_t) const assign_slot_size =  NS(Buffer_get_slot_size)(
        NS(Argument_get_const_cobjects_buffer)( dest_buffer_arg ) );

    NS(buffer_size_t) const dest_slot_size = NS(Buffer_get_slot_size)(
        NS(Argument_get_const_cobjects_buffer)( dest_buffer_arg ) );

    NS(buffer_size_t) const src_slot_size = NS(Buffer_get_slot_size)(
        NS(Argument_get_const_cobjects_buffer)( src_buffer_arg ) );

    SIXTRL_ASSERT( ptr_blocks  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ptr_threads != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( !NS(KernelConfig_needs_update)( kernel_config ) );
    SIXTRL_ASSERT( NS(KernelConfig_get_arch_id)( kernel_config ) ==
        NS(ARCHITECTURE_CUDA) );

    SIXTRL_ASSERT( assign_buffer_arg != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(Argument_get_arch_id)( assign_buffer_arg ) ==
               NS(ARCHITECTURE_CUDA) );

    SIXTRL_ASSERT( NS(Argument_uses_cobjects_buffer)( assign_buffer_arg ) );
    SIXTRL_ASSERT( NS(Argument_get_const_cobjects_buffer)(
        assign_buffer_arg ) != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( assign_slot_size > ( NS(buffer_size_t) )0 );


    SIXTRL_ASSERT( dest_buffer_arg != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(Argument_get_arch_id)( dest_buffer_arg ) ==
               NS(ARCHITECTURE_CUDA) );
    SIXTRL_ASSERT( NS(Argument_uses_cobjects_buffer)( dest_buffer_arg ) );
    SIXTRL_ASSERT( NS(Argument_get_const_cobjects_buffer)(
        dest_buffer_arg ) != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( NS(Buffer_get_num_of_objects)(
        NS(Argument_get_const_cobjects_buffer)( dest_buffer_arg ) ) >
            dest_buffer_id );
    SIXTRL_ASSERT( dest_slot_size > ( NS(buffer_size_t) )0u );


    SIXTRL_ASSERT( src_buffer_arg != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(Argument_get_arch_id)( src_buffer_arg ) ==
               NS(ARCHITECTURE_CUDA) );

    SIXTRL_ASSERT( NS(Argument_uses_cobjects_buffer)( src_buffer_arg ) );
    SIXTRL_ASSERT( NS(Argument_get_const_cobjects_buffer)(
        src_buffer_arg ) != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( NS(Buffer_get_num_of_objects)(
        NS(Argument_get_const_cobjects_buffer)( src_buffer_arg ) ) >
            src_buffer_id );
    SIXTRL_ASSERT( src_slot_size > ( NS(buffer_size_t) )0u );

    NS(AssignAddressItem_process_managed_buffer_cuda)<<< *ptr_blocks,
        *ptr_threads >>>(
        NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
            assign_buffer_arg ), assign_slot_size,
        NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
            dest_buffer_arg ), dest_slot_size, dest_buffer_id,
        NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
            src_buffer_arg ), src_slot_size, src_buffer_id );
}

void NS(Particles_buffer_store_all_addresses_cuda_wrapper)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT particles_addresses_arg,
    NS(CudaArgument)* SIXTRL_RESTRICT particles_arg,
    NS(CudaArgument)* SIXTRL_RESTRICT dbg_register_arg )
{
    dim3 const* ptr_blocks = NS(CudaKernelConfig_get_ptr_const_blocks)(
        kernel_config );

    dim3 const* ptr_threads =
        NS(CudaKernelConfig_get_ptr_const_threads_per_block)( kernel_config );

    if( dbg_register_arg == SIXTRL_NULLPTR )
    {
        SIXTRL_ASSERT( ptr_blocks  != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( ptr_threads != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( !NS(KernelConfig_needs_update)( kernel_config ) );
        SIXTRL_ASSERT( NS(KernelConfig_get_arch_id)( kernel_config ) ==
            NS(ARCHITECTURE_CUDA) );

        SIXTRL_ASSERT( particles_arg != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Argument_get_arch_id)( particles_arg ) ==
                   NS(ARCHITECTURE_CUDA) );

        SIXTRL_ASSERT( NS(Argument_uses_cobjects_buffer)( particles_arg ) );
        SIXTRL_ASSERT( NS(Argument_get_const_cobjects_buffer)(
            particles_arg ) != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( NS(Argument_get_cobjects_buffer_slot_size)(
            particles_arg ) > ( NS(buffer_size_t) )0u );

        SIXTRL_ASSERT( particles_addresses_arg != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Argument_get_arch_id)( particles_addresses_arg ) ==
                   NS(ARCHITECTURE_CUDA) );

        SIXTRL_ASSERT( NS(Argument_uses_cobjects_buffer)(
            particles_addresses_arg ) );

        SIXTRL_ASSERT( NS(Argument_get_const_cobjects_buffer)(
            particles_addresses_arg ) != SIXTRL_NULLPTR );

        NS(Particles_buffer_store_all_addresses_cuda)<<<
            *ptr_blocks, *ptr_threads >>>(
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                particles_addresses_arg ),
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                particles_arg ),
            NS(Argument_get_cobjects_buffer_slot_size)( particles_arg ) );


    }
    else if( ( ptr_blocks != SIXTRL_NULLPTR ) &&
        ( ptr_threads != SIXTRL_NULLPTR ) &&
        ( NS(KernelConfig_get_arch_id)( kernel_config ) ==
                NS(ARCHITECTURE_CUDA) ) &&
        ( !NS(KernelConfig_needs_update)( kernel_config ) ) )
    {
        NS(Particles_buffer_store_all_addresses_cuda_debug)<<<
            *ptr_blocks, *ptr_threads >>>(
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                particles_addresses_arg ),
            NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
                particles_arg ),
            NS(Argument_get_cobjects_buffer_slot_size)( particles_arg ),
            NS(CudaArgument_get_cuda_arg_buffer_as_debugging_register_begin)(
                dbg_register_arg ) );
    }

    ::cudaError_t const err = ::cudaDeviceSynchronize();
    SIXTRL_ASSERT( err == ::cudaSuccess );

    ( void )err;
}

/* end: sixtracklib/cuda/wrappers/track_job_wrappers.cu */
