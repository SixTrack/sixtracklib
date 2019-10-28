#ifndef SIXTRACKLIB_COMMON_TRACK_TRACK_KERNEL_OPTIMIZED_IMPL_C99_H__
#define SIXTRACKLIB_COMMON_TRACK_TRACK_KERNEL_OPTIMIZED_IMPL_C99_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/debug_register.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/internal/particles_defines.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/track/definitions.h"
    #include "sixtracklib/common/track/track.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, Host */

SIXTRL_STATIC SIXTRL_FN NS(track_status_t)
NS(Track_particles_until_turn_opt_kernel_impl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const particle_set_index,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    NS(particle_index_t) const until_turn, NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(track_status_t)
NS(Track_particles_until_turn_debug_opt_kernel_impl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const particle_set_index,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    NS(particle_index_t) const until_turn, NS(buffer_size_t) const slot_size,
    SIXTRL_ARGPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flags );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(track_status_t)
NS(Track_particles_elem_by_elem_until_turn_opt_kernel_impl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const particle_set_index,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig) *const
        SIXTRL_RESTRICT elem_by_elem_config,
    NS(particle_index_t) const until_turn, NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(track_status_t)
NS(Track_particles_elem_by_elem_until_turn_debug_opt_kernel_impl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const particle_set_index,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig) *const
        SIXTRL_RESTRICT elem_by_elem_config,
    NS(particle_index_t) const until_turn, NS(buffer_size_t) const slot_size,
    SIXTRL_ARGPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flags );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(track_status_t)
NS(Track_particles_line_opt_kernel_impl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const particle_set_index,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    NS(buffer_size_t) const belem_begin_id,
    NS(buffer_size_t) const belem_end_id,
    bool const finish_turn, NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(track_status_t)
NS(Track_particles_line_debug_opt_kernel_impl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const particle_set_index,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    NS(buffer_size_t) const belem_begin_id,
    NS(buffer_size_t) const belem_end_id,
    bool const finish_turn, NS(buffer_size_t) const slot_size,
    SIXTRL_ARGPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flags );

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

/* ************************************************************************* */
/* Inline functions implementation */
/* ************************************************************************* */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, Host */

SIXTRL_INLINE NS(track_status_t)
NS(Track_particles_until_turn_opt_kernel_impl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const particle_set_index,
    NS(particle_num_elements_t) pidx, NS(particle_num_elements_t) const stride,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    NS(particle_index_t) const until_turn, NS(buffer_size_t) const slot_size )
{
    typedef NS(particle_num_elements_t) nelements_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesGenericAddr)* ptr_particles_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* pset_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*  be_iter_t;
    typedef NS(particle_real_t) real_t;
    typedef NS(particle_index_t) index_t;

    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) particles;

    SIXTRL_PARTICLE_DATAPTR_DEC real_t reals[] =
    {
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0
    };

    SIXTRL_PARTICLE_DATAPTR_DEC index_t indices[] =
    {
        ( index_t )0, ( index_t )0, ( index_t )0, ( index_t )0
    };

    pset_iter_t pset_it = NS(ManagedBuffer_get_object)(
        pbuffer, particle_set_index, slot_size );

    ptr_particles_t in_particles = ( ptr_particles_t )( uintptr_t
        )NS(Object_get_begin_addr)( pset_it );

    nelements_t const num_particles = (
        ( NS(Object_get_type_id)( pset_it ) == NS(OBJECT_TYPE_PARTICLE ) ) &&
        ( in_particles != SIXTRL_NULLPTR ) )
        ? in_particles->num_particles : ( nelements_t )0u;

    be_iter_t belem_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
        belem_buffer, slot_size );

    be_iter_t belem_end = NS(ManagedBuffer_get_const_objects_index_end)(
        belem_buffer, slot_size );

    NS(track_status_t) status = SIXTRL_TRACK_SUCCESS;

    NS(Particles_init_from_flat_arrays)(
        &particles, ( nelements_t )1u, &reals[ 0 ], &indices[ 0 ] );

    SIXTRL_ASSERT( slot_size > ( SIXTRL_UINT64_T )0u );
    SIXTRL_ASSERT( in_particles != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( pset_it != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( particle_set_index >= ( nelements_t )0u );
    SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
        pbuffer, slot_size ) > particle_set_index );

    SIXTRL_ASSERT( belem_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( belem_end   != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( uintptr_t )belem_end >= ( uintptr_t )belem_begin );
    SIXTRL_ASSERT( until_turn >= ( NS(particle_index_t) )0u );
    SIXTRL_ASSERT( pidx >=  ( nelements_t )0u );
    SIXTRL_ASSERT( stride > ( nelements_t )0u );

    for( ; pidx < num_particles ; pidx += stride )
    {
        status |= NS(Particles_copy_from_generic_addr_data)(
            &particles, 0u, in_particles, pidx );

        status |= NS(Track_particle_until_turn_objs)(
            &particles, 0, belem_begin, belem_end, until_turn );

        status |= NS(Particles_copy_to_generic_addr_data)(
            in_particles, pidx, &particles, 0u );
    }

    return status;
}

SIXTRL_INLINE NS(track_status_t)
NS(Track_particles_until_turn_debug_opt_kernel_impl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const particle_set_index,
    NS(particle_num_elements_t) pidx, NS(particle_num_elements_t) const stride,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    NS(particle_index_t) const until_turn, NS(buffer_size_t) const slot_size,
    SIXTRL_ARGPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flags )
{
    typedef NS(particle_num_elements_t) nelements_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesGenericAddr)* ptr_particles_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* pset_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const* be_iter_t;
    typedef NS(particle_real_t) real_t;
    typedef NS(particle_index_t) index_t;

    NS(track_status_t)  status = SIXTRL_TRACK_STATUS_GENERAL_FAILURE;
    NS(arch_debugging_t) flags = SIXTRL_ARCH_DEBUGGING_MIN_FLAG;

    NS(arch_debugging_t) const PARTICLE_IDX_ILLEGAL_FLAG        = flags;
    NS(arch_debugging_t) const PBUFFER_NULL_FLAG                = flags <<  1u;
    NS(arch_debugging_t) const BELEM_BUFFER_NULL_FLAG           = flags <<  2u;
    NS(arch_debugging_t) const SLOT_SIZE_ILLEGAL_FLAG           = flags <<  3u;
    NS(arch_debugging_t) const PSET_INDEX_ILLEGAL_FLAG          = flags <<  4u;
    NS(arch_debugging_t) const PBUFFER_REQUIRES_REMAP_FLAG      = flags <<  5u;
    NS(arch_debugging_t) const BELEM_BUFFER_REQUIRES_REMAP_FLAG = flags <<  6u;
    NS(arch_debugging_t) const LINE_BOUNDARIES_ILLEGAL_FLAG     = flags <<  7u;
    NS(arch_debugging_t) const UNTIL_TURN_ILLEGAL_FLAG          = flags <<  8u;

    flags = ( NS(arch_debugging_t) )0u;

    if( ( pbuffer != SIXTRL_NULLPTR ) && ( belem_buffer != SIXTRL_NULLPTR ) &&
        ( slot_size > ( SIXTRL_UINT64_T )0u ) &&
        ( pidx >= ( nelements_t )0u ) && ( stride > (  nelements_t )0u ) )
    {
        pset_iter_t pset_it = NS(ManagedBuffer_get_object)(
            pbuffer, particle_set_index, slot_size );

        ptr_particles_t in_particles = ( ptr_particles_t )( uintptr_t
            )NS(Object_get_begin_addr)( pset_it );

        nelements_t const num_particles = (
            ( NS(Object_get_type_id)( pset_it ) == NS(OBJECT_TYPE_PARTICLE ) ) &&
            ( in_particles != SIXTRL_NULLPTR ) )
            ? in_particles->num_particles : ( nelements_t )0u;

        be_iter_t belem_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
            belem_buffer, slot_size );

        be_iter_t belem_end = NS(ManagedBuffer_get_const_objects_index_end)(
            belem_buffer, slot_size );

        if( ( !NS(ManagedBuffer_needs_remapping)( pbuffer, slot_size ) ) &&
            ( !NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) ) &&
            ( in_particles != SIXTRL_NULLPTR ) &&
            ( belem_begin != SIXTRL_NULLPTR ) &&
            ( ( uintptr_t )belem_begin <= ( uintptr_t )belem_end ) &&
            ( until_turn >= ( NS(particle_index_t) )0 ) )
        {
            SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) particles;

            SIXTRL_PARTICLE_DATAPTR_DEC real_t reals[] =
            {
                ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
                ( real_t )0.0, ( real_t )0.0,
                ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
                ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
                ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
                ( real_t )0.0, ( real_t )0.0, ( real_t )0.0
            };

            SIXTRL_PARTICLE_DATAPTR_DEC index_t indices[] =
            {
                ( index_t )0, ( index_t )0, ( index_t )0, ( index_t )0
            };

            status = SIXTRL_TRACK_SUCCESS;

            NS(Particles_init_from_flat_arrays)(
                &particles, ( nelements_t )1u, &reals[ 0 ], &indices[ 0 ] );

            for( ; pidx < num_particles ; pidx += stride )
            {
                status |= NS(Particles_copy_from_generic_addr_data)(
                    &particles, 0u, in_particles, pidx );

                status |= NS(Track_particle_until_turn_objs)(
                    &particles, 0, belem_begin, belem_end, until_turn );

                status |= NS(Particles_copy_to_generic_addr_data)(
                    in_particles, pidx, &particles, 0u );
            }
        }
        else
        {
            if( NS(ManagedBuffer_needs_remapping)( pbuffer, slot_size ) )
                flags |= PBUFFER_REQUIRES_REMAP_FLAG;

            if( NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) )
                flags |= BELEM_BUFFER_REQUIRES_REMAP_FLAG;

            if( ( in_particles == SIXTRL_NULLPTR ) ||
                ( NS(ManagedBuffer_get_num_objects)( pbuffer, slot_size ) <=
                  particle_set_index ) )
                flags |= PSET_INDEX_ILLEGAL_FLAG;

            if( ( belem_begin == SIXTRL_NULLPTR ) ||
                ( ( uintptr_t )belem_begin > ( uintptr_t )belem_end ) )
                flags |= LINE_BOUNDARIES_ILLEGAL_FLAG;

            if( until_turn < ( NS(particle_index_t) )0 )
                flags |= UNTIL_TURN_ILLEGAL_FLAG;
        }
    }
    else
    {
        if( pbuffer == SIXTRL_NULLPTR )      flags |= PBUFFER_NULL_FLAG;
        if( belem_buffer == SIXTRL_NULLPTR ) flags |= BELEM_BUFFER_NULL_FLAG;
        if( slot_size == 0u )                flags |= SLOT_SIZE_ILLEGAL_FLAG;
        if( pidx < ( nelements_t )0u )       flags |= PARTICLE_IDX_ILLEGAL_FLAG;
        if( stride <= ( nelements_t )0u )    flags |= PARTICLE_IDX_ILLEGAL_FLAG;
    }

    if( status != SIXTRL_TRACK_SUCCESS )
    {
        flags = NS(DebugReg_store_arch_status)( flags, status );
    }

    if( ptr_status_flags != SIXTRL_NULLPTR )
    {
        *ptr_status_flags = flags;
    }

    return status;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(track_status_t)
NS(Track_particles_elem_by_elem_until_turn_opt_kernel_impl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const particle_set_index,
    NS(particle_num_elements_t) pidx, NS(particle_num_elements_t) const stride,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig) *const
        SIXTRL_RESTRICT elem_by_elem_config,
    NS(particle_index_t) const until_turn, NS(buffer_size_t) const slot_size )
{
    typedef NS(particle_num_elements_t) nelements_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesGenericAddr)* ptr_particles_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* pset_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*  be_iter_t;
    typedef NS(particle_real_t) real_t;
    typedef NS(particle_index_t) index_t;

    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) particles;

    SIXTRL_PARTICLE_DATAPTR_DEC real_t reals[] =
    {
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0
    };

    SIXTRL_PARTICLE_DATAPTR_DEC index_t indices[] =
    {
        ( index_t )0, ( index_t )0, ( index_t )0, ( index_t )0
    };

    pset_iter_t pset_it = NS(ManagedBuffer_get_object)(
        pbuffer, particle_set_index, slot_size );

    ptr_particles_t in_particles = ( ptr_particles_t )( uintptr_t
        )NS(Object_get_begin_addr)( pset_it );

    nelements_t const num_particles = (
        ( NS(Object_get_type_id)( pset_it ) == NS(OBJECT_TYPE_PARTICLE ) ) &&
        ( in_particles != SIXTRL_NULLPTR ) )
        ? in_particles->num_particles : ( nelements_t )0u;

    be_iter_t belem_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
        belem_buffer, slot_size );

    be_iter_t belem_end = NS(ManagedBuffer_get_const_objects_index_end)(
        belem_buffer, slot_size );

    NS(track_status_t) status = SIXTRL_TRACK_SUCCESS;

    NS(Particles_init_from_flat_arrays)(
        &particles, ( nelements_t )1u, &reals[ 0 ], &indices[ 0 ] );

    SIXTRL_ASSERT( slot_size > ( SIXTRL_UINT64_T )0u );
    SIXTRL_ASSERT( in_particles != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( pset_it != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( particle_set_index >= ( nelements_t )0u );
    SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
        pbuffer, slot_size ) > particle_set_index );

    SIXTRL_ASSERT( elem_by_elem_config != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( belem_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( belem_end   != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( uintptr_t )belem_end >= ( uintptr_t )belem_begin );
    SIXTRL_ASSERT( until_turn >= ( NS(particle_index_t) )0u );
    SIXTRL_ASSERT( pidx >=  ( nelements_t )0u );
    SIXTRL_ASSERT( stride > ( nelements_t )0u );

    for( ; pidx < num_particles ; pidx += stride )
    {
        status |= NS(Particles_copy_from_generic_addr_data)(
            &particles, 0u, in_particles, pidx );

        status |= NS(Track_particle_element_by_element_until_turn_objs)(
            &particles, 0, elem_by_elem_config, belem_begin, belem_end,
                until_turn );

        status |= NS(Particles_copy_to_generic_addr_data)(
            in_particles, pidx, &particles, 0u );
    }

    return status;
}

SIXTRL_INLINE NS(track_status_t)
NS(Track_particles_elem_by_elem_until_turn_debug_opt_kernel_impl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const particle_set_index,
    NS(particle_num_elements_t) pidx, NS(particle_num_elements_t) const stride,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig) *const
        SIXTRL_RESTRICT elem_by_elem_config,
    NS(particle_index_t) const until_turn, NS(buffer_size_t) const slot_size,
    SIXTRL_ARGPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flags )
{
    typedef NS(particle_num_elements_t) nelements_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesGenericAddr)* ptr_particles_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* pset_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const* be_iter_t;
    typedef NS(particle_real_t) real_t;
    typedef NS(particle_index_t) index_t;

    NS(track_status_t)  status = SIXTRL_TRACK_STATUS_GENERAL_FAILURE;
    NS(arch_debugging_t) flags = SIXTRL_ARCH_DEBUGGING_MIN_FLAG;

    NS(arch_debugging_t) const PARTICLE_IDX_ILLEGAL_FLAG        = flags;
    NS(arch_debugging_t) const PBUFFER_NULL_FLAG                = flags <<  1u;
    NS(arch_debugging_t) const BELEM_BUFFER_NULL_FLAG           = flags <<  2u;
    NS(arch_debugging_t) const SLOT_SIZE_ILLEGAL_FLAG           = flags <<  3u;
    NS(arch_debugging_t) const PSET_INDEX_ILLEGAL_FLAG          = flags <<  4u;
    NS(arch_debugging_t) const PBUFFER_REQUIRES_REMAP_FLAG      = flags <<  5u;
    NS(arch_debugging_t) const BELEM_BUFFER_REQUIRES_REMAP_FLAG = flags <<  6u;
    NS(arch_debugging_t) const LINE_BOUNDARIES_ILLEGAL_FLAG     = flags <<  7u;
    NS(arch_debugging_t) const UNTIL_TURN_ILLEGAL_FLAG          = flags <<  8u;
    NS(arch_debugging_t) const ELEM_BY_ELEM_CONFIG_ILLEGAL_FLAG = flags <<  9u;

    flags = ( NS(arch_debugging_t) )0u;

    if( ( pbuffer != SIXTRL_NULLPTR ) && ( belem_buffer != SIXTRL_NULLPTR ) &&
        ( slot_size > ( SIXTRL_UINT64_T )0u ) &&
        ( elem_by_elem_config != SIXTRL_NULLPTR ) &&
        ( pidx >= ( nelements_t )0u ) && ( stride > ( nelements_t )0u ) )
    {
        pset_iter_t pset_it = NS(ManagedBuffer_get_object)(
            pbuffer, particle_set_index, slot_size );

        ptr_particles_t in_particles = ( ptr_particles_t )( uintptr_t
            )NS(Object_get_begin_addr)( pset_it );

        nelements_t const num_particles = (
            ( NS(Object_get_type_id)( pset_it ) == NS(OBJECT_TYPE_PARTICLE ) ) &&
            ( in_particles != SIXTRL_NULLPTR ) )
            ? in_particles->num_particles : ( nelements_t )0u;

        be_iter_t belem_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
            belem_buffer, slot_size );

        be_iter_t belem_end = NS(ManagedBuffer_get_const_objects_index_end)(
            belem_buffer, slot_size );

        if( ( !NS(ManagedBuffer_needs_remapping)( pbuffer, slot_size ) ) &&
            ( !NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) ) &&
            ( in_particles != SIXTRL_NULLPTR ) && ( belem_begin != SIXTRL_NULLPTR ) &&
            ( ( uintptr_t )belem_begin <= ( uintptr_t )belem_end ) &&
            ( until_turn >= ( NS(particle_index_t) )0 ) )
        {
            SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) particles;

            SIXTRL_PARTICLE_DATAPTR_DEC real_t reals[] =
            {
                ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
                ( real_t )0.0, ( real_t )0.0,
                ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
                ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
                ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
                ( real_t )0.0, ( real_t )0.0, ( real_t )0.0
            };

            SIXTRL_PARTICLE_DATAPTR_DEC index_t indices[] =
            {
                ( index_t )0, ( index_t )0, ( index_t )0, ( index_t )0
            };

            NS(Particles_init_from_flat_arrays)(
                &particles, ( nelements_t )1u, &reals[ 0 ], &indices[ 0 ] );

            status = SIXTRL_TRACK_SUCCESS;

            for( ; pidx < num_particles ; pidx += stride )
            {
                status |= NS(Particles_copy_from_generic_addr_data)(
                    &particles, 0u, in_particles, pidx );

                status |= NS(Track_particle_element_by_element_until_turn_objs)(
                    &particles, 0, elem_by_elem_config, belem_begin, belem_end,
                        until_turn );

                status |= NS(Particles_copy_to_generic_addr_data)(
                    in_particles, pidx, &particles, 0u );
            }
        }
        else
        {
            if( NS(ManagedBuffer_needs_remapping)( pbuffer, slot_size ) )
                flags |= PBUFFER_REQUIRES_REMAP_FLAG;

            if( NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) )
                flags |= BELEM_BUFFER_REQUIRES_REMAP_FLAG;

            if( ( in_particles == SIXTRL_NULLPTR ) ||
                ( NS(ManagedBuffer_get_num_objects)( pbuffer, slot_size ) <=
                  particle_set_index ) )
                flags |= PSET_INDEX_ILLEGAL_FLAG;

            if( ( belem_begin == SIXTRL_NULLPTR ) ||
                ( ( uintptr_t )belem_begin > ( uintptr_t )belem_end ) )
                flags |= LINE_BOUNDARIES_ILLEGAL_FLAG;

            if( until_turn < ( NS(particle_index_t) )0 )
                flags |= UNTIL_TURN_ILLEGAL_FLAG;
        }
    }
    else
    {
        if( pbuffer == SIXTRL_NULLPTR )      flags |= PBUFFER_NULL_FLAG;
        if( belem_buffer == SIXTRL_NULLPTR ) flags |= BELEM_BUFFER_NULL_FLAG;
        if( slot_size == 0u )                flags |= SLOT_SIZE_ILLEGAL_FLAG;
        if( pidx < ( nelements_t )0u )       flags |= PARTICLE_IDX_ILLEGAL_FLAG;
        if( stride <= ( nelements_t )0u )    flags |= PARTICLE_IDX_ILLEGAL_FLAG;

        if( elem_by_elem_config == SIXTRL_NULLPTR )
                flags |= ELEM_BY_ELEM_CONFIG_ILLEGAL_FLAG;
    }

    if( status != SIXTRL_TRACK_SUCCESS )
    {
        flags = NS(DebugReg_store_arch_status)( flags, status );
    }

    if(  ptr_status_flags != SIXTRL_NULLPTR )
    {
        *ptr_status_flags = flags;
    }

    return status;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(track_status_t) NS(Track_particles_line_opt_kernel_impl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const particle_set_index,
    NS(particle_num_elements_t) pidx, NS(particle_num_elements_t) const stride,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    NS(buffer_size_t) const belem_begin_id,
    NS(buffer_size_t) const belem_end_id,
    bool const finish_turn, NS(buffer_size_t) const slot_size )
{
    typedef NS(particle_num_elements_t) nelements_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesGenericAddr)* ptr_particles_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* pset_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*  be_iter_t;
    typedef NS(particle_real_t) real_t;
    typedef NS(particle_index_t) index_t;

    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) particles;

    SIXTRL_PARTICLE_DATAPTR_DEC real_t reals[] =
    {
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0
    };

    SIXTRL_PARTICLE_DATAPTR_DEC index_t indices[] =
    {
        ( index_t )0, ( index_t )0, ( index_t )0, ( index_t )0
    };

    pset_iter_t pset_it = NS(ManagedBuffer_get_object)(
        pbuffer, particle_set_index, slot_size );

    ptr_particles_t in_particles = ( ptr_particles_t )( uintptr_t
        )NS(Object_get_begin_addr)( pset_it );

    nelements_t const num_particles = (
        ( NS(Object_get_type_id)( pset_it ) == NS(OBJECT_TYPE_PARTICLE ) ) &&
        ( in_particles != SIXTRL_NULLPTR ) )
        ? in_particles->num_particles : ( nelements_t )0u;

    be_iter_t belem_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
        belem_buffer, slot_size );

    be_iter_t belem_end = belem_begin;

    NS(track_status_t) status = SIXTRL_TRACK_SUCCESS;

    NS(Particles_init_from_flat_arrays)(
        &particles, ( nelements_t )1u, &reals[ 0 ], &indices[ 0 ] );

    SIXTRL_ASSERT( slot_size > ( SIXTRL_UINT64_T )0u );
    SIXTRL_ASSERT( in_particles != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( pset_it != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( particle_set_index >= ( nelements_t )0u );
    SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
        pbuffer, slot_size ) > particle_set_index );

    SIXTRL_ASSERT( belem_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( belem_begin_id <= belem_end_id );
    SIXTRL_ASSERT( belem_end_id <= NS(ManagedBuffer_get_num_objects)(
        belem_buffer, slot_size ) );

    SIXTRL_ASSERT( pidx >=  ( nelements_t )0u );
    SIXTRL_ASSERT( stride > ( nelements_t )0u );

    belem_begin = belem_begin + belem_begin_id;
    belem_end   = belem_end   + belem_end_id;

    for( ; pidx < num_particles ; pidx += stride )
    {
        status |= NS(Particles_copy_from_generic_addr_data)(
            &particles, 0u, in_particles, pidx );

        status |= NS(Track_particle_line_objs)(
            &particles, 0, belem_begin, belem_end, finish_turn );

        status |= NS(Particles_copy_to_generic_addr_data)(
            in_particles, pidx, &particles, 0u );
    }

    return status;
}

SIXTRL_INLINE NS(track_status_t) NS(Track_particles_line_debug_opt_kernel_impl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const particle_set_index,
    NS(particle_num_elements_t) pidx, NS(particle_num_elements_t) const stride,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    NS(buffer_size_t) const belem_begin_id, NS(buffer_size_t) const belem_end_id,
    bool const finish_turn, NS(buffer_size_t) const slot_size,
    SIXTRL_ARGPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flags )
{
    typedef NS(particle_num_elements_t) nelements_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesGenericAddr)* ptr_particles_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* pset_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const* be_iter_t;
    typedef NS(particle_real_t) real_t;
    typedef NS(particle_index_t) index_t;

    NS(track_status_t)  status = SIXTRL_TRACK_STATUS_GENERAL_FAILURE;
    NS(arch_debugging_t) flags = SIXTRL_ARCH_DEBUGGING_MIN_FLAG;

    NS(arch_debugging_t) const PARTICLE_IDX_ILLEGAL_FLAG        = flags;
    NS(arch_debugging_t) const PBUFFER_NULL_FLAG                = flags <<  1u;
    NS(arch_debugging_t) const BELEM_BUFFER_NULL_FLAG           = flags <<  2u;
    NS(arch_debugging_t) const SLOT_SIZE_ILLEGAL_FLAG           = flags <<  3u;
    NS(arch_debugging_t) const PSET_INDEX_ILLEGAL_FLAG          = flags <<  4u;
    NS(arch_debugging_t) const PBUFFER_REQUIRES_REMAP_FLAG      = flags <<  5u;
    NS(arch_debugging_t) const BELEM_BUFFER_REQUIRES_REMAP_FLAG = flags <<  6u;
    NS(arch_debugging_t) const LINE_BOUNDARIES_ILLEGAL_FLAG     = flags <<  7u;

    flags = ( NS(arch_debugging_t) )0u;

    if( ( pbuffer != SIXTRL_NULLPTR ) && ( belem_buffer != SIXTRL_NULLPTR ) &&
        ( slot_size > ( SIXTRL_UINT64_T )0u ) &&
        ( pidx >= ( nelements_t )0u ) && ( stride > ( nelements_t )0u ) )
    {
        pset_iter_t pset_it = NS(ManagedBuffer_get_object)(
            pbuffer, particle_set_index, slot_size );

        ptr_particles_t in_particles = ( ptr_particles_t )( uintptr_t
            )NS(Object_get_begin_addr)( pset_it );

        nelements_t const num_particles = (
            ( NS(Object_get_type_id)( pset_it ) == NS(OBJECT_TYPE_PARTICLE ) ) &&
            ( in_particles != SIXTRL_NULLPTR ) )
            ? in_particles->num_particles : ( nelements_t )0u;

        be_iter_t belem_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
            belem_buffer, slot_size );

        be_iter_t belem_end = belem_begin;

        if( ( !NS(ManagedBuffer_needs_remapping)( pbuffer, slot_size ) ) &&
            ( !NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) ) &&
            ( in_particles != SIXTRL_NULLPTR ) &&
            ( belem_begin != SIXTRL_NULLPTR ) &&
            ( belem_begin_id <= belem_end_id ) &&
            ( belem_begin_id <= NS(ManagedBuffer_get_num_objects)(
                belem_buffer, slot_size ) ) )
        {
            SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) particles;

            SIXTRL_PARTICLE_DATAPTR_DEC real_t reals[] =
            {
                ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
                ( real_t )0.0, ( real_t )0.0,
                ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
                ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
                ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
                ( real_t )0.0, ( real_t )0.0, ( real_t )0.0
            };

            SIXTRL_PARTICLE_DATAPTR_DEC index_t indices[] =
            {
                ( index_t )0, ( index_t )0, ( index_t )0, ( index_t )0
            };

            NS(Particles_init_from_flat_arrays)(
                &particles, ( nelements_t )1u, &reals[ 0 ], &indices[ 0 ] );

            belem_begin = belem_begin + belem_begin_id;
            belem_end   = belem_end   + belem_end_id;

            status = SIXTRL_TRACK_SUCCESS;

            for( ; pidx < num_particles ; pidx += stride )
            {
                status |= NS(Particles_copy_from_generic_addr_data)(
                    &particles, 0u, in_particles, pidx );

                status |= NS(Track_particle_line_objs)(
                    &particles, 0, belem_begin, belem_end, finish_turn );

                status |= NS(Particles_copy_to_generic_addr_data)(
                    in_particles, pidx, &particles, 0u );
            }
        }
        else
        {
            if( NS(ManagedBuffer_needs_remapping)( pbuffer, slot_size ) )
                flags |= PBUFFER_REQUIRES_REMAP_FLAG;

            if( NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) )
                flags |= BELEM_BUFFER_REQUIRES_REMAP_FLAG;

            if( ( in_particles == SIXTRL_NULLPTR ) ||
                ( NS(ManagedBuffer_get_num_objects)( pbuffer, slot_size ) <=
                  particle_set_index ) )
                flags |= PSET_INDEX_ILLEGAL_FLAG;

            if( ( belem_begin == SIXTRL_NULLPTR ) ||
                ( belem_begin_id > belem_end_id ) ||
                ( belem_end_id > NS(ManagedBuffer_get_num_objects)(
                    belem_buffer, slot_size ) ) )
                flags |= LINE_BOUNDARIES_ILLEGAL_FLAG;
        }
    }
    else
    {
        if( pbuffer == SIXTRL_NULLPTR )      flags |= PBUFFER_NULL_FLAG;
        if( belem_buffer == SIXTRL_NULLPTR ) flags |= BELEM_BUFFER_NULL_FLAG;
        if( slot_size == 0u )                flags |= SLOT_SIZE_ILLEGAL_FLAG;
        if( pidx < ( nelements_t )0u )       flags |= PARTICLE_IDX_ILLEGAL_FLAG;
        if( stride <= ( nelements_t )0u )    flags |= PARTICLE_IDX_ILLEGAL_FLAG;
    }

    if( status != SIXTRL_TRACK_SUCCESS )
    {
        flags = NS(DebugReg_store_arch_status)( flags, status );
    }

    if(  ptr_status_flags != SIXTRL_NULLPTR )
    {
        *ptr_status_flags = flags;
    }

    return status;
}

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_TRACK_TRACK_KERNEL_OPTIMIZED_IMPL_C99_H__ */
/* end: sixtracklib/common/track/track_opt_kernel_impl.h */
