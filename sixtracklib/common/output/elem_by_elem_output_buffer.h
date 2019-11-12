#ifndef SIXTRACKLIB_COMMON_OUTPUT_ELEM_BY_ELEM_OUTPUT_BUFFER_H__
#define SIXTRACKLIB_COMMON_OUTPUT_ELEM_BY_ELEM_OUTPUT_BUFFER_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(ElemByElemConfig_calculate_output_buffer_params)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
    NS(buffer_size_t) const output_buffer_slot_size );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(ElemByElemConfig_calculate_output_buffer_params_for_particle_sets)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT pb,
    NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
    NS(buffer_size_t) const output_buffer_slot_size );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(ElemByElemConfig_calculate_output_buffer_params_detailed)(
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn_id,
    NS(particle_index_t) const max_turn_id,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
    NS(buffer_size_t) const output_buffer_slot_size );

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(ElemByElemConfig_prepare_output_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t) const num_elem_by_elem_turns,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(ElemByElemConfig_prepare_output_buffer_for_particle_sets)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT pb,
    NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(ElemByElemConfig_prepare_output_buffer_detailed)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*SIXTRL_RESTRICT output_buffer,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn_id,
    NS(particle_index_t) const max_elem_by_elem_turn_id,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset );

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(ElemByElemConfig_prepare_output_buffer_from_conf)(
    SIXTRL_BE_ARGPTR_DEC struct NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*SIXTRL_RESTRICT output_buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(ElemByElemConfig_assign_output_buffer)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        struct NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const out_buffer_index_offset );

/* ------------------------------------------------------------------------- */


SIXTRL_FN SIXTRL_STATIC int
NS(ElemByElemConfig_get_min_max_element_id_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_elem,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_elem,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_e_by_e_objs,
    NS(particle_index_t) const start_elem_id );

SIXTRL_FN SIXTRL_STATIC int
NS(ElemByElemConfig_find_min_max_element_id_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_elem,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_elem,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_e_by_e_objs,
    NS(particle_index_t) const start_elem_id );

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* *****          Implementation of C inline functions                   *** */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/output/output_buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#if !defined( _GPUCODE )

SIXTRL_INLINE int NS(ElemByElemConfig_get_min_max_element_id_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_elem,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_elem,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_e_by_e_objects,
    NS(particle_index_t) const start_elem_id )
{
    NS(Particles_init_min_max_attributes_for_find)( SIXTRL_NULLPTR,
        SIXTRL_NULLPTR, ptr_min_elem, ptr_max_elem, SIXTRL_NULLPTR,
            SIXTRL_NULLPTR );

    return NS(ElemByElemConfig_find_min_max_element_id_from_buffer)( belements,
        ptr_min_elem, ptr_max_elem, ptr_num_e_by_e_objects, start_elem_id );
}

SIXTRL_INLINE int NS(ElemByElemConfig_find_min_max_element_id_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_element_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_element_id,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT ptr_num_e_by_e_objects,
    NS(particle_index_t) const start_elem_id )
{
    int success = -1;

    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(particle_index_t) index_t;
    typedef NS(object_type_id_t) type_id_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* obj_iter_t;

    buf_size_t const num_objects = NS(Buffer_get_num_of_objects)( belements );

    SIXTRL_ASSERT( belements != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( belements ) );

    if( ( num_objects > ( buf_size_t )0u ) &&
        ( start_elem_id >= ( index_t )0u ) )
    {
        index_t min_element_id = start_elem_id;
        index_t max_element_id = start_elem_id;
        buf_size_t num_elem_by_elem_objects = ( buf_size_t )0u;

        obj_iter_t obj_it  = NS(Buffer_get_const_objects_begin)( belements );
        obj_iter_t obj_end = NS(Buffer_get_const_objects_end)( belements );

        for( ; obj_it != obj_end ; ++obj_it )
        {
            type_id_t const type_id = NS(Object_get_type_id)( obj_it );

            if( ( type_id != NS(OBJECT_TYPE_NONE) ) &&
                ( type_id != NS(OBJECT_TYPE_PARTICLE) ) &&
                ( type_id != NS(OBJECT_TYPE_INVALID) ) &&
                ( type_id != NS(OBJECT_TYPE_LINE) ) &&
                ( type_id != NS(OBJECT_TYPE_ELEM_BY_ELEM_CONF) ) )
            {
                ++num_elem_by_elem_objects;
                ++obj_it;
                break;
            }

            ++min_element_id;
        }

        max_element_id = min_element_id;

        for( ; obj_it != obj_end ; ++obj_it )
        {
            type_id_t const type_id = NS(Object_get_type_id)( obj_it );

            if( ( type_id != NS(OBJECT_TYPE_NONE) ) &&
                ( type_id != NS(OBJECT_TYPE_PARTICLE) ) &&
                ( type_id != NS(OBJECT_TYPE_INVALID) ) &&
                ( type_id != NS(OBJECT_TYPE_LINE) ) &&
                ( type_id != NS(OBJECT_TYPE_ELEM_BY_ELEM_CONF) ) )
            {
                ++num_elem_by_elem_objects;
                ++max_element_id;
            }
        }

        SIXTRL_ASSERT( num_elem_by_elem_objects <= num_objects );

        if( num_elem_by_elem_objects == ( buf_size_t )0u )
        {
            min_element_id = max_element_id = start_elem_id;
        }

        SIXTRL_ASSERT( min_element_id <= max_element_id );

        if( ( min_element_id >= start_elem_id ) &&
            ( max_element_id >= min_element_id ) &&
            ( max_element_id <  ( index_t )(
                min_element_id + num_elem_by_elem_objects ) ) )
        {
            if( (  ptr_min_element_id != SIXTRL_NULLPTR ) &&
                ( *ptr_min_element_id > min_element_id ) )
            {
                *ptr_min_element_id = min_element_id;
            }

            if( (  ptr_max_element_id != SIXTRL_NULLPTR ) &&
                ( *ptr_max_element_id <  max_element_id ) )
            {
                *ptr_max_element_id = max_element_id;
            }

            if(  ptr_num_e_by_e_objects != SIXTRL_NULLPTR )
            {
                *ptr_num_e_by_e_objects = num_elem_by_elem_objects;
            }

            success = 0;
        }
    }

    return success;
}

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_OUTPUT_ELEM_BY_ELEM_OUTPUT_BUFFER_H__ */

/* end: sixtracklib/common/output/elem_by_elem_output_buffer.h" */
