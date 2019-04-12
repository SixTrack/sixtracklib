#ifndef SIXTRACKL_COMMON_OUTPUT_OUTPUT_BUFFER_C99_H__
#define SIXTRACKL_COMMON_OUTPUT_OUTPUT_BUFFER_C99_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_monitor/be_monitor.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/internal/elem_by_elem_config_defines.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ------------------------------------------------------------------------- */
/* OutputBuffer initialization: */

typedef SIXTRL_UINT32_T NS(output_buffer_flag_t);

#if !defined( SIXTRL_OUTPUT_BUFFER_NONE )
    #define   SIXTRL_OUTPUT_BUFFER_NONE  0x00
#endif /* SIXTRL_OUTPUT_BUFFER_NONE ) */

#if !defined( SIXTRL_OUTPUT_BUFFER_ELEM_BY_ELEM )
    #define   SIXTRL_OUTPUT_BUFFER_ELEM_BY_ELEM  0x01
#endif /* SIXTRL_OUTPUT_BUFFER_ELEM_BY_ELEM */

#if !defined( SIXTRL_OUTPUT_BUFFER_BEAM_MONITORS )
    #define   SIXTRL_OUTPUT_BUFFER_BEAM_MONITORS 0x02
#endif /* SIXTRL_OUTPUT_BUFFER_BEAM_MONITORS */

SIXTRL_STATIC SIXTRL_FN bool NS(OutputBuffer_requires_output_buffer)(
    NS(output_buffer_flag_t) const flags );

SIXTRL_STATIC SIXTRL_FN bool NS(OutputBuffer_requires_elem_by_elem_output)(
    NS(output_buffer_flag_t) const flags );

SIXTRL_STATIC SIXTRL_FN bool NS(OutputBuffer_requires_beam_monitor_output)(
    NS(output_buffer_flag_t) const flags );

#if !defined( _GPUCODE )

SIXTRL_STATIC_VAR NS(output_buffer_flag_t) const NS(OUTPUT_BUFFER_NONE) =
        ( NS(output_buffer_flag_t) )SIXTRL_OUTPUT_BUFFER_NONE;

SIXTRL_STATIC_VAR NS(output_buffer_flag_t) const
    NS(OUTPUT_BUFFER_ELEM_BY_ELEM) =
        ( NS(output_buffer_flag_t) )SIXTRL_OUTPUT_BUFFER_ELEM_BY_ELEM;

SIXTRL_STATIC_VAR NS(output_buffer_flag_t) const
    NS(OUTPUT_BUFFER_BEAM_MONITORS) =
        ( NS(output_buffer_flag_t) )SIXTRL_OUTPUT_BUFFER_BEAM_MONITORS;

#endif /* !defined( _GPUCODE ) */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(output_buffer_flag_t)
NS(OutputBuffer_required_for_tracking)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer)
        *const SIXTRL_RESTRICT belems_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(output_buffer_flag_t)
NS(OutputBuffer_required_for_tracking_of_particle_sets)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT pb,
    NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT indices_begin,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT belem_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(output_buffer_flag_t)
NS(OutputBuffer_required_for_tracking_detailed)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT belem_buffer,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn_id,
    NS(particle_index_t) const max_turn_id,
    NS(buffer_size_t) const until_turn_elem_by_elem );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(OutputBuffer_prepare)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_elem_by_elem_out_index_offset,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_beam_monitor_out_index_offset,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id);

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(OutputBuffer_prepare_for_particle_sets)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT pb,
    NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_elem_by_elem_out_index_offset,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_beam_monitor_out_index_offset,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(OutputBuffer_prepare_detailed)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn_id,
    NS(particle_index_t) const max_turn_id,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_elem_by_elem_out_index_offset,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_beam_monitor_out_index_offset,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT
        ptr_max_elem_by_elem_turn_id );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN  SIXTRL_HOST_FN int
NS(OutputBuffer_calculate_output_buffer_params)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
    NS(buffer_size_t) const output_buffer_slot_size );

SIXTRL_EXTERN  SIXTRL_HOST_FN int
NS(OutputBuffer_calculate_output_buffer_params_for_particles_sets)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT pb,
    NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT indices_begin,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
    NS(buffer_size_t) const output_buffer_slot_size );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(OutputBuffer_calculate_output_buffer_params_detailed)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn_id,
    NS(particle_index_t) const max_turn_id,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
    NS(buffer_size_t) const output_buffer_slot_size );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(OutputBuffer_find_min_max_attributes)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const beam_elements_buffer,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_elem_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_elem_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_turn_id,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_e_by_e_objs,
    NS(particle_index_t) const start_elem_id );

SIXTRL_FN SIXTRL_STATIC int NS(OutputBuffer_get_min_max_attributes)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const beam_elements_buffer,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_elem_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_elem_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_turn_id,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_e_by_e_objs,
    NS(particle_index_t) const start_elem_id );

SIXTRL_FN SIXTRL_STATIC int
NS(OutputBuffer_find_min_max_attributes_on_particle_sets)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT indices_begin,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const beam_elements_buffer,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_elem_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_elem_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_turn_id,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_e_by_e_objs,
    NS(particle_index_t) const start_elem_id );

SIXTRL_FN SIXTRL_STATIC int
NS(OutputBuffer_get_min_max_attributes_on_particle_sets)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT indices_begin,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const beam_elements_buffer,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_elem_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_elem_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_turn_id,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_e_by_e_objs,
    NS(particle_index_t) const start_elem_id );

/* ------------------------------------------------------------------------- */

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* *****          Implementation of C inline functions                   *** */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/output/elem_by_elem_output_buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE bool NS(OutputBuffer_requires_output_buffer)(
    NS(output_buffer_flag_t) const flags )
{
    return ( flags != SIXTRL_OUTPUT_BUFFER_NONE );
}

SIXTRL_INLINE bool NS(OutputBuffer_requires_elem_by_elem_output)(
    NS(output_buffer_flag_t) const flags )
{
    return ( ( flags & SIXTRL_OUTPUT_BUFFER_ELEM_BY_ELEM ) ==
        SIXTRL_OUTPUT_BUFFER_ELEM_BY_ELEM );
}

SIXTRL_INLINE bool NS(OutputBuffer_requires_beam_monitor_output)(
    NS(output_buffer_flag_t) const flags )
{
    return ( ( flags & SIXTRL_OUTPUT_BUFFER_BEAM_MONITORS ) ==
        SIXTRL_OUTPUT_BUFFER_BEAM_MONITORS );
}

#if !defined( _GPUCODE )

SIXTRL_INLINE int NS(OutputBuffer_find_min_max_attributes)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const belements_buffer,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_elem_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_elem_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_turn_id,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_e_by_e_objs,
    NS(particle_index_t) const start_elem_id )
{
    int success = NS(Particles_find_min_max_attributes)( particles,
        ptr_min_part_id, ptr_max_part_id, ptr_min_elem_id, ptr_max_elem_id,
            ptr_min_turn_id, ptr_max_turn_id );

    if( success == 0 )
    {
        success = NS(ElemByElemConfig_find_min_max_element_id_from_buffer)(
            belements_buffer, ptr_min_elem_id, ptr_max_elem_id,
                ptr_num_e_by_e_objs, start_elem_id );

        if( (  success == 0 ) && (  ptr_num_e_by_e_objs != SIXTRL_NULLPTR ) &&
            ( *ptr_num_e_by_e_objs == ( NS(buffer_size_t) )0u ) )
        {
            success = -1;
        }
    }

    return success;
}

SIXTRL_INLINE int NS(OutputBuffer_get_min_max_attributes)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const belem_buffer,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_elem_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_elem_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_turn_id,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_e_by_e_objs,
    NS(particle_index_t) const start_elem_id )
{
    NS(Particles_init_min_max_attributes_for_find)( ptr_min_part_id,
        ptr_max_part_id, ptr_min_elem_id, ptr_max_elem_id,
            ptr_min_turn_id, ptr_max_turn_id );

    return NS(OutputBuffer_find_min_max_attributes)( particles, belem_buffer,
        ptr_min_part_id, ptr_max_part_id, ptr_min_elem_id, ptr_max_elem_id,
            ptr_min_turn_id, ptr_max_turn_id, ptr_num_e_by_e_objs,
                start_elem_id );
}

SIXTRL_INLINE int NS(OutputBuffer_find_min_max_attributes_on_particle_sets)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT indices_begin,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const belements_buffer,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_elem_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_elem_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_turn_id,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_e_by_e_objs,
    NS(particle_index_t) const start_elem_id )
{
    int ret = NS(Particles_buffer_find_min_max_attributes_of_particles_set)(
        particles_buffer, num_particle_sets, indices_begin, ptr_min_part_id,
            ptr_max_part_id, ptr_min_elem_id, ptr_max_elem_id,
                ptr_min_turn_id, ptr_max_turn_id );

    if( ret == 0 )
    {
        ret = NS(ElemByElemConfig_find_min_max_element_id_from_buffer)(
            belements_buffer, ptr_min_elem_id, ptr_max_elem_id,
                ptr_num_e_by_e_objs, start_elem_id );

        if( ( ret == 0 ) && (  ptr_num_e_by_e_objs != SIXTRL_NULLPTR ) &&
            ( *ptr_num_e_by_e_objs == ( NS(buffer_size_t) )0u ) )
        {
            ret = -1;
        }
    }

    return ret;
}

SIXTRL_INLINE int NS(OutputBuffer_get_min_max_attributes_on_particle_sets)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT indices_begin,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const belements_buffer,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_elem_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_elem_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_turn_id,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_e_by_e_objs,
    NS(particle_index_t) const start_elem_id )
{
    NS(Particles_init_min_max_attributes_for_find)( ptr_min_part_id,
        ptr_max_part_id, ptr_min_elem_id, ptr_max_elem_id,
            ptr_min_turn_id, ptr_max_turn_id );

    return NS(OutputBuffer_find_min_max_attributes_on_particle_sets)(
        particles_buffer, num_particle_sets, indices_begin, belements_buffer,
            ptr_min_part_id, ptr_max_part_id, ptr_min_elem_id, ptr_max_elem_id,
                ptr_min_turn_id, ptr_max_turn_id, ptr_num_e_by_e_objs,
                    start_elem_id );
}

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKL_COMMON_OUTPUT_OUTPUT_BUFFER_C99_H__ */

/* end: sixtracklib/common/output/output_buffer.h */
