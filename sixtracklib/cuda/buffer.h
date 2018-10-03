#ifndef SIXTRACKLIB_CUDA_BUFFER_H__
#define SIXTRACKLIB_CUDA_BUFFER_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <limits.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/generated/modules.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC bool NS(Buffer_uses_global_cuda_datastore)(
    const NS(Buffer) *const buffer );

SIXTRL_FN SIXTRL_STATIC bool NS(Buffer_uses_shared_cuda_datastore)(
    const NS(Buffer) *const buffer );

SIXTRL_FN SIXTRL_STATIC bool NS(Buffer_uses_local_cuda_datastore)(
    const NS(Buffer) *const buffer );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(Buffer_get_section_extent_cuda)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_slot_id );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(Buffer_get_section_data_extent_cuda)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_slot_id );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(Buffer_get_section_size_cuda)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_slot_id );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(Buffer_get_section_num_elements_cuda)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_slot_id );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(Buffer_get_section_element_size_cuda)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_slot_id );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(Buffer_get_section_max_num_elements_cuda)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_slot_id );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(Buffer_get_capacity_from_header_cuda)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC void NS(Buffer_set_section_extent_cuda)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_slot_id,
    NS(buffer_size_t) const section_num_elements );

SIXTRL_FN SIXTRL_STATIC void NS(Buffer_set_section_num_elements_cuda)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_slot_id,
    NS(buffer_size_t) const section_num_elements );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_reserve_cuda)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_num_objects,
    NS(buffer_size_t) const max_num_slots,
    NS(buffer_size_t) const max_num_dataptrs,
    NS(buffer_size_t) const max_num_garbage_elems );

SIXTRL_FN SIXTRL_STATIC void NS(Buffer_free_cuda)(
    NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC NS(Object)* NS(Buffer_add_object_cuda)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    const void *const SIXTRL_RESTRICT object,
    NS(buffer_size_t)        const object_size,
    NS(object_type_id_t)     const type_id,
    NS(buffer_size_t)        const num_obj_dataptr,
    const NS(buffer_size_t) *const SIXTRL_RESTRICT obj_dataptr_offsets,
    const NS(buffer_size_t) *const SIXTRL_RESTRICT obj_dataptr_sizes,
    const NS(buffer_size_t) *const SIXTRL_RESTRICT obj_dataptr_counts );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_clear_cuda)(
    struct NS(Buffer)* SIXTRL_RESTRICT buffer );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Buffer_remap_cuda)(
    NS(Buffer)* SIXTRL_RESTRICT buffer );

#if defined( _GPUCODE )
    SIXTRL_FN SIXTRL_STATIC int NS(Buffer_remap_flat_buffer_cuda_global)(
        __global unsigned char* SIXTRL_RESTRICT data_buffer_begin,
        NS(buffer_size_t) const slot_size );

    SIXTRL_FN SIXTRL_STATIC int NS(Buffer_remap_flat_buffer_cuda_shared)(
        __local unsigned char* SIXTRL_RESTRICT data_buffer_begin,
        NS(buffer_size_t) const slot_size );

    SIXTRL_FN SIXTRL_STATIC int NS(Buffer_remap_flat_buffer_cuda_local)(
        __private unsigned char* SIXTRL_RESTRICT data_buffer_begin,
        NS(buffer_size_t) const slot_size );

#endif /* defined( _GPUCODE ) */


#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* *******             Implementation of inline functions            ******* */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/generated/modules.h"
//     #include "sixtracklib/cuda/impl/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <CL/cl.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE  bool NS(Buffer_uses_global_cuda_datastore)(
    const NS(Buffer) *const buffer )
{
    return false;
}

SIXTRL_INLINE  bool NS(Buffer_uses_shared_cuda_datastore)(
    const NS(Buffer) *const buffer )
{
    return false;
}

SIXTRL_INLINE  bool NS(Buffer_uses_local_cuda_datastore)(
    const NS(Buffer) *const buffer )
{
    return false;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE  NS(buffer_size_t) NS(Buffer_get_section_extent_cuda)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_slot_id )
{
    ( void )buffer;
    ( void )header_slot_id;

    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE  NS(buffer_size_t) NS(Buffer_get_section_data_extent_cuda)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_slot_id )
{
    ( void )buffer;
    ( void )header_slot_id;

    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE  NS(buffer_size_t) NS(Buffer_get_section_size_cuda)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_slot_id )
{
    ( void )buffer;
    ( void )header_slot_id;

    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE  NS(buffer_size_t) NS(Buffer_get_section_num_elements_cuda)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_slot_id )
{
    ( void )buffer;
    ( void )header_slot_id;

    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE  NS(buffer_size_t) NS(Buffer_get_section_element_size_cuda)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_slot_id )
{
    ( void )buffer;
    ( void )header_slot_id;

    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE  NS(buffer_size_t) NS(Buffer_get_section_max_num_elements_cuda)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_slot_id )
{
    ( void )buffer;
    ( void )header_slot_id;

    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE  NS(buffer_size_t) NS(Buffer_get_capacity_from_header_cuda)(
    const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    ( void )buffer;
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE  void NS(Buffer_set_section_extent_cuda)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_slot_id,
    NS(buffer_size_t) const section_extent )
{
    ( void )buffer;
    ( void )header_slot_id;
    ( void )section_extent;
    return;
}

SIXTRL_INLINE void NS(Buffer_set_section_num_elements_cuda)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const header_slot_id,
    NS(buffer_size_t) const section_num_elements )
{
    ( void )buffer;
    ( void )header_slot_id;
    ( void )section_num_elements;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE  int NS(Buffer_reserve_cuda)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_num_objects,
    NS(buffer_size_t) const max_num_slots,
    NS(buffer_size_t) const max_num_dataptrs,
    NS(buffer_size_t) const max_num_garbage_elems )
{
    ( void )buffer;
    ( void )max_num_objects;
    ( void )max_num_slots;
    ( void )max_num_dataptrs;
    ( void )max_num_garbage_elems;

    return -1;
}

SIXTRL_INLINE  void NS(Buffer_free_cuda)(
    NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    ( void )buffer;
    return;
}

SIXTRL_INLINE  NS(Object)* NS(Buffer_add_object_cuda)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    const void *const SIXTRL_RESTRICT object,
    NS(buffer_size_t)        const object_size,
    NS(object_type_id_t)     const type_id,
    NS(buffer_size_t)        const num_obj_dataptr,
    const NS(buffer_size_t) *const SIXTRL_RESTRICT obj_dataptr_offsets,
    const NS(buffer_size_t) *const SIXTRL_RESTRICT obj_dataptr_sizes,
    const NS(buffer_size_t) *const SIXTRL_RESTRICT obj_dataptr_counts )
{
    ( void )buffer;
    ( void )object;
    ( void )object_size;
    ( void )type_id;
    ( void )num_obj_dataptr;
    ( void )obj_dataptr_offsets;
    ( void )obj_dataptr_sizes;
    ( void )obj_dataptr_counts;

    return SIXTRL_NULLPTR;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE  int NS(Buffer_clear_cuda)(
    struct NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    ( void )buffer;
    return -1;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Buffer_remap_cuda)( NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    ( void )buffer;
    return -1;
}

#if defined( _GPUCODE )
    SIXTRL_INLINE  int NS(Buffer_remap_flat_buffer_cuda_global)(
        __global unsigned char* SIXTRL_RESTRICT data_buffer_begin,
        NS(buffer_size_t) const slot_size )
    {
        ( void )data_buffer_begin;
        ( void )slot_size;

        return -1;
    }

    SIXTRL_INLINE  int NS(Buffer_remap_flat_buffer_cuda_shared)(
        __local unsigned char* SIXTRL_RESTRICT data_buffer_begin,
        NS(buffer_size_t) const slot_size )
    {
        ( void )data_buffer_begin;
        ( void )slot_size;

        return -1;
    }

    SIXTRL_INLINE  int NS(Buffer_remap_flat_buffer_cuda_local)(
        __private unsigned char* SIXTRL_RESTRICT data_buffer_begin,
        NS(buffer_size_t) const slot_size )
    {
        ( void )data_buffer_begin;
        ( void )slot_size;

        return -1;
    }

#endif /* defined( _GPUCODE ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_CUDA_BUFFER_H__ */

/* end: sixtracklib/cuda/buffer.h */
