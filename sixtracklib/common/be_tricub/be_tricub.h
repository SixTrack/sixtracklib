#ifndef SIXTRACKLIB_COMMON_BE_TRICUB_C99_H__
#define SIXTRACKLIB_COMMON_BE_TRICUB_C99_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/assign_address_item.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* !defined( __cplusplus ) && !defined( _GPUCODE ) */

#if !defined( SIXTRL_BE_TRICUBMAP_MATRIX_DIM )
    #define   SIXTRL_BE_TRICUBMAP_MATRIX_DIM 8
#endif /* !defined( SIXTRL_BE_TRICUBMAP_MATRIX_DIM ) */

typedef SIXTRL_INT64_T NS(be_tricub_int_t);
typedef SIXTRL_REAL_T  NS(be_tricub_real_t);

typedef struct NS(TriCubData)
{
    NS(be_tricub_real_t)    x0              SIXTRL_ALIGN( 8 );
    NS(be_tricub_real_t)    dx              SIXTRL_ALIGN( 8 );
    NS(be_tricub_int_t)     nx              SIXTRL_ALIGN( 8 );

    NS(be_tricub_real_t)    y0              SIXTRL_ALIGN( 8 );
    NS(be_tricub_real_t)    dy              SIXTRL_ALIGN( 8 );
    NS(be_tricub_int_t)     ny              SIXTRL_ALIGN( 8 );

    NS(be_tricub_real_t)    z0              SIXTRL_ALIGN( 8 );
    NS(be_tricub_real_t)    dz              SIXTRL_ALIGN( 8 );
    NS(be_tricub_int_t)     nz              SIXTRL_ALIGN( 8 );

    NS(be_tricub_int_t)     method          SIXTRL_ALIGN( 8 );

    NS(buffer_addr_t)       table_addr      SIXTRL_ALIGN( 8 );
}
NS(TriCubData);

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCubData_preset)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data );

SIXTRL_STATIC SIXTRL_FN void NS(TriCubData_clear)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCubData_x0)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCubData_dx)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_int_t) NS(TriCubData_nx)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCubData_y0)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCubData_dy)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_int_t) NS(TriCubData_ny)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCubData_z0)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCubData_dz)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_int_t) NS(TriCubData_nz)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_int_t) NS(TriCubData_method)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_int_t) NS(TriCubData_table_size)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data );

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t) NS(TriCubData_table_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(be_tricub_real_t) const*
NS(TriCubData_const_table_begin)( SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData)
    *const SIXTRL_RESTRICT data );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(be_tricub_real_t)*
NS(TriCubData_table_begin)( SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
    SIXTRL_RESTRICT data );

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_STATIC SIXTRL_FN void NS(TriCubData_set_x0)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_real_t) const x0 );

SIXTRL_STATIC SIXTRL_FN void NS(TriCubData_set_dx)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_real_t) const dx );

SIXTRL_STATIC SIXTRL_FN void NS(TriCubData_set_nx)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_int_t)  const nx );

SIXTRL_STATIC SIXTRL_FN void NS(TriCubData_set_y0)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_real_t) const y0 );

SIXTRL_STATIC SIXTRL_FN void NS(TriCubData_set_dy)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_real_t) const dy );

SIXTRL_STATIC SIXTRL_FN void NS(TriCubData_set_ny)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_int_t)  const ny );

SIXTRL_STATIC SIXTRL_FN void NS(TriCubData_set_z0)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_real_t) const z0 );

SIXTRL_STATIC SIXTRL_FN void NS(TriCubData_set_dz)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_real_t) const dz );

SIXTRL_STATIC SIXTRL_FN void NS(TriCubData_set_nz)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_int_t)  const nz );

SIXTRL_STATIC SIXTRL_FN void NS(TriCubData_set_method)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_int_t) const method );

SIXTRL_STATIC SIXTRL_FN void NS(TriCubData_set_table_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(buffer_addr_t) const table_addr );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t) NS(TriCubData_type_id)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(TriCubData_num_dataptrs)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(TriCubData_num_slots)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const*
NS(TriCubData_const_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const obj );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCubData_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* obj );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const*
NS(TriCubData_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCubData_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const*
NS(TriCubData_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCubData_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(TriCubData_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(TriCubData_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(TriCubData_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TriCubData_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(be_tricub_int_t) const nx, NS(be_tricub_int_t) const ny,
    NS(be_tricub_int_t) const nz,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCubData_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(be_tricub_int_t) const nx, NS(be_tricub_int_t) const ny,
    NS(be_tricub_int_t) const nz );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCubData_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(be_tricub_real_t) const x0, NS(be_tricub_real_t) const dx,
    NS(be_tricub_int_t)  const nx,
    NS(be_tricub_real_t) const y0, NS(be_tricub_real_t) const dy,
    NS(be_tricub_int_t)  const ny,
    NS(be_tricub_real_t) const z0, NS(be_tricub_real_t) const dz,
    NS(be_tricub_int_t)  const nz,
    NS(buffer_addr_t) const table_addr );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCubData_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data );

#endif /* !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCubData_copy)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT dest,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT src );

/* ========================================================================= */

typedef struct NS(TriCub)
{
    NS(be_tricub_real_t)    x               SIXTRL_ALIGN( 8 );
    NS(be_tricub_real_t)    y               SIXTRL_ALIGN( 8 );
    NS(be_tricub_real_t)    z               SIXTRL_ALIGN( 8 );
    NS(be_tricub_real_t)    length          SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t)       data_addr       SIXTRL_ALIGN( 8 );
}
NS(TriCub);

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(TriCub)* NS(TriCub_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN void NS(TriCub_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCub_x)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCub_y)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCub_z)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCub_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t) NS(TriCub_data_addr)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub );


SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const*
NS(TriCub_const_data)( SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const
    SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCub_data)( SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN void NS(TriCub_set_x)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const x );

SIXTRL_STATIC SIXTRL_FN void NS(TriCub_set_y)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const y );

SIXTRL_STATIC SIXTRL_FN void NS(TriCub_set_z)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const z );

SIXTRL_STATIC SIXTRL_FN void NS(TriCub_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const length );

SIXTRL_STATIC SIXTRL_FN void NS(TriCub_set_data_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(buffer_addr_t) const data_addr );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t) NS(TriCub_type_id)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(TriCub_num_dataptrs)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(TriCub_num_slots)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCub_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCub_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCub_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(TriCub) const*
NS(TriCub_const_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const obj );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(TriCub)*
NS(TriCub_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* obj );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(TriCub) const*
NS(TriCub_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(TriCub)*
NS(TriCub_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(TriCub) const*
NS(TriCub_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(TriCub)*
NS(TriCub_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TriCub_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)*
NS(TriCub_new)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)*
NS(TriCub_add)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(be_tricub_real_t) const x, NS(be_tricub_real_t) const y,
    NS(be_tricub_real_t) const z, NS(be_tricub_real_t) const length,
    NS(buffer_addr_t) const data_addr );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)*
NS(TriCub_add_copy)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub );

/* ------------------------------------------------------------------------ */

SIXTRL_EXTERN SIXTRL_HOST_FN
SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem) const*
NS(TriCub_buffer_create_assign_address_item)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT assign_items_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const belements_index,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT
        tricub_data_buffer,
    NS(buffer_size_t) const tricub_data_index );

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCub_copy)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)* SIXTRL_RESTRICT dest,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT src );

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* !defined( __cplusplus ) && !defined( _GPUCODE ) */

/* ************************************************************************* */
/* ***          Implementation of inline functions and methods          **** */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer/managed_buffer.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #if !defined( _GPUCODE )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* !defined( __cplusplus ) && !defined( _GPUCODE ) */


SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* NS(TriCubData_preset)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data )
{
    if( data != SIXTRL_NULLPTR )
    {
        NS(TriCubData_set_nx)( data, ( NS(be_tricub_int_t) )0.0 );
        NS(TriCubData_set_ny)( data, ( NS(be_tricub_int_t) )0.0 );
        NS(TriCubData_set_nz)( data, ( NS(be_tricub_int_t) )0.0 );
        NS(TriCubData_set_table_addr)( data, ( NS(buffer_addr_t) )0.0 );

        NS(TriCubData_clear)( data );
    }

    return data;
}

SIXTRL_INLINE void NS(TriCubData_clear)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data )
{
    if( data != SIXTRL_NULLPTR )
    {
        NS(TriCubData_set_x0)( data, ( NS(be_tricub_real_t) )0.0 );
        NS(TriCubData_set_dx)( data, ( NS(be_tricub_real_t) )0.0 );

        NS(TriCubData_set_y0)( data, ( NS(be_tricub_real_t) )0.0 );
        NS(TriCubData_set_dy)( data, ( NS(be_tricub_real_t) )0.0 );

        NS(TriCubData_set_z0)( data, ( NS(be_tricub_real_t) )0.0 );
        NS(TriCubData_set_dz)( data, ( NS(be_tricub_real_t) )0.0 );

        NS(TriCubData_set_method)( data, ( NS(be_tricub_int_t) )0 );

        if( ( NS(TriCubData_table_size)( data ) > ( NS(be_tricub_int_t) )0u ) &&
            ( NS(TriCubData_table_addr)( data ) != ( NS(buffer_addr_t) )0u ) )
        {
            typedef NS(be_tricub_real_t) real_t;

            SIXTRL_STATIC_VAR real_t const ZERO = ( real_t )0.0;

            NS(buffer_size_t) const num_table_entries =
                NS(TriCubData_table_size)( data );

            SIXTRL_BUFFER_DATAPTR_DEC real_t* ptr_table_data =
                NS(TriCubData_table_begin)( data );

            SIXTRL_ASSERT( ptr_table_data != SIXTRL_NULLPTR );
            SIXTRL_ASSERT( num_table_entries > ( NS(buffer_size_t) )0u );

            SIXTRACKLIB_SET_VALUES(
                real_t, ptr_table_data, num_table_entries, ZERO );
        }
    }
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCubData_x0)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data )
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->x0;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCubData_dx)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data )
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->dx;
}

SIXTRL_INLINE NS(be_tricub_int_t) NS(TriCubData_nx)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data )
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->nx;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCubData_y0)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data )
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->y0;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCubData_dy)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data )
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->dy;
}

SIXTRL_INLINE NS(be_tricub_int_t) NS(TriCubData_ny)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data )
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->ny;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCubData_z0)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data )
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->z0;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCubData_dz)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data )
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->dz;
}

SIXTRL_INLINE NS(be_tricub_int_t) NS(TriCubData_nz)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data )
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->nz;
}

SIXTRL_INLINE NS(be_tricub_int_t) NS(TriCubData_method)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data )
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->method;
}

SIXTRL_INLINE NS(be_tricub_int_t) NS(TriCubData_table_size)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data )
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->nx * data->ny * data->nz * 8u;
}

SIXTRL_INLINE NS(buffer_addr_t) NS(TriCubData_table_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data )
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->table_addr;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(be_tricub_real_t) const*
NS(TriCubData_const_table_begin)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data )
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    SIXTRL_ASSERT(
        ( ( NS(TriCubData_table_size)( data ) == ( NS(be_tricub_int_t) )0u ) &&
          ( data->table_addr == ( NS(buffer_addr_t) )0u ) &&
          ( ( data->nx == ( NS(be_tricub_int_t) )0 ) ||
            ( data->ny == ( NS(be_tricub_int_t) )0 ) ||
            ( data->nz == ( NS(be_tricub_int_t) )0 ) ) ) ||
        ( ( NS(TriCubData_table_size)( data )  > ( NS(be_tricub_int_t) )0u ) &&
          ( data->table_addr != ( NS(buffer_addr_t) )0u ) ) );

    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(be_tricub_real_t) const* )(
        uintptr_t )data->table_addr;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(be_tricub_real_t)*
NS(TriCubData_table_begin)( SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
    SIXTRL_RESTRICT data )
{
    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(be_tricub_real_t)*
        )NS(TriCubData_const_table_begin)( data );
}

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_INLINE void NS(TriCubData_set_x0)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_real_t) const x0 )
{
    if( data != SIXTRL_NULLPTR ) data->x0 = x0;
}

SIXTRL_INLINE void NS(TriCubData_set_dx)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
     NS(be_tricub_real_t) const dx )
{
    if( data != SIXTRL_NULLPTR ) data->dx = dx;
}

SIXTRL_INLINE void NS(TriCubData_set_nx)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
     NS(be_tricub_int_t)  const nx )
{
    if( data != SIXTRL_NULLPTR ) data->nx = nx;
}

SIXTRL_INLINE void NS(TriCubData_set_y0)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
     NS(be_tricub_real_t) const y0 )
{
    if( data != SIXTRL_NULLPTR ) data->y0 = y0;
}

SIXTRL_INLINE void NS(TriCubData_set_dy)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
     NS(be_tricub_real_t) const dy )
{
    if( data != SIXTRL_NULLPTR ) data->dy = dy;
}

SIXTRL_INLINE void NS(TriCubData_set_ny)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
     NS(be_tricub_int_t)  const ny )
{
    if( data != SIXTRL_NULLPTR ) data->ny = ny;
}

SIXTRL_INLINE void NS(TriCubData_set_z0)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
     NS(be_tricub_real_t) const z0 )
{
    if( data != SIXTRL_NULLPTR ) data->z0 = z0;
}

SIXTRL_INLINE void NS(TriCubData_set_dz)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
     NS(be_tricub_real_t) const dz )
{
    if( data != SIXTRL_NULLPTR ) data->dz = dz;
}

SIXTRL_INLINE void NS(TriCubData_set_nz)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
     NS(be_tricub_int_t)  const nz )
{
    if( data != SIXTRL_NULLPTR ) data->nz = nz;
}

SIXTRL_INLINE void NS(TriCubData_set_method)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
     NS(be_tricub_int_t) const method )
{
    if( data != SIXTRL_NULLPTR ) data->method = method;
}

SIXTRL_INLINE void NS(TriCubData_set_table_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(buffer_addr_t) const table_addr )
{
    if( data != SIXTRL_NULLPTR ) data->table_addr = table_addr;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(object_type_id_t) NS(TriCubData_type_id)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data )
{
    ( void )data;
    return NS(OBJECT_TYPE_TRICUB_DATA);
}

SIXTRL_INLINE NS(buffer_size_t) NS(TriCubData_num_dataptrs)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data )
{
    ( void )data;
    return ( NS(buffer_size_t) )1u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(TriCubData_num_slots)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    buf_size_t num_slots = ( buf_size_t )0u;

    if( ( data != SIXTRL_NULLPTR ) && ( slot_size > ( buf_size_t )0u ) )
    {
        buf_size_t data_size = NS(ManagedBuffer_get_slot_based_length)(
            sizeof( NS(TriCubData) ), slot_size );

        data_size += NS(ManagedBuffer_get_slot_based_length)(
            sizeof( NS(be_tricub_real_t) ) * NS(TriCubData_table_size)( data ),
                slot_size );

        num_slots = data_size / slot_size;

        if( ( num_slots * slot_size ) < data_size ) ++num_slots;
        SIXTRL_ASSERT( ( num_slots * slot_size ) >= data_size );
    }

    return num_slots;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const*
NS(TriCubData_const_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const obj )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const* ptr_elem_t;

    ptr_elem_t ptr_elem =
        ( NS(Object_get_type_id)( obj ) == NS(OBJECT_TYPE_TRICUB_DATA) )
            ? ( ptr_elem_t )( uintptr_t )NS(Object_get_begin_addr)( obj )
            : SIXTRL_NULLPTR;

    SIXTRL_ASSERT( ( ptr_elem == SIXTRL_NULLPTR ) ||
        ( ( NS(Object_get_size)( obj ) >= sizeof( NS(TriCubData) ) ) &&
          ( NS(Object_get_begin_addr)( obj ) != ( NS(buffer_addr_t) )0u ) &&
          ( NS(Object_get_type_id)( obj ) == NS(OBJECT_TYPE_TRICUB_DATA) ) ) );

    return ptr_elem;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCubData_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* obj )
{
    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
        )NS(TriCubData_const_from_obj_index)( obj );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const*
NS(TriCubData_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size )
{
    return NS(TriCubData_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCubData_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size )
{
    return NS(TriCubData_from_obj_index)(
        NS(ManagedBuffer_get_object)( buffer_begin, index, slot_size ) );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(arch_status_t) NS(TriCubData_copy)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT dest,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT src )
{
    typedef NS(be_tricub_int_t)  int_t;
    typedef NS(be_tricub_real_t) real_t;

    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( dest != SIXTRL_NULLPTR ) && ( src != SIXTRL_NULLPTR ) &&
        ( dest != src ) )
    {
        int_t const src_table_size = NS(TriCubData_table_size)( src );
        int_t const dest_table_size = NS(TriCubData_table_size)( dest );

        SIXTRL_BUFFER_DATAPTR_DEC real_t* dest_table_begin =
            NS(TriCubData_table_begin)( dest );

        SIXTRL_BUFFER_DATAPTR_DEC real_t const* src_table_begin =
            NS(TriCubData_const_table_begin)( src );

        if( ( src_table_size > ( int_t )0 ) &&
            ( src_table_size == dest_table_size ) &&
            ( dest_table_begin != SIXTRL_NULLPTR ) &&
            ( src_table_begin  != SIXTRL_NULLPTR ) )
        {
            if( dest_table_begin != src_table_begin )
            {
                SIXTRACKLIB_COPY_VALUES( real_t, dest_table_begin,
                    src_table_begin, src_table_size );
            }

            status = SIXTRL_ARCH_STATUS_SUCCESS;
        }
        else if( ( src_table_size == ( int_t )0u ) &&
                 ( src_table_size == dest_table_size ) &&
                 ( src_table_begin == dest_table_begin ) )
        {
            status = SIXTRL_ARCH_STATUS_SUCCESS;
        }

        if( status == SIXTRL_ARCH_STATUS_SUCCESS )
        {
            NS(TriCubData_set_x0)( dest, NS(TriCubData_x0)( src ) );
            NS(TriCubData_set_dx)( dest, NS(TriCubData_dx)( src ) );
            NS(TriCubData_set_nx)( dest, NS(TriCubData_nx)( src ) );

            NS(TriCubData_set_y0)( dest, NS(TriCubData_y0)( src ) );
            NS(TriCubData_set_dy)( dest, NS(TriCubData_dy)( src ) );
            NS(TriCubData_set_ny)( dest, NS(TriCubData_ny)( src ) );

            NS(TriCubData_set_z0)( dest, NS(TriCubData_z0)( src ) );
            NS(TriCubData_set_dz)( dest, NS(TriCubData_dz)( src ) );
            NS(TriCubData_set_nz)( dest, NS(TriCubData_nz)( src ) );

            NS(TriCubData_set_method)( dest, NS(TriCubData_method)( src ) );
        }
    }
    else if( ( dest == src ) && ( dest != SIXTRL_NULLPTR ) )
    {
        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* ========================================================================= */


SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(TriCub)* NS(TriCub_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub )
{
    if( tricub != SIXTRL_NULLPTR )
    {
        NS(TriCub_set_data_addr)( tricub, ( NS(buffer_addr_t) )0u );
        NS(TriCub_clear)( tricub );
    }

    return tricub;
}

SIXTRL_INLINE void NS(TriCub_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub )
{
    if( tricub != SIXTRL_NULLPTR )
    {
        NS(TriCub_set_x)( tricub, ( NS(be_tricub_real_t) )0.0 );
        NS(TriCub_set_y)( tricub, ( NS(be_tricub_real_t) )0.0 );
        NS(TriCub_set_z)( tricub, ( NS(be_tricub_real_t) )0.0 );
        NS(TriCub_set_length)( tricub, ( NS(be_tricub_real_t) )0.0 );
    }
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCub_x)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->x;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCub_y)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->y;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCub_z)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->z;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCub_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->length;
}

SIXTRL_INLINE NS(buffer_addr_t) NS(TriCub_data_addr)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->data_addr;
}


SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const*
NS(TriCub_const_data)( SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const
    SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const* )(
        uintptr_t )tricub->data_addr;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCub_data)( SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* )( uintptr_t
        )tricub->data_addr;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS(TriCub_set_x)( SIXTRL_BE_ARGPTR_DEC NS(TriCub)*
    SIXTRL_RESTRICT tricub, NS(be_tricub_real_t) const x )
{
    if( tricub != SIXTRL_NULLPTR ) tricub->x = x;
}

SIXTRL_INLINE void NS(TriCub_set_y)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const y )
{
    if( tricub != SIXTRL_NULLPTR ) tricub->y = y;
}

SIXTRL_INLINE void NS(TriCub_set_z)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const z )
{
    if( tricub != SIXTRL_NULLPTR ) tricub->z = z;
}

SIXTRL_INLINE void NS(TriCub_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)*
    SIXTRL_RESTRICT tricub, NS(be_tricub_real_t) const length )
{
    if( tricub != SIXTRL_NULLPTR ) tricub->length = length;
}

SIXTRL_INLINE void NS(TriCub_set_data_addr)( SIXTRL_BE_ARGPTR_DEC NS(TriCub)*
    SIXTRL_RESTRICT tricub, NS(buffer_addr_t) const data_addr )
{
    if( tricub != SIXTRL_NULLPTR ) tricub->data_addr = data_addr;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(object_type_id_t) NS(TriCub_type_id)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    ( void )tricub;
    return NS(OBJECT_TYPE_TRICUB);
}

SIXTRL_INLINE NS(buffer_size_t) NS(TriCub_num_dataptrs)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    ( void )tricub;
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(TriCub_num_slots)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub,
    NS(buffer_size_t) const slot_size )
{
    SIXTRL_ASSERT( NS(TriCub_num_dataptrs)( tricub ) ==
        ( NS(buffer_size_t) )0u );

    ( void )tricub;

    return NS(Buffer_num_slots_for_trivial_object)(
        sizeof( NS(TriCub) ), slot_size );
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCub_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub,
    NS(buffer_size_t) const slot_size )
{
    SIXTRL_ASSERT( NS(TriCub_num_dataptrs)( tricub ) ==
        ( NS(buffer_size_t) )0u );

    ( void )tricub;
    return NS(Buffer_set_attr_arrays_for_trivial_object)(
        offsets_begin, max_num_offsets, slot_size );
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCub_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    SIXTRL_ASSERT( NS(TriCub_num_dataptrs)( tricub ) ==
        ( NS(buffer_size_t) )0u );

    ( void )tricub;
    return NS(Buffer_set_attr_arrays_for_trivial_object)(
        counts_begin, max_num_counts, SIXTRL_BUFFER_DEFAULT_SLOT_SIZE );
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCub_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub,
    NS(buffer_size_t) const slot_size )
{
    SIXTRL_ASSERT( NS(TriCub_num_dataptrs)( tricub ) ==
        ( NS(buffer_size_t) )0u );

    ( void )tricub;
    return NS(Buffer_set_attr_arrays_for_trivial_object)(
        sizes_begin, max_num_sizes, slot_size );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(TriCub) const*
NS(TriCub_const_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const obj )
{
    typedef SIXTRL_BE_ARGPTR_DEC NS(TriCub) const* ptr_elem_t;

    ptr_elem_t ptr_elem =
        ( NS(Object_get_type_id)( obj ) == NS(OBJECT_TYPE_TRICUB) )
            ? ( ptr_elem_t )( uintptr_t )NS(Object_get_begin_addr)( obj )
            : SIXTRL_NULLPTR;

    SIXTRL_ASSERT( ( ptr_elem == SIXTRL_NULLPTR ) ||
        ( ( NS(Object_get_size)( obj ) >= sizeof( NS(TriCub) ) ) &&
          ( NS(Object_get_begin_addr)( obj ) != ( NS(buffer_addr_t) )0u ) &&
          ( NS(Object_get_type_id)( obj ) == NS(OBJECT_TYPE_TRICUB) ) ) );

    return ptr_elem;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(TriCub)*
NS(TriCub_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* obj )
{
    return ( SIXTRL_BE_ARGPTR_DEC NS(TriCub)* )NS(TriCub_from_obj_index)( obj );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(TriCub) const*
NS(TriCub_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size )
{
    return NS(TriCub_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(TriCub)* NS(TriCub_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size )
{
    return NS(TriCub_from_obj_index)(
        NS(ManagedBuffer_get_object)( buffer_begin, index, slot_size ) );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(arch_status_t) NS(TriCub_copy)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)* SIXTRL_RESTRICT dest,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT src )
{
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( dest != SIXTRL_NULLPTR ) && ( src != SIXTRL_NULLPTR ) )
    {
        *dest = *src;
        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* !defined( __cplusplus ) && !defined( _GPUCODE ) */


#endif /* SIXTRACKLIB_COMMON_BE_TRICUB_C99_H__ */
/* end: sixtracklib/common/be_tricub/be_tricub.h */

