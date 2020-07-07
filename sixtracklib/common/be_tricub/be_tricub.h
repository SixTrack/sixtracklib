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

    NS(be_tricub_int_t)     mirror_x        SIXTRL_ALIGN( 8 );
    NS(be_tricub_int_t)     mirror_y        SIXTRL_ALIGN( 8 );
    NS(be_tricub_int_t)     mirror_z        SIXTRL_ALIGN( 8 );

    NS(buffer_addr_t)       table_addr      SIXTRL_ALIGN( 8 );
}
NS(TriCubData);

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t) NS(TriCubData_type_id)(
    void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(TriCubData_num_dataptrs)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(TriCubData_num_slots)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCubData_preset)( SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
    SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCubData_clear)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCubData_x0)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCubData_dx)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_int_t) NS(TriCubData_nx)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCubData_y0)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCubData_dy)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_int_t) NS(TriCubData_ny)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCubData_z0)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCubData_dz)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_int_t) NS(TriCubData_nz)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_int_t) NS(TriCubData_mirror_x)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_int_t) NS(TriCubData_mirror_y)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_int_t) NS(TriCubData_mirror_z)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_int_t) NS(TriCubData_table_size)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t) NS(TriCubData_table_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(be_tricub_real_t) const*
NS(TriCubData_const_table_begin)( SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData)
    *const SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(be_tricub_real_t)*
NS(TriCubData_table_begin)( SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
    SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCubData_set_x0)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_real_t) const x0 ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCubData_set_dx)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_real_t) const dx ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCubData_set_nx)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_int_t)  const nx ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCubData_set_y0)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_real_t) const y0 ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCubData_set_dy)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_real_t) const dy ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCubData_set_ny)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_int_t)  const ny ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCubData_set_z0)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_real_t) const z0 ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCubData_set_dz)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_real_t) const dz ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCubData_set_nz)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_int_t)  const nz ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCubData_set_mirror_x)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_int_t) const mirror_x ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCubData_set_mirror_y)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_int_t) const mirror_y ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCubData_set_mirror_z)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_int_t) const mirror_z ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCubData_set_table_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(buffer_addr_t) const table_addr ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const*
NS(TriCubData_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
    NS(Object) *const obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCubData_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC
    NS(Object)* obj ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const*
NS(TriCubData_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCubData_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const*
NS(TriCubData_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCubData_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_HOST_FN NS(arch_size_t) NS(TriCubData_ptr_offset)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData)
        *const SIXTRL_RESTRICT d ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(TriCubData_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(TriCubData_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(TriCubData_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(object_type_id_t)
    NS(TriCubData_type_id_ext)( void ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_size_t) NS(TriCubData_ptr_offset_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData)
        *const SIXTRL_RESTRICT d ) SIXTRL_NOEXCEPT;

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TriCubData_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(be_tricub_int_t) const nx, NS(be_tricub_int_t) const ny,
    NS(be_tricub_int_t) const nz,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_requ_dataptrs ) SIXTRL_NOEXCEPT;

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
    NS(be_tricub_int_t)  const nz, NS(be_tricub_int_t) const mirror_x,
    NS(be_tricub_int_t) const mirror_y, NS(be_tricub_int_t) const mirror_z,
    NS(buffer_addr_t) const table_addr );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCubData_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT orig );

#endif /* !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCubData_copy)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT dest,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT;

/* ========================================================================= */

typedef struct NS(TriCub)
{
    NS(be_tricub_real_t)    x_shift            SIXTRL_ALIGN( 8 );
    NS(be_tricub_real_t)    y_shift            SIXTRL_ALIGN( 8 );
    NS(be_tricub_real_t)    tau_shift          SIXTRL_ALIGN( 8 );
    NS(be_tricub_real_t)    dipolar_kick_px    SIXTRL_ALIGN( 8 );
    NS(be_tricub_real_t)    dipolar_kick_py    SIXTRL_ALIGN( 8 );
    NS(be_tricub_real_t)    dipolar_kick_ptau  SIXTRL_ALIGN( 8 );
    NS(be_tricub_real_t)    length             SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t)       data_addr          SIXTRL_ALIGN( 8 );
}
NS(TriCub);

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t) NS(TriCub_type_id)(
    void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(TriCub_num_dataptrs)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const
        SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(TriCub_num_slots)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(TriCub)* NS(TriCub_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCub_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT;

/*  -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --  */

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCub_x_shift)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const
        SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCub_y_shift)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const
        SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCub_tau_shift)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const
        SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCub_dipolar_kick_px)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const
        SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCub_dipolar_kick_py)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const
        SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCub_dipolar_kick_ptau)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const
        SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(be_tricub_real_t) NS(TriCub_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const
        SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t) NS(TriCub_data_addr)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const
        SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT;


SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const*
NS(TriCub_const_data)( SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const
    SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCub_data)( SIXTRL_BE_ARGPTR_DEC NS(TriCub)*
    SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCub_set_x_shift)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const x_shift ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCub_set_y_shift)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const y_shift ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCub_set_tau_shift)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const tau_shift ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCub_set_dipolar_kick_px)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const dipolar_kick_px ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCub_set_dipolar_kick_py)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const dipolar_kick_py ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCub_set_dipolar_kick_ptau)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const dipolar_kick_ptau ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCub_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const length ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCub_set_data_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(buffer_addr_t) const data_addr ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub) const*
NS(TriCub_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
    NS(Object) *const obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)*
NS(TriCub_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC
    NS(Object)* obj ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub) const*
NS(TriCub_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)*
NS(TriCub_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub) const*
NS(TriCub_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)*
NS(TriCub_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_HOST_FN NS(arch_size_t) NS(TriCub_data_addr_offset)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const
        SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(TriCub_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(TriCub_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(TriCub_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const
        SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(object_type_id_t)
    NS(TriCub_type_id_ext)( void ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_size_t) NS(TriCub_data_addr_offset_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const
        SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT;

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TriCub_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_requ_dataptrs ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)*
NS(TriCub_new)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)*
NS(TriCub_add)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(be_tricub_real_t) const x_shift, NS(be_tricub_real_t) const y_shift,
    NS(be_tricub_real_t) const tau_shift,
    NS(be_tricub_real_t) const dipolar_kick_px,
    NS(be_tricub_real_t) const dipolar_kick_py,
    NS(be_tricub_real_t) const dipolar_kick_ptau,
    NS(be_tricub_real_t) const length,
    NS(buffer_addr_t) const data_addr );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)*
NS(TriCub_add_copy)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub );

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_EXTERN SIXTRL_HOST_FN
SIXTRL_BUFFER_DATAPTR_DEC NS(AssignAddressItem) const*
NS(TriCub_buffer_create_assign_address_item)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT assign_items_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    NS(buffer_size_t) const belements_index,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT
        tricub_data_buffer,
    NS(buffer_size_t) const tricub_data_buffer_id,
    NS(buffer_size_t) const tricub_data_index );

#endif /* !defined( _GPUCODE ) */

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(TriCub_copy)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)* SIXTRL_RESTRICT dest,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const
        SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT;

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* !defined( __cplusplus ) && !defined( _GPUCODE ) */

/* ************************************************************************* */
/* ***          Implementation of inline functions and methods          **** */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer/managed_buffer.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/internal/compiler_attributes.h"
    #if !defined( _GPUCODE )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* !defined( __cplusplus ) && !defined( _GPUCODE ) */


SIXTRL_INLINE NS(object_type_id_t) NS(TriCubData_type_id)() SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(OBJECT_TYPE_TRICUB_DATA);
}

SIXTRL_INLINE NS(buffer_size_t) NS(TriCubData_num_dataptrs)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( data ) ) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )1u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(TriCubData_num_slots)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(buffer_size_t) num_slots = ( NS(buffer_size_t) )0u;

    if( ( data != SIXTRL_NULLPTR ) && ( slot_size > ( NS(buffer_size_t) )0u ) )
    {
        NS(buffer_size_t) num_bytes = NS(ManagedBuffer_get_slot_based_length)(
            sizeof( NS(TriCubData) ), slot_size );

        num_bytes += NS(ManagedBuffer_get_slot_based_length)(
            sizeof( NS(be_tricub_real_t) ) * NS(TriCubData_table_size)( data ),
                slot_size );

        num_slots = num_bytes / slot_size;
        if( ( num_slots * slot_size ) < num_bytes ) ++num_slots;
    }

    return num_slots;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* NS(TriCubData_preset)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    if( data != SIXTRL_NULLPTR )
    {
        NS(arch_status_t) status = NS(TriCubData_set_nx)(
            data, ( NS(be_tricub_int_t) )0.0 );

        status |= NS(TriCubData_set_ny)( data, ( NS(be_tricub_int_t) )0.0 );
        status |= NS(TriCubData_set_nz)( data, ( NS(be_tricub_int_t) )0.0 );
        status |= NS(TriCubData_set_table_addr)(
            data, ( NS(buffer_addr_t) )0.0 );

        status |= NS(TriCubData_clear)( data );

        SIXTRL_ASSERT( status == ( NS(arch_status_t)
            )SIXTRL_ARCH_STATUS_SUCCESS );
        ( void )status;
    }

    return data;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCubData_clear)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data )
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( data != SIXTRL_NULLPTR )
    {
        NS(TriCubData_set_x0)( data, ( NS(be_tricub_real_t) )0.0 );
        NS(TriCubData_set_dx)( data, ( NS(be_tricub_real_t) )0.0 );

        NS(TriCubData_set_y0)( data, ( NS(be_tricub_real_t) )0.0 );
        NS(TriCubData_set_dy)( data, ( NS(be_tricub_real_t) )0.0 );

        NS(TriCubData_set_z0)( data, ( NS(be_tricub_real_t) )0.0 );
        NS(TriCubData_set_dz)( data, ( NS(be_tricub_real_t) )0.0 );

        NS(TriCubData_set_mirror_x)( data, ( NS(be_tricub_int_t) )0 );
        NS(TriCubData_set_mirror_y)( data, ( NS(be_tricub_int_t) )0 );
        NS(TriCubData_set_mirror_z)( data, ( NS(be_tricub_int_t) )0 );

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

    return status;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCubData_x0)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(TriCubData) *const SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->x0;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCubData_dx)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(TriCubData) *const SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->dx;
}

SIXTRL_INLINE NS(be_tricub_int_t) NS(TriCubData_nx)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(TriCubData) *const SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->nx;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCubData_y0)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(TriCubData) *const SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->y0;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCubData_dy)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(TriCubData) *const SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->dy;
}

SIXTRL_INLINE NS(be_tricub_int_t) NS(TriCubData_ny)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(TriCubData) *const SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->ny;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCubData_z0)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(TriCubData) *const SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->z0;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCubData_dz)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(TriCubData) *const SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->dz;
}

SIXTRL_INLINE NS(be_tricub_int_t) NS(TriCubData_nz)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(TriCubData) *const SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->nz;
}

SIXTRL_INLINE NS(be_tricub_int_t) NS(TriCubData_mirror_x)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->mirror_x;
}

SIXTRL_INLINE NS(be_tricub_int_t) NS(TriCubData_mirror_y)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->mirror_y;
}

SIXTRL_INLINE NS(be_tricub_int_t) NS(TriCubData_mirror_z)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->mirror_z;
}

SIXTRL_INLINE NS(be_tricub_int_t) NS(TriCubData_table_size)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->nx * data->ny * data->nz * 8u;
}

SIXTRL_INLINE NS(buffer_addr_t) NS(TriCubData_table_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->table_addr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(be_tricub_real_t) const*
NS(TriCubData_const_table_begin)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(TriCubData) *const SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
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
    SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(be_tricub_real_t)*
        )NS(TriCubData_const_table_begin)( data );
}

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_INLINE NS(arch_status_t) NS(TriCubData_set_x0)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(be_tricub_real_t) const x0 ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    data->x0 = x0;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCubData_set_dx)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
     NS(be_tricub_real_t) const dx ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( data != SIXTRL_NULLPTR ) && ( dx >= ( NS(be_tricub_real_t) )0 ) )
    {
        data->dx = dx;
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCubData_set_nx)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
     NS(be_tricub_int_t)  const nx ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( data != SIXTRL_NULLPTR ) && ( nx >= ( NS(be_tricub_int_t) )0 ) )
    {
        data->nx = nx;
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCubData_set_y0)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
     NS(be_tricub_real_t) const y0 ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    data->y0 = y0;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCubData_set_dy)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
     NS(be_tricub_real_t) const dy ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( data != SIXTRL_NULLPTR ) && ( dy >= ( NS(be_tricub_real_t) )0 ) )
    {
        data->dy = dy;
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCubData_set_ny)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
     NS(be_tricub_int_t)  const ny ) SIXTRL_NOEXCEPT
{
     NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( data != SIXTRL_NULLPTR ) && ( ny >= ( NS(be_tricub_int_t) )0 ) )
    {
        data->ny = ny;
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCubData_set_z0)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
     NS(be_tricub_real_t) const z0 ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    data->z0 = z0;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCubData_set_dz)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
     NS(be_tricub_real_t) const dz ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( data != SIXTRL_NULLPTR ) && ( dz >= ( NS(be_tricub_real_t) )0 ) )
    {
        data->dz = dz;
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCubData_set_nz)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
     NS(be_tricub_int_t)  const nz ) SIXTRL_NOEXCEPT
{
     NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( data != SIXTRL_NULLPTR ) && ( nz >= ( NS(be_tricub_int_t) )0 ) )
    {
        data->nz = nz;
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCubData_set_mirror_x)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
     NS(be_tricub_int_t) const mirror_x ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    data->mirror_x = mirror_x;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCubData_set_mirror_y)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
     NS(be_tricub_int_t) const mirror_y ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    data->mirror_y = mirror_y;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCubData_set_mirror_z)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
     NS(be_tricub_int_t) const mirror_z ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    data->mirror_z = mirror_z;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCubData_set_table_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT data,
    NS(buffer_addr_t) const table_addr ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    data->table_addr = table_addr;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

/*  -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --  */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const*
NS(TriCubData_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
    NS(Object) *const obj ) SIXTRL_NOEXCEPT
{
    return ( ( NS(Object_get_size)( obj ) >= sizeof( NS( TriCubData ) ) ) &&
        ( NS(Object_get_type_id)( obj ) == NS(OBJECT_TYPE_TRICUB_DATA) ) &&
        ( NS(Object_get_begin_addr)( obj ) != ( NS(buffer_addr_t) )0 ) )
            ? ( SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const* )(
                uintptr_t )NS(Object_get_begin_addr)( obj )
            : SIXTRL_NULLPTR;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCubData_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC
    NS(Object)* obj ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
        )NS(TriCubData_const_from_obj_index)( obj );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const*
NS(TriCubData_const_from_managed_buffer)( SIXTRL_BUFFER_DATAPTR_DEC unsigned
        char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(TriCubData_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCubData_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(TriCubData_from_obj_index)(
        NS(ManagedBuffer_get_object)( buffer_begin, index, slot_size ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if !defined( _GPUCODE )
    
SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const*
NS(TriCubData_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(TriCubData_const_from_obj_index)(
        NS(Buffer_get_const_object)( buffer, index ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCubData_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(TriCubData_from_obj_index)(
        NS(Buffer_get_object)( buffer, index ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_size_t) NS(TriCubData_ptr_offset)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData)
        *const SIXTRL_RESTRICT SIXTRL_UNUSED( d ) ) SIXTRL_NOEXCEPT
{
    return ( NS(arch_size_t) )offsetof( NS(TriCubData), table_addr );
}

#endif /* !defined( _GPUCODE ) */
    
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t) NS(TriCubData_copy)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* SIXTRL_RESTRICT dest,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData)
        *const SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT
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
            status |= NS(TriCubData_set_x0)( dest, NS(TriCubData_x0)( src ) );
            status |= NS(TriCubData_set_dx)( dest, NS(TriCubData_dx)( src ) );
            status |= NS(TriCubData_set_nx)( dest, NS(TriCubData_nx)( src ) );

            status |= NS(TriCubData_set_y0)( dest, NS(TriCubData_y0)( src ) );
            status |= NS(TriCubData_set_dy)( dest, NS(TriCubData_dy)( src ) );
            status |= NS(TriCubData_set_ny)( dest, NS(TriCubData_ny)( src ) );

            status |= NS(TriCubData_set_z0)( dest, NS(TriCubData_z0)( src ) );
            status |= NS(TriCubData_set_dz)( dest, NS(TriCubData_dz)( src ) );
            status |= NS(TriCubData_set_nz)( dest, NS(TriCubData_nz)( src ) );

            status |= NS(TriCubData_set_mirror_x)(
                dest, NS(TriCubData_mirror_x)( src ) );

            status |= NS(TriCubData_set_mirror_y)(
                dest, NS(TriCubData_mirror_y)( src ) );

            status |= NS(TriCubData_set_mirror_z)(
                dest, NS(TriCubData_mirror_z)( src ) );
        }
    }
    else if( ( dest == src ) && ( dest != SIXTRL_NULLPTR ) )
    {
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* ========================================================================= */

SIXTRL_INLINE NS(object_type_id_t) NS(TriCub_type_id)( void ) SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(OBJECT_TYPE_TRICUB);
}

SIXTRL_INLINE NS(buffer_size_t) NS(TriCub_num_dataptrs)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT
        SIXTRL_UNUSED( tricub ) ) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(TriCub_num_slots)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const SIXTRL_RESTRICT tricub,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(buffer_size_t) num_slots = ( NS(buffer_size_t) )0u;
    NS(buffer_size_t) const num_bytes = NS(ManagedBuffer_get_slot_based_length)(
        sizeof( NS(TriCub) ), slot_size );

    SIXTRL_ASSERT( slot_size > ( NS(buffer_size_t) )0 );
    SIXTRL_ASSERT( NS(TriCub_num_dataptrs)( tricub ) ==
        ( NS(buffer_size_t) )0u );
    ( void )tricub;

    num_slots = num_bytes / slot_size;
    if( num_slots * slot_size < num_bytes ) ++num_slots;
    return num_slots;
}


SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(TriCub)* NS(TriCub_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT
{
    if( tricub != SIXTRL_NULLPTR )
    {
        NS(TriCub_set_data_addr)( tricub, ( NS(buffer_addr_t) )0u );
        NS(TriCub_clear)( tricub );
    }

    return tricub;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCub_clear)( SIXTRL_BE_ARGPTR_DEC
    NS(TriCub)* SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( tricub != SIXTRL_NULLPTR )
    {
        status  = NS(TriCub_set_x_shift)(
            tricub, ( NS(be_tricub_real_t) )0.0 );

        status |= NS(TriCub_set_y_shift)(
            tricub, ( NS(be_tricub_real_t) )0.0 );

        status |= NS(TriCub_set_tau_shift)(
            tricub, ( NS(be_tricub_real_t) )0.0 );

        status |= NS(TriCub_set_dipolar_kick_px)(
            tricub, ( NS(be_tricub_real_t) )0.0 );

        status |= NS(TriCub_set_dipolar_kick_py)(
            tricub, ( NS(be_tricub_real_t) )0.0 );

        status |= NS(TriCub_set_dipolar_kick_ptau)(
            tricub, ( NS(be_tricub_real_t) )0.0 );

        status |= NS(TriCub_set_length)( tricub, ( NS(be_tricub_real_t) )0.0 );
    }

    return status;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCub_x_shift)( SIXTRL_BE_ARGPTR_DEC
    const NS(TriCub) *const SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->x_shift;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCub_y_shift)( SIXTRL_BE_ARGPTR_DEC
    const NS(TriCub) *const SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->y_shift;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCub_tau_shift)( SIXTRL_BE_ARGPTR_DEC
    const NS(TriCub) *const SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->tau_shift;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCub_dipolar_kick_px)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const
        SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->dipolar_kick_px;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCub_dipolar_kick_py)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const
        SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->dipolar_kick_py;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCub_dipolar_kick_ptau)(
    SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const
        SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->dipolar_kick_ptau;
}

SIXTRL_INLINE NS(be_tricub_real_t) NS(TriCub_length)( SIXTRL_BE_ARGPTR_DEC
    const NS(TriCub) *const SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->length;
}

SIXTRL_INLINE NS(buffer_addr_t) NS(TriCub_data_addr)( SIXTRL_BE_ARGPTR_DEC
    const NS(TriCub) *const SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return tricub->data_addr;
}


SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const*
NS(TriCub_const_data)( SIXTRL_BE_ARGPTR_DEC const NS(TriCub) *const
    SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const* )(
        uintptr_t )tricub->data_addr;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)*
NS(TriCub_data)( SIXTRL_BE_ARGPTR_DEC NS(TriCub)*
    SIXTRL_RESTRICT tricub ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData)* )( uintptr_t
        )tricub->data_addr;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(arch_status_t) NS(TriCub_set_x_shift)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const x_shift ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    tricub->x_shift = x_shift;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCub_set_y_shift)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const y_shift ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    tricub->y_shift = y_shift;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCub_set_tau_shift)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const tau_shift ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    tricub->tau_shift = tau_shift;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCub_set_dipolar_kick_px)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const dipolar_kick_px ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    tricub->dipolar_kick_px = dipolar_kick_px;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCub_set_dipolar_kick_py)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const dipolar_kick_py ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    tricub->dipolar_kick_py = dipolar_kick_py;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCub_set_dipolar_kick_ptau)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const dipolar_kick_ptau ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    tricub->dipolar_kick_ptau = dipolar_kick_ptau;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCub_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(be_tricub_real_t) const length ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    tricub->length = length;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(TriCub_set_data_addr)( SIXTRL_BE_ARGPTR_DEC
    NS(TriCub)* SIXTRL_RESTRICT tricub,
    NS(buffer_addr_t) const data_addr ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( tricub != SIXTRL_NULLPTR );
    tricub->data_addr = data_addr;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

/*  -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --  */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(TriCub) const*
NS(TriCub_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
    NS(Object) *const obj ) SIXTRL_NOEXCEPT
{
    return ( ( NS(Object_get_size)( obj ) >= sizeof( NS(TriCub) ) ) &&
        ( NS(Object_get_type_id)( obj ) == NS(OBJECT_TYPE_TRICUB) ) &&
        ( NS(Object_get_begin_addr)( obj ) != ( NS(buffer_addr_t) )0 ) )
        ? ( SIXTRL_BE_ARGPTR_DEC NS(TriCub) const* )( uintptr_t
            )NS(Object_get_begin_addr)( obj )
        : SIXTRL_NULLPTR;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(TriCub)* NS(TriCub_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* obj ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_ARGPTR_DEC NS(TriCub)*
        )NS(TriCub_const_from_obj_index)( obj );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(TriCub) const*
NS(TriCub_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(TriCub_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(TriCub)* NS(TriCub_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(TriCub_from_obj_index)(
        NS(ManagedBuffer_get_object)( buffer_begin, index, slot_size ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub) const*
NS(TriCub_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(TriCub_const_from_obj_index)(
        NS(Buffer_get_const_object)( buffer, index ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)* NS(TriCub_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(TriCub_from_obj_index)( NS(Buffer_get_object)( buffer, index ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_size_t) NS(TriCub_data_addr_offset)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( tricub ) ) SIXTRL_NOEXCEPT
{
    return ( NS(arch_size_t) )offsetof( NS(TriCub), data_addr );
}

#endif /* !defined( _GPUCODE ) */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t) NS(TriCub_copy)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCub)* SIXTRL_RESTRICT dest,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCub) *const
        SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( dest != SIXTRL_NULLPTR ) && ( src != SIXTRL_NULLPTR ) )
    {
        if( dest != src )
        {
            status  = NS(TriCub_set_x_shift)( dest, NS(TriCub_x_shift)( src ) );
            status |= NS(TriCub_set_y_shift)( dest, NS(TriCub_y_shift)( src ) );
            status |= NS(TriCub_set_tau_shift)(
                dest, NS(TriCub_tau_shift)( src ) );

            status |= NS(TriCub_set_dipolar_kick_px)(
                dest, NS(TriCub_dipolar_kick_px)( src ) );

            status |= NS(TriCub_set_dipolar_kick_py)(
                dest, NS(TriCub_dipolar_kick_py)( src ) );

            status |= NS(TriCub_set_dipolar_kick_ptau)(
                dest, NS(TriCub_dipolar_kick_ptau)( src ) );

            status |= NS(TriCub_set_length)( dest, NS(TriCub_length)( src ) );
            status |= NS(TriCub_set_data_addr)(
                dest, NS(TriCub_data_addr)( src ) );
        }
        else
        {
            status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
        }
    }

    return status;
}

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* !defined( __cplusplus ) && !defined( _GPUCODE ) */
#endif /* SIXTRACKLIB_COMMON_BE_TRICUB_C99_H__ */
