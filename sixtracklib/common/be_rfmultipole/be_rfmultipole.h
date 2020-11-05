#ifndef SIXTRACKL_COMMON_BE_RFMULTIPOLE_BE_RFMULTIPOLE_C99_H__
#define SIXTRACKL_COMMON_BE_RFMULTIPOLE_BE_RFMULTIPOLE_C99_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #if !defined( _GPUCODE )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>
    #include <stdint.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

typedef SIXTRL_REAL_T  NS(rf_multipole_real_t);
typedef SIXTRL_INT64_T NS(rf_multipole_int_t);

typedef struct NS(RFMultipole)
{
    NS(rf_multipole_int_t)    order               SIXTRL_ALIGN( 8 );
    NS(rf_multipole_real_t)   voltage             SIXTRL_ALIGN( 8 );
    NS(rf_multipole_real_t)   frequency           SIXTRL_ALIGN( 8 );
    NS(rf_multipole_real_t)   lag                 SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t)         bal_addr            SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t)         phase_addr          SIXTRL_ALIGN( 8 );
}
NS(RFMultipole);

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t)
    NS(RFMultipole_type_id)( void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(RFMultipole_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(RFMultipole_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)*
NS(RFMultipole_preset)( SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)*
    SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT
        multipole ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(rf_multipole_real_t) NS(RFMultipole_voltage)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(rf_multipole_real_t) NS(RFMultipole_frequency)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(rf_multipole_real_t) NS(RFMultipole_lag)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(rf_multipole_int_t) NS(RFMultipole_order )(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(RFMultipole_bal_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(RFMultipole_phase_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t) NS(RFMultipole_bal_addr)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
NS(RFMultipole_const_bal_begin)( SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole)
    *const SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
NS(RFMultipole_const_bal_end)( SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole)
    *const SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(rf_multipole_real_t) NS(RFMultipole_bal)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const bal_index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(rf_multipole_real_t) NS(RFMultipole_knl)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const knl_index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(rf_multipole_real_t) NS(RFMultipole_ksl)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const ksl_index ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t) NS(RFMultipole_phase_addr)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
NS(RFMultipole_const_phase_begin)( SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole)
    *const SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
NS(RFMultipole_const_phase_end)( SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole)
    *const SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(rf_multipole_real_t) NS(RFMultipole_phase)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const phase_index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(rf_multipole_real_t) NS(RFMultipole_phase_n)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const pn_index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(rf_multipole_real_t) NS(RFMultipole_phase_s)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const ps_index ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_voltage)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const voltage ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_frequency)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const frequency ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_lag)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const lag ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_order )(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const order ) SIXTRL_NOEXCEPT;

/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_bal_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(buffer_addr_t) const bal_addr ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
NS(RFMultipole_bal_begin)( SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)*
    SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
NS(RFMultipole_bal_end)( SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)*
    SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_bal_value)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const bal_index,
    NS(rf_multipole_real_t) const bal_value ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_all_bal_values)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const bal_value ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_bal)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const* SIXTRL_RESTRICT
        bal_values_begin ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_knl_value)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const knl_index,
    NS(rf_multipole_real_t) const knl_value ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_all_knl_values)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const knl_value ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_knl)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
        SIXTRL_RESTRICT knl_values_begin ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_ksl_value)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const ksl_index,
    NS(rf_multipole_real_t) const ksl_value ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_all_ksl_values)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const ksl_value ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_ksl)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
        SIXTRL_RESTRICT ksl_values_begin ) SIXTRL_NOEXCEPT;

/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_phase_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(buffer_addr_t) const phase_addr ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
NS(RFMultipole_phase_begin)( SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)*
    SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
NS(RFMultipole_phase_end)( SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)*
    SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_phase_value)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const phase_index,
    NS(rf_multipole_real_t) const phase_value ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_all_phase_values)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const phase_value ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_phase)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
        SIXTRL_RESTRICT phase_values_begin ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_phase_n_value)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const pn_index,
    NS(rf_multipole_real_t) const pn_value ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_all_phase_n_values)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const pn_value ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_phase_n)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
        SIXTRL_RESTRICT pn_values ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_phase_s_value)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const ps_index,
    NS(rf_multipole_real_t) const ps_value ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_all_phase_s_values)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const ps_value ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_set_phase_s)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
        SIXTRL_RESTRICT ps_values ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultipole_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT dst,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultipole) const*
NS(RFMultipole_const_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)*
NS(RFMultipole_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultipole) const*
NS(RFMultipole_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)*
NS(RFMultipole_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultipole) const*
NS(RFMultipole_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)*
NS(RFMultipole_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(RFMultipole_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(RFMultipole_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(RFMultipole_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(object_type_id_t)
    NS(RFMultipole_type_id_ext)( void ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(RFMultipole_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(rf_multipole_int_t) const order,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_requ_dataptrs ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)*
NS(RFMultipole_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(rf_multipole_int_t) const order );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)*
NS(RFMultipole_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(rf_multipole_int_t) const order,
    NS(rf_multipole_real_t) const voltage,
    NS(rf_multipole_real_t) const frequency,
    NS(rf_multipole_real_t) const lag,
    NS(buffer_addr_t) const ext_bal_addr,
    NS(buffer_addr_t) const ext_phase_addr );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)*
NS(RFMultipole_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT orig );

#endif /* !defined( _GPUCODE ) */
#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer/managed_buffer.h"
    #include "sixtracklib/common/internal/math_factorial.h"
    #include "sixtracklib/common/internal/math_constants.h"
    #include "sixtracklib/common/internal/math_functions.h"
    #include "sixtracklib/common/internal/type_comparison_helpers.hpp"

    #if !defined( _GPUCODE )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_INLINE NS(object_type_id_t)
    NS(RFMultipole_type_id)( void ) SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(OBJECT_TYPE_RF_MULTIPOLE);
}

SIXTRL_INLINE NS(buffer_size_t) NS(RFMultipole_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( mpole ) ) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )2u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(RFMultipole_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) buf_size_t;
    buf_size_t num_slots = ( buf_size_t )0u;

    if( ( slot_size > ( buf_size_t )0 ) && ( mpole != SIXTRL_NULLPTR ) )
    {
        buf_size_t num_bytes = NS(ManagedBuffer_get_slot_based_length)(
            sizeof( NS(RFMultipole) ), slot_size );

        buf_size_t const n_bal_vals = NS(RFMultipole_bal_length)( mpole );
        buf_size_t const n_phase_vals = NS(RFMultipole_phase_length)( mpole );

        num_bytes += NS(ManagedBuffer_get_slot_based_length)(
            sizeof( NS(rf_multipole_real_t) ) * n_bal_vals, slot_size );

        num_bytes += NS(ManagedBuffer_get_slot_based_length)(
            sizeof( NS(rf_multipole_real_t) ) * n_phase_vals, slot_size );


        num_slots = num_bytes / slot_size;
        if( num_slots * slot_size < num_bytes ) ++num_slots;
    }

    return num_slots;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* NS(RFMultipole_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT
        multipole ) SIXTRL_NOEXCEPT
{
    if( multipole != SIXTRL_NULLPTR ) NS(RFMultipole_clear)( multipole );
    return multipole;
}

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_clear)( SIXTRL_BE_ARGPTR_DEC
    NS(RFMultipole)* SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT
{
    typedef NS(rf_multipole_real_t) real_t;
    typedef NS(rf_multipole_int_t)  order_t;

    NS(arch_status_t) status =
        NS(RFMultipole_set_order)( multipole, ( order_t )0 );

    status |= NS(RFMultipole_set_voltage)( multipole, ( real_t )0 );
    status |= NS(RFMultipole_set_frequency)( multipole, ( real_t )0 );
    status |= NS(RFMultipole_set_lag)( multipole, ( real_t )0 );
    status |= NS(RFMultipole_set_bal_addr)( multipole, ( NS(buffer_addr_t) )0 );
    status |= NS(RFMultipole_set_phase_addr)(
        multipole, ( NS(buffer_addr_t) )0 );

    return status;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(rf_multipole_real_t) NS(RFMultipole_voltage)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return mpole->voltage;
}

SIXTRL_INLINE NS(rf_multipole_real_t) NS(RFMultipole_frequency)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return mpole->frequency;
}

SIXTRL_INLINE NS(rf_multipole_real_t) NS(RFMultipole_lag)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return mpole->lag;
}

SIXTRL_INLINE NS(rf_multipole_int_t) NS(RFMultipole_order )(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return mpole->order;
}

SIXTRL_INLINE NS(buffer_size_t) NS(RFMultipole_bal_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( mpole->order >= ( NS(rf_multipole_int_t) )0 );
    return ( NS(buffer_size_t) )( 2 * mpole->order + 2 );
}

SIXTRL_INLINE NS(buffer_size_t) NS(RFMultipole_phase_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( mpole->order >= ( NS(rf_multipole_int_t) )0 );
    return ( NS(buffer_size_t) )( 2 * mpole->order + 2 );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_addr_t) NS(RFMultipole_bal_addr)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return mpole->bal_addr;
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
NS(RFMultipole_const_bal_begin)( SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole)
    *const SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return ( SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const* )( uintptr_t
        )mpole->bal_addr;
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
NS(RFMultipole_const_bal_end)( SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole)
    *const SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT
{
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const* bal_end =
        NS(RFMultipole_const_bal_begin)( mpole );

    if( bal_end != SIXTRL_NULLPTR )
    {
        SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(RFMultipole_order)( mpole ) >=
            ( NS(rf_multipole_int_t) )0 );

        bal_end = bal_end + NS(RFMultipole_bal_length)( mpole );
    }

    return bal_end;
}

SIXTRL_INLINE NS(rf_multipole_real_t) NS(RFMultipole_bal)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const bal_index ) SIXTRL_NOEXCEPT
{
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const* bal_values =
        NS(RFMultipole_const_bal_begin)( mpole );

    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( bal_values != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(RFMultipole_bal_length)( mpole ) > bal_index );

    return bal_values[ bal_index ];
}

SIXTRL_INLINE NS(rf_multipole_real_t) NS(RFMultipole_knl)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const knl_index ) SIXTRL_NOEXCEPT
{
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const* bal_values =
        NS(RFMultipole_const_bal_begin)( mpole );

    NS(buffer_size_t) const bal_index = ( NS(buffer_size_t) )( 2 * knl_index );

    SIXTRL_ASSERT( mpole      != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( bal_values != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( knl_index  >= ( NS(rf_multipole_int_t) )0 );
    SIXTRL_ASSERT( knl_index  <= NS(RFMultipole_order)( mpole ) );
    SIXTRL_ASSERT( NS(RFMultipole_bal_length)( mpole ) > bal_index );

    return bal_values[ bal_index ] * NS(Math_factorial)( knl_index );
}

SIXTRL_INLINE NS(rf_multipole_real_t) NS(RFMultipole_ksl)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const ksl_index ) SIXTRL_NOEXCEPT
{
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const* bal_values =
        NS(RFMultipole_const_bal_begin)( mpole );

    NS(buffer_size_t) const bal_index = (
        NS(buffer_size_t) )( 2 * ksl_index +  1 );

    SIXTRL_ASSERT( mpole      != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( bal_values != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ksl_index  >= ( NS(rf_multipole_int_t) )0 );
    SIXTRL_ASSERT( ksl_index  <= NS(RFMultipole_order)( mpole ) );
    SIXTRL_ASSERT( NS(RFMultipole_bal_length)( mpole ) > bal_index );

    return bal_values[ bal_index ] * NS(Math_factorial)( ksl_index );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_addr_t) NS(RFMultipole_phase_addr)( SIXTRL_BE_ARGPTR_DEC
    const NS(RFMultipole) *const SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return mpole->phase_addr;
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
NS(RFMultipole_const_phase_begin)( SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole)
    *const SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return ( SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const* )( uintptr_t
        )mpole->phase_addr;
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
NS(RFMultipole_const_phase_end)( SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole)
    *const SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT
{
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const* phase_end =
        NS(RFMultipole_const_phase_begin)( mpole );

    if( phase_end != SIXTRL_NULLPTR )
    {
        SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(RFMultipole_order)( mpole ) >=
            ( NS(rf_multipole_int_t) )0 );

        phase_end = phase_end + NS(RFMultipole_phase_length)( mpole );
    }

    return phase_end;
}

SIXTRL_INLINE NS(rf_multipole_real_t) NS(RFMultipole_phase)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const phase_index ) SIXTRL_NOEXCEPT
{
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const* phase_values_begin =
        NS(RFMultipole_const_phase_begin)( mpole );

    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( phase_values_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(RFMultipole_phase_length)( mpole ) > phase_index );

    return phase_values_begin[ phase_index ];
}

SIXTRL_INLINE NS(rf_multipole_real_t) NS(RFMultipole_phase_n)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const pn_index ) SIXTRL_NOEXCEPT
{
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const* phase_values_begin =
        NS(RFMultipole_const_phase_begin)( mpole );

    NS(buffer_size_t) const phase_index = ( NS(buffer_size_t) )( 2 * pn_index );

    SIXTRL_ASSERT( mpole      != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( pn_index   >= ( NS(rf_multipole_int_t) )0 );
    SIXTRL_ASSERT( pn_index   <= NS(RFMultipole_order)( mpole ) );
    SIXTRL_ASSERT( phase_values_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(RFMultipole_phase_length)( mpole ) > phase_index );

    return phase_values_begin[ phase_index ];
}

SIXTRL_INLINE NS(rf_multipole_real_t) NS(RFMultipole_phase_s)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultipole) *const SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const ps_index ) SIXTRL_NOEXCEPT
{
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const* phase_values_begin =
        NS(RFMultipole_const_phase_begin)( mpole );

    NS(buffer_size_t) const phase_index = (
        NS(buffer_size_t) )( 2 * ps_index + 1 );

    SIXTRL_ASSERT( mpole      != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ps_index   >= ( NS(rf_multipole_int_t) )0 );
    SIXTRL_ASSERT( ps_index   <= NS(RFMultipole_order)( mpole ) );
    SIXTRL_ASSERT( phase_values_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(RFMultipole_phase_length)( mpole ) > phase_index );

    return phase_values_begin[ phase_index ];
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_voltage)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const voltage ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    mpole->voltage = voltage;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_frequency)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const frequency ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    mpole->frequency = frequency;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_lag)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const lag ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    mpole->lag = lag;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_order )(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const order ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( order >= ( NS(rf_multipole_int_t) )0 );
    mpole->order = order;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_bal_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(buffer_addr_t) const bal_addr ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    mpole->bal_addr = bal_addr;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
NS(RFMultipole_bal_begin)( SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)*
    SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return ( SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* )( uintptr_t
        )mpole->bal_addr;
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
NS(RFMultipole_bal_end)( SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)*
    SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
        )NS(RFMultipole_const_bal_end)( mpole );
}

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_bal_value)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const bal_index,
    NS(rf_multipole_real_t) const bal_value ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( mpole->bal_addr != ( NS(buffer_addr_t) )0 );
    SIXTRL_ASSERT( bal_index < NS(RFMultipole_bal_length)( mpole ) );

    NS(RFMultipole_bal_begin)( mpole )[ bal_index ] = bal_value;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_all_bal_values)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const bal_value ) SIXTRL_NOEXCEPT
{
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* bal_it =
        NS(RFMultipole_bal_begin)( mpole );

    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* bal_end =
        NS(RFMultipole_bal_end)( mpole );

    SIXTRL_ASSERT( bal_it  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( bal_end != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( ( uintptr_t )bal_it ) <= ( uintptr_t )bal_end );

    for( ; bal_it != bal_end ; ++bal_it ) *bal_it = bal_value;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_bal)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const* SIXTRL_RESTRICT
        in_bal_it ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* bal_it =
        NS(RFMultipole_bal_begin)( mpole );

    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* bal_end =
        NS(RFMultipole_bal_end)( mpole );

    if( ( bal_it != SIXTRL_NULLPTR ) && ( bal_end != SIXTRL_NULLPTR ) &&
        ( ( ( uintptr_t )bal_it ) <= ( uintptr_t )bal_end ) &&
        ( in_bal_it != SIXTRL_NULLPTR ) )
    {
        for( ; bal_it != bal_end ; ++bal_it, ++in_bal_it ) *bal_it = *in_bal_it;
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_knl_value)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const knl_index,
    NS(rf_multipole_real_t) const knl_value ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* bal_values =
        NS(RFMultipole_bal_begin)( mpole );

    NS(buffer_size_t) const bal_index = ( NS(buffer_size_t) )( 2 * knl_index );

    if( ( mpole != SIXTRL_NULLPTR ) && ( bal_values != SIXTRL_NULLPTR ) &&
        ( knl_index >= ( NS(rf_multipole_int_t) )0 ) &&
        ( knl_index >= NS(RFMultipole_order)( mpole ) ) &&
        ( bal_index <  NS(RFMultipole_bal_length)( mpole ) ) )
    {
        bal_values[ bal_index ] = knl_value / NS(Math_factorial)( knl_index );
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_all_knl_values)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const knl_value ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* bal_values =
        NS(RFMultipole_bal_begin)( mpole );

    NS(buffer_size_t) bal_index = ( NS(buffer_size_t) )0u;
    NS(rf_multipole_int_t) knl_index = ( NS(rf_multipole_int_t) )0;
    NS(rf_multipole_int_t) const order = NS(RFMultipole_order)( mpole );

    if( ( mpole != SIXTRL_NULLPTR ) && ( bal_values != SIXTRL_NULLPTR ) )
    {
        for( ; knl_index <= order ; ++knl_index, bal_index += 2u )
        {
            bal_values[ bal_index ] = knl_value /
                NS(Math_factorial)( knl_index );
        }

        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_knl)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
        SIXTRL_RESTRICT in_knl_it ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* bal_values =
        NS(RFMultipole_bal_begin)( mpole );

    NS(buffer_size_t) bal_index = ( NS(buffer_size_t) )0u;
    NS(rf_multipole_int_t) knl_index = ( NS(rf_multipole_int_t) )0;
    NS(rf_multipole_int_t) const order = NS(RFMultipole_order)( mpole );

    if( ( mpole != SIXTRL_NULLPTR ) && ( bal_values != SIXTRL_NULLPTR ) &&
        ( in_knl_it != SIXTRL_NULLPTR ) )
    {
        for( ; knl_index <= order ; ++knl_index, bal_index += 2u, ++in_knl_it )
        {
            bal_values[ bal_index ] = ( *in_knl_it ) /
                NS(Math_factorial)( knl_index );
        }

        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_ksl_value)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const ksl_index,
    NS(rf_multipole_real_t) const ksl_value ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* bal_values =
        NS(RFMultipole_bal_begin)( mpole );

    NS(buffer_size_t) const bal_index = (
        NS(buffer_size_t) )( 2 * ksl_index + 1 );

    if( ( mpole != SIXTRL_NULLPTR ) && ( bal_values != SIXTRL_NULLPTR ) &&
        ( ksl_index >= ( NS(rf_multipole_int_t) )0 ) &&
        ( ksl_index >= NS(RFMultipole_order)( mpole ) ) &&
        ( bal_index <  NS(RFMultipole_bal_length)( mpole ) ) )
    {
        bal_values[ bal_index ] = ksl_value / NS(Math_factorial)( ksl_index );
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_all_ksl_values)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const ksl_value ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* bal_values =
        NS(RFMultipole_bal_begin)( mpole );

    NS(buffer_size_t) bal_index = ( NS(buffer_size_t) )0u;
    NS(rf_multipole_int_t) ksl_index = ( NS(rf_multipole_int_t) )1;
    NS(rf_multipole_int_t) const order = NS(RFMultipole_order)( mpole );

    if( ( mpole != SIXTRL_NULLPTR ) && ( bal_values != SIXTRL_NULLPTR ) )
    {
        for( ; ksl_index <= order ; ++ksl_index, bal_index += 2u )
        {
            bal_values[ bal_index ] = ksl_value /
                NS(Math_factorial)( ksl_index );
        }

        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_ksl)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
        SIXTRL_RESTRICT in_ksl_it ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* bal_values =
        NS(RFMultipole_bal_begin)( mpole );

    NS(buffer_size_t) bal_index = ( NS(buffer_size_t) )0u;
    NS(rf_multipole_int_t) ksl_index = ( NS(rf_multipole_int_t) )1;
    NS(rf_multipole_int_t) const order = NS(RFMultipole_order)( mpole );

    if( ( mpole != SIXTRL_NULLPTR ) && ( bal_values != SIXTRL_NULLPTR ) &&
        ( in_ksl_it != SIXTRL_NULLPTR ) )
    {
        for( ; ksl_index <= order ; ++ksl_index, bal_index += 2u, ++in_ksl_it )
        {
            bal_values[ bal_index ] = ( *in_ksl_it ) /
                NS(Math_factorial)( ksl_index );
        }

        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_phase_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(buffer_addr_t) const phase_addr ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    mpole->phase_addr = phase_addr;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
NS(RFMultipole_phase_begin)( SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)*
    SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return ( SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* )( uintptr_t
        )mpole->phase_addr;
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
NS(RFMultipole_phase_end)( SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)*
    SIXTRL_RESTRICT mpole ) SIXTRL_NOEXCEPT
{
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* phase_values_end =
        NS(RFMultipole_bal_begin)( mpole );

    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );

    if( phase_values_end != SIXTRL_NULLPTR )
    {
        phase_values_end = phase_values_end +
            NS(RFMultipole_phase_length)( mpole );
    }

    return phase_values_end;
}

/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_phase_value)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const phase_index,
    NS(rf_multipole_real_t) const phase_value ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( mpole->phase_addr != ( NS(buffer_addr_t) )0 );
    SIXTRL_ASSERT( phase_index < NS(RFMultipole_phase_length)( mpole ) );

    NS(RFMultipole_phase_begin)( mpole )[ phase_index ] = phase_value;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_all_phase_values)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const phase_value ) SIXTRL_NOEXCEPT
{
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* phase_it =
        NS(RFMultipole_phase_begin)( mpole );

    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* phase_end =
        NS(RFMultipole_phase_end)( mpole );

    SIXTRL_ASSERT( phase_it  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( phase_end != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( ( uintptr_t )phase_it ) <= ( uintptr_t )phase_end );

    for( ; phase_it != phase_end ; ++phase_it ) *phase_it = phase_value;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_phase)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
        SIXTRL_RESTRICT in_phase_it ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* phase_it =
        NS(RFMultipole_bal_begin)( mpole );

    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* phase_end =
        NS(RFMultipole_bal_end)( mpole );

    if( ( phase_it != SIXTRL_NULLPTR ) && ( phase_end != SIXTRL_NULLPTR ) &&
        ( ( ( uintptr_t )phase_it ) <= ( uintptr_t )phase_end ) &&
        ( in_phase_it != SIXTRL_NULLPTR ) )
    {
        for( ; phase_it != phase_end ; ++phase_it, ++in_phase_it )
        {
            *phase_it = *in_phase_it;
        }

        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_phase_n_value)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const pn_index,
    NS(rf_multipole_real_t) const pn_value ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* phase_values =
        NS(RFMultipole_phase_begin)( mpole );

    NS(buffer_size_t) const phase_index = ( NS(buffer_size_t) )( 2 * pn_index );

    if( ( mpole != SIXTRL_NULLPTR ) && ( phase_values != SIXTRL_NULLPTR ) &&
        ( pn_index >= ( NS(rf_multipole_int_t) )0 ) &&
        ( pn_index >= NS(RFMultipole_order)( mpole ) ) &&
        ( phase_index <  NS(RFMultipole_bal_length)( mpole ) ) )
    {
        phase_values[ phase_index ] = pn_value;
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_all_phase_n_values)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const pn_value ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* phase_values =
        NS(RFMultipole_phase_begin)( mpole );

    NS(buffer_size_t) phase_index = ( NS(buffer_size_t) )0u;
    NS(rf_multipole_int_t) pn_index = ( NS(rf_multipole_int_t) )0;
    NS(rf_multipole_int_t) const order = NS(RFMultipole_order)( mpole );

    if( ( mpole != SIXTRL_NULLPTR ) && ( phase_values != SIXTRL_NULLPTR ) )
    {
        for( ; pn_index <= order ; ++pn_index, phase_index += 2u )
        {
            phase_values[ pn_index ] = pn_value;
        }

        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_phase_n)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
        SIXTRL_RESTRICT in_pn_it ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* phase_values =
        NS(RFMultipole_phase_begin)( mpole );

    NS(buffer_size_t) phase_index = ( NS(buffer_size_t) )0u;
    NS(rf_multipole_int_t) pn_index = ( NS(rf_multipole_int_t) )0;
    NS(rf_multipole_int_t) const order = NS(RFMultipole_order)( mpole );

    if( ( mpole != SIXTRL_NULLPTR ) && ( phase_values != SIXTRL_NULLPTR ) &&
        ( in_pn_it != SIXTRL_NULLPTR ) )
    {
        for( ; pn_index <= order ; ++pn_index, phase_index += 2u, ++in_pn_it )
        {
            phase_values[ phase_index ] = *in_pn_it;
        }

        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_phase_s_value)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const ps_index,
    NS(rf_multipole_real_t) const ps_value ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* phase_values =
        NS(RFMultipole_phase_begin)( mpole );

    NS(buffer_size_t) const phase_index = (
        NS(buffer_size_t) )( 2 * ps_index + 1 );

    if( ( mpole != SIXTRL_NULLPTR ) && ( phase_values != SIXTRL_NULLPTR ) &&
        ( ps_index >= ( NS(rf_multipole_int_t) )0 ) &&
        ( ps_index >= NS(RFMultipole_order)( mpole ) ) &&
        ( phase_index <  NS(RFMultipole_bal_length)( mpole ) ) )
    {
        phase_values[ phase_index ] = ps_value;
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_all_phase_s_values)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const ps_value ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* phase_values =
        NS(RFMultipole_phase_begin)( mpole );

    NS(buffer_size_t) phase_index = ( NS(buffer_size_t) )0u;
    NS(rf_multipole_int_t) ps_index = ( NS(rf_multipole_int_t) )1;
    NS(rf_multipole_int_t) const order = NS(RFMultipole_order)( mpole );

    if( ( mpole != SIXTRL_NULLPTR ) && ( phase_values != SIXTRL_NULLPTR ) )
    {
        for( ; ps_index <= order ; ++ps_index, phase_index += 2u )
        {
            phase_values[ ps_index ] = ps_value;
        }

        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_set_phase_s)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT mpole,
    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
        SIXTRL_RESTRICT in_ps_it ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)* phase_values =
        NS(RFMultipole_phase_begin)( mpole );

    NS(buffer_size_t) phase_index = ( NS(buffer_size_t) )0u;
    NS(rf_multipole_int_t) ps_index = ( NS(rf_multipole_int_t) )1;
    NS(rf_multipole_int_t) const order = NS(RFMultipole_order)( mpole );

    if( ( mpole != SIXTRL_NULLPTR ) && ( phase_values != SIXTRL_NULLPTR ) &&
        ( in_ps_it != SIXTRL_NULLPTR ) )
    {
        for( ; ps_index <= order ; ++ps_index, phase_index += 2u, ++in_ps_it )
        {
            phase_values[ phase_index ] = *in_ps_it;
        }

        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t) NS(RFMultipole_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)* SIXTRL_RESTRICT dst,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(RFMultipole) *const
        SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( dst != SIXTRL_NULLPTR ) && ( src != SIXTRL_NULLPTR ) )
    {
        if( ( dst != src ) &&
            ( NS(RFMultipole_order)( dst ) == NS(RFMultipole_order)( src ) ) )
        {
            SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
                in_phase_it = NS(RFMultipole_const_phase_begin)( src );

            SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
                in_phase_end = NS(RFMultipole_const_phase_end)( src );

            SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
                dst_phase_it = NS(RFMultipole_phase_begin)( dst );

            SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
                in_bal_it = NS(RFMultipole_const_bal_begin)( src );

            SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
                in_bal_end = NS(RFMultipole_const_bal_end)( src );

            SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
                dst_bal_it = NS(RFMultipole_bal_begin)( dst );

            status = ( NS(arch_status_t)
                )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

            if( ( in_phase_it  != SIXTRL_NULLPTR ) &&
                ( in_phase_end != SIXTRL_NULLPTR ) &&
                ( dst_phase_it != SIXTRL_NULLPTR ) )
            {
                for( ; in_phase_it != in_phase_end ;
                        ++in_phase_it, ++dst_phase_it )
                {
                    *dst_phase_it = *in_phase_it;
                }

                status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
            }
            else if( ( in_phase_it  == SIXTRL_NULLPTR ) &&
                     ( in_phase_end == SIXTRL_NULLPTR ) &&
                     ( dst_phase_it == SIXTRL_NULLPTR ) )
            {
                status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
            }

            if( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS )
            {
                status = (
                    NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

                if( ( in_bal_it  != SIXTRL_NULLPTR ) &&
                    ( in_bal_end != SIXTRL_NULLPTR ) &&
                    ( dst_bal_it != SIXTRL_NULLPTR ) )
                {
                    for( ; in_bal_it != in_bal_end ;
                            ++in_bal_it, ++dst_bal_it )
                    {
                        *dst_bal_it = *in_bal_it;
                    }

                    status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
                }
                else if( ( in_bal_it  == SIXTRL_NULLPTR ) &&
                         ( in_bal_end == SIXTRL_NULLPTR ) &&
                         ( dst_bal_it == SIXTRL_NULLPTR ) )
                {
                    status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
                }
            }

            if( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS )
            {
                status = NS(RFMultipole_set_voltage)(
                    dst, NS(RFMultipole_voltage)( src ) );

                status |= NS(RFMultipole_set_frequency)(
                    dst, NS(RFMultipole_frequency)( src ) );

                status |= NS(RFMultipole_set_lag)(
                    dst, NS(RFMultipole_lag)( src ) );
            }
        }
        else if( src == dst )
        {
            status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
        }
    }

    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(RFMultipole) const*
NS(RFMultipole_const_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const obj ) SIXTRL_NOEXCEPT
{
    SIXTRL_BE_ARGPTR_DEC NS(RFMultipole) const* elem = SIXTRL_NULLPTR;
    if( ( obj != SIXTRL_NULLPTR ) &&
        ( NS(Object_get_type_id)( obj ) == NS(OBJECT_TYPE_RF_MULTIPOLE) ) &&
        ( NS(Object_get_size)( obj ) >= sizeof( NS(RFMultipole) ) ) )
    {
        elem = ( SIXTRL_BE_ARGPTR_DEC NS(RFMultipole) const* )(
            uintptr_t )NS(Object_get_begin_addr)( obj );
    }

    return elem;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)*
NS(RFMultipole_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* obj ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)*
        )NS(RFMultipole_const_from_obj_index)( obj );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(RFMultipole) const*
NS(RFMultipole_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(RFMultipole_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)*
NS(RFMultipole_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(RFMultipole_from_obj_index)(
        NS(ManagedBuffer_get_object)( buffer_begin, index, slot_size ) );
}

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(RFMultipole) const*
NS(RFMultipole_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(RFMultipole_const_from_obj_index)( NS(Buffer_get_const_object)(
        buffer, index ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(RFMultipole)*
NS(RFMultipole_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(RFMultipole_from_obj_index)( NS(Buffer_get_object)(
        buffer, index ) );
}

#endif /* Host */
#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */
#endif /* SIXTRACKL_COMMON_BE_RFMULTIPOLE_BE_RFMULTIPOLE_C99_H__ */
