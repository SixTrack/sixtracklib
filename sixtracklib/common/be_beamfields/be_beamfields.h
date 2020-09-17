#ifndef SIXTRACKLIB_COMMON_BE_BEAMFIELDS_BE_BEAMFIELDS_C99_H__
#define SIXTRACKLIB_COMMON_BE_BEAMFIELDS_BE_BEAMFIELDS_C99_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/compiler_attributes.h"
    #include "sixtracklib/common/internal/math_interpol.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/be_beamfields/gauss_fields.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#ifndef SIXTRL_BB_GET_PTR
    #define SIXTRL_BB_GET_PTR(dataptr,name) \
        (SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T*)(((SIXTRL_BE_DATAPTR_DEC SIXTRL_UINT64_T*) \
        (&((dataptr)->name))) + ((SIXTRL_UINT64_T) (dataptr)->name) + 1)
#endif /* defined( SIXTRL_BB_GET_PTR ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* BeamBeam4D: */

typedef struct NS(BeamBeam4D)
{
    SIXTRL_UINT64_T   data_size SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t) data_addr SIXTRL_ALIGN( 8 );
}
NS(BeamBeam4D);

typedef struct
{
    SIXTRL_REAL_T q_part            SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T N_part            SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T sigma_x           SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T sigma_y           SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T beta_s            SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T min_sigma_diff    SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T Delta_x           SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T Delta_y           SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T Dpx_sub           SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T Dpy_sub           SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T enabled           SIXTRL_ALIGN( 8 );
}NS(BB4D_data);

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t) NS(BeamBeam4D_type_id)(
    void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(BeamBeam4D_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const
        SIXTRL_RESTRICT elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(BeamBeam4D_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT elem,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam );

SIXTRL_STATIC SIXTRL_FN void NS(BeamBeam4D_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t) NS(BeamBeam4D_data_addr)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const
        SIXTRL_RESTRICT bb_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(BeamBeam4D_set_data_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT bb_elem,
    NS(buffer_addr_t) const data_addr ) SIXTRL_NOEXCEPT;


SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(BB4D_data) const*
NS(BeamBeam4D_const_data)( SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const
    SIXTRL_RESTRICT bb_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(BB4D_data)*
NS(BeamBeam4D_data)( SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)*
    SIXTRL_RESTRICT bb_elem ) SIXTRL_NOEXCEPT;


SIXTRL_STATIC SIXTRL_FN NS(arch_size_t) NS(BeamBeam4D_data_size)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const
        SIXTRL_RESTRICT bb_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(BeamBeam4D_set_data_size)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT bb_elem,
    NS(arch_size_t) const data_size ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D) const*
NS(BeamBeam4D_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
    NS(Object) *const SIXTRL_RESTRICT obj_index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
    SIXTRL_RESTRICT obj_index ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D) const*
NS(BeamBeam4D_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D) const*
NS(BeamBeam4D_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(BeamBeam4D_data_addr_offset)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const
        SIXTRL_RESTRICT elem ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(object_type_id_t)
NS(BeamBeam4D_type_id_ext)( void ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BeamBeam4D_data_addr_offset_ext)( SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D)
    *const SIXTRL_RESTRICT bb_elem ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(BeamBeam4D_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_size,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_requ_dataptrs ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_size );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const  data_size,
    NS(buffer_addr_t) const bb4d_data_addr );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT orig );

#endif /* !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN int NS(BeamBeam4D_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT source );

/* ************************************************************************* */
/* BeamBeam6D: */

typedef SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* NS(beambeam6d_real_ptr_t);
typedef SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const*
        NS(beambeam6d_real_const_ptr_t);

typedef struct NS(BeamBeam6D)
{
    SIXTRL_UINT64_T    data_size    SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t)  data_addr    SIXTRL_ALIGN( 8 );
}
NS(BeamBeam6D);

typedef struct{
    SIXTRL_REAL_T sphi      SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T cphi      SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T tphi      SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T salpha    SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T calpha    SIXTRL_ALIGN( 8 );
}NS(BB6D_boost_data);

typedef struct{
    SIXTRL_REAL_T Sig_11_0  SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T Sig_12_0  SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T Sig_13_0  SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T Sig_14_0  SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T Sig_22_0  SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T Sig_23_0  SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T Sig_24_0  SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T Sig_33_0  SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T Sig_34_0  SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T Sig_44_0  SIXTRL_ALIGN( 8 );
}NS(BB6D_Sigmas);

typedef struct{
    SIXTRL_REAL_T q_part             SIXTRL_ALIGN( 8 );
    NS(BB6D_boost_data) parboost     SIXTRL_ALIGN( 8 );
    NS(BB6D_Sigmas) Sigmas_0_star    SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T min_sigma_diff     SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T threshold_singular SIXTRL_ALIGN( 8 );
    SIXTRL_INT64_T N_slices          SIXTRL_ALIGN( 8 );

    SIXTRL_REAL_T delta_x            SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T delta_y            SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T x_CO               SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T px_CO              SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T y_CO               SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T py_CO              SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T sigma_CO           SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T delta_CO           SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T Dx_sub             SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T Dpx_sub            SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T Dy_sub             SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T Dpy_sub            SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T Dsigma_sub         SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T Ddelta_sub         SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T enabled            SIXTRL_ALIGN( 8 );

    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* N_part_per_slice   SIXTRL_ALIGN( 8 );
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* x_slices_star      SIXTRL_ALIGN( 8 );
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* y_slices_star      SIXTRL_ALIGN( 8 );
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* sigma_slices_star  SIXTRL_ALIGN( 8 );
}NS(BB6D_data);


SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t) NS(BeamBeam6D_type_id)(
    void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(BeamBeam6D_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const
        SIXTRL_RESTRICT elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(BeamBeam6D_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT elem,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT beam_beam );

SIXTRL_STATIC SIXTRL_FN void NS(BeamBeam6D_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT beam_beam );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t) NS(BeamBeam6D_data_addr)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const
        SIXTRL_RESTRICT bb_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(BeamBeam6D_set_data_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT bb_elem,
    NS(buffer_addr_t) const data_addr ) SIXTRL_NOEXCEPT;


SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(BB6D_data) const*
NS(BeamBeam6D_const_data)( SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const
    SIXTRL_RESTRICT bb_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(BB6D_data)*
NS(BeamBeam6D_data)( SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)*
    SIXTRL_RESTRICT bb_elem ) SIXTRL_NOEXCEPT;


SIXTRL_STATIC SIXTRL_FN NS(arch_size_t) NS(BeamBeam6D_data_size)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const
        SIXTRL_RESTRICT bb_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(BeamBeam6D_set_data_size)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT bb_elem,
    NS(arch_size_t) const data_size ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D) const*
NS(BeamBeam6D_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
    NS(Object) *const SIXTRL_RESTRICT obj_index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
    SIXTRL_RESTRICT obj_index ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D) const*
NS(BeamBeam6D_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D) const*
NS(BeamBeam6D_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(BeamBeam6D_data_addr_offset)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const
        SIXTRL_RESTRICT elem ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(object_type_id_t)
NS(BeamBeam6D_type_id_ext)( void ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BeamBeam6D_data_addr_offset_ext)( SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D)
    *const SIXTRL_RESTRICT bb_elem ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(BeamBeam6D_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_size,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_requ_dataptrs ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const data_size );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const  data_size,
    NS(buffer_addr_t) const bb4d_data_addr );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT orig );

#endif /* !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN int NS(BeamBeam6D_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT source );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN void NS(BeamBeam6D_boost)(
    SIXTRL_BE_DATAPTR_DEC NS(BB6D_boost_data)* data,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT x_star,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT px_star,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT y_star,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT py_star,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT sigma_star,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT delta_star );

SIXTRL_STATIC SIXTRL_FN void NS(BeamBeam6D_inv_boost)(
        SIXTRL_BE_DATAPTR_DEC NS(BB6D_boost_data)* SIXTRL_RESTRICT data,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT x,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT px,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT y,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT py,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT sigma,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT delta);

SIXTRL_STATIC SIXTRL_FN void NS(BeamBeam6D_propagate_Sigma_matrix)(
        SIXTRL_BE_DATAPTR_DEC NS(BB6D_Sigmas)* SIXTRL_RESTRICT data,
        SIXTRL_REAL_T S, SIXTRL_REAL_T threshold_singular,
        SIXTRL_UINT64_T handle_singularities,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT Sig_11_hat_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT Sig_33_hat_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT costheta_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT sintheta_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT dS_Sig_11_hat_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT dS_Sig_33_hat_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT dS_costheta_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT dS_sintheta_ptr);

/* ************************************************************************* */
/* SCCoasting: */

typedef struct NS(SCCoasting)
{
    SIXTRL_REAL_T   number_of_particles     SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   circumference           SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   sigma_x                 SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   sigma_y                 SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   length                  SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   x_co                    SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   y_co                    SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   min_sigma_diff          SIXTRL_ALIGN( 8 );
    SIXTRL_UINT64_T enabled                 SIXTRL_ALIGN( 8 );
}
NS(SCCoasting);

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)*
NS(SCCoasting_preset)( SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)*
    SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(SCCoasting_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)*
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t)
    NS(SCCoasting_type_id)( void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(SCCoasting_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(SCCoasting_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting) *const
        SIXTRL_RESTRICT sc_elem,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(SCCoasting_number_of_particles)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT belem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(SCCoasting_circumference)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT belem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(SCCoasting_sigma_x)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT belem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(SCCoasting_sigma_y)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT belem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(SCCoasting_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT belem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(SCCoasting_x_co)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT belem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(SCCoasting_y_co)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT belem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(SCCoasting_min_sigma_diff)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT belem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(SCCoasting_enabled)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT belem ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SCCoasting_set_number_of_particles)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)* SIXTRL_RESTRICT belem,
    SIXTRL_REAL_T const number_of_particles ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SCCoasting_set_circumference)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)* SIXTRL_RESTRICT belem,
    SIXTRL_REAL_T const circumference ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SCCoasting_set_sigma_x)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)* SIXTRL_RESTRICT belem,
    SIXTRL_REAL_T const sigma_x ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(SCCoasting_set_sigma_y)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)* SIXTRL_RESTRICT belem,
    SIXTRL_REAL_T const sigma_y ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(SCCoasting_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)* SIXTRL_RESTRICT belem,
    SIXTRL_REAL_T const length ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(SCCoasting_set_x_co)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)* SIXTRL_RESTRICT belem,
    SIXTRL_REAL_T const x_co ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(SCCoasting_set_y_co)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)* SIXTRL_RESTRICT belem,
    SIXTRL_REAL_T const y_co ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SCCoasting_set_min_sigma_diff)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)* SIXTRL_RESTRICT belem,
    SIXTRL_REAL_T const min_sigma_diff ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(SCCoasting_set_enabled)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)* SIXTRL_RESTRICT belem,
    bool const is_enabled ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(SCCoasting) const*
NS(SCCoasting_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC
    const NS(Object) *const SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)*
NS(SCCoasting_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC
    NS(Object)* SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(SCCoasting) const*
NS(SCCoasting_const_from_managed_buffer)( SIXTRL_BUFFER_DATAPTR_DEC
        unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)*
NS(SCCoasting_from_managed_buffer)( SIXTRL_BUFFER_DATAPTR_DEC
        unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(object_type_id_t)
NS(SCCoasting_type_id_ext)( void ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(SCCoasting) const*
NS(SCCoasting_const_from_buffer)( SIXTRL_BUFFER_ARGPTR_DEC
        const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)*
NS(SCCoasting_from_buffer)( SIXTRL_BUFFER_ARGPTR_DEC
        NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(SCCoasting_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(SCCoasting) *const
        SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(SCCoasting_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(SCCoasting_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(SCCoasting_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)*
NS(SCCoasting_new)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
    SIXTRL_RESTRICT buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)*
NS(SCCoasting_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_REAL_T const number_of_particles, SIXTRL_REAL_T const circumference,
    SIXTRL_REAL_T const sigma_x, SIXTRL_REAL_T const sigma_y,
    SIXTRL_REAL_T const length, SIXTRL_REAL_T const x_co,
    SIXTRL_REAL_T const y_co, SIXTRL_REAL_T const min_sigma_diff,
    bool const enabled );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)*
NS(SCCoasting_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(SCCoasting) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

#endif /* !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(SCCoasting_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)* SIXTRL_RESTRICT dest,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT;

/* ************************************************************************* */
/* NS(SCQGaussProfile): */

typedef struct NS(SCQGaussProfile)
{
    SIXTRL_REAL_T   number_of_particles   SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   bunchlength_rms       SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   sigma_x               SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   sigma_y               SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   length                SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   x_co                  SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   y_co                  SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   min_sigma_diff        SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   q_param               SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   cq                    SIXTRL_ALIGN( 8 );
    SIXTRL_UINT64_T enabled               SIXTRL_ALIGN( 8 );
}
NS(SCQGaussProfile);

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)*
NS(SCQGaussProfile_preset)( SIXTRL_BE_ARGPTR_DEC
    NS(SCQGaussProfile)* SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(SCQGaussProfile_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)*
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t)
NS(SCQGaussProfile_type_id)( void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(SCQGaussProfile_num_dataptrs)( SIXTRL_BE_ARGPTR_DEC const
    NS(SCQGaussProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(SCQGaussProfile_num_slots)( SIXTRL_BE_ARGPTR_DEC const
    NS(SCQGaussProfile) *const SIXTRL_RESTRICT sc_elem,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(SCQGaussProfile_number_of_particles)( SIXTRL_BE_ARGPTR_DEC const
    NS(SCQGaussProfile) *const
        SIXTRL_RESTRICT belem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(SCQGaussProfile_bunchlength_rms)( SIXTRL_BE_ARGPTR_DEC const
    NS(SCQGaussProfile) *const
        SIXTRL_RESTRICT belem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(SCQGaussProfile_sigma_x)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT belem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(SCQGaussProfile_sigma_y)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT belem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(SCQGaussProfile_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT belem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(SCQGaussProfile_x_co)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT belem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(SCQGaussProfile_y_co)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT belem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(SCQGaussProfile_min_sigma_diff)( SIXTRL_BE_ARGPTR_DEC const
    NS(SCQGaussProfile) *const
        SIXTRL_RESTRICT belem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(SCQGaussProfile_q_param)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile) *const
        SIXTRL_RESTRICT belem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(SCQGaussProfile_cq)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile) *const
        SIXTRL_RESTRICT belem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(SCQGaussProfile_enabled)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT belem ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SCQGaussProfile_set_number_of_particles)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)* SIXTRL_RESTRICT belem,
    SIXTRL_REAL_T const number_of_particles ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SCQGaussProfile_set_bunchlength_rms)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)* SIXTRL_RESTRICT belem,
    SIXTRL_REAL_T const circumference ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SCQGaussProfile_set_sigma_x)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)* SIXTRL_RESTRICT belem,
    SIXTRL_REAL_T const sigma_x ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SCQGaussProfile_set_sigma_y)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)* SIXTRL_RESTRICT belem,
    SIXTRL_REAL_T const sigma_y ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SCQGaussProfile_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)* SIXTRL_RESTRICT belem,
    SIXTRL_REAL_T const length ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SCQGaussProfile_set_x_co)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)* SIXTRL_RESTRICT belem,
    SIXTRL_REAL_T const x_co ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SCQGaussProfile_set_y_co)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)* SIXTRL_RESTRICT belem,
    SIXTRL_REAL_T const y_co ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SCQGaussProfile_set_min_sigma_diff)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)* SIXTRL_RESTRICT belem,
    SIXTRL_REAL_T const min_sigma_diff ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SCQGaussProfile_set_q_param)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)* SIXTRL_RESTRICT belem,
    SIXTRL_REAL_T const q_param ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SCQGaussProfile_set_enabled)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)* SIXTRL_RESTRICT belem,
    bool const is_enabled ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC
NS(SCQGaussProfile) const*
NS(SCQGaussProfile_const_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)*
NS(SCQGaussProfile_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC
    NS(Object)* SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC
    NS(SCQGaussProfile) const*
NS(SCQGaussProfile_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)*
NS(SCQGaussProfile_from_managed_buffer)( SIXTRL_BUFFER_DATAPTR_DEC
        unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(object_type_id_t)
NS(SCQGaussProfile_type_id_ext)( void ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC
NS(SCQGaussProfile) const*
NS(SCQGaussProfile_const_from_buffer)( SIXTRL_BUFFER_ARGPTR_DEC
        const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC
NS(SCQGaussProfile)* NS(SCQGaussProfile_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(SCQGaussProfile_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(SCQGaussProfile) *const
        SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(SCQGaussProfile_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(SCQGaussProfile_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(SCQGaussProfile_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC
NS(SCQGaussProfile)* NS(SCQGaussProfile_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC
NS(SCQGaussProfile)* NS(SCQGaussProfile_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_REAL_T const number_of_particles,
    SIXTRL_REAL_T const bunchlength_rms,
    SIXTRL_REAL_T const sigma_x, SIXTRL_REAL_T const sigma_y,
    SIXTRL_REAL_T const length,
    SIXTRL_REAL_T const x_co, SIXTRL_REAL_T const y_co,
    SIXTRL_REAL_T const min_sigma_diff,
    SIXTRL_REAL_T const q_param, bool const enabled );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC
NS(SCQGaussProfile)* NS(SCQGaussProfile_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(SCQGaussProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

#endif /* !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(SCQGaussProfile_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)* SIXTRL_RESTRICT dest,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT;

/* ************************************************************************* */
/* NS(SpaceChargeInterpolatedProfile): */

typedef struct NS(LineDensityProfileData)
{
    NS(math_interpol_int_t) method           SIXTRL_ALIGN( 8 );
    NS(math_abscissa_idx_t) capacity         SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t)       values_addr      SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t)       derivatives_addr SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T           z0               SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T           dz               SIXTRL_ALIGN( 8 );
    NS(math_abscissa_idx_t) num_values       SIXTRL_ALIGN( 8 );
}
NS(LineDensityProfileData);

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)*
NS(LineDensityProfileData_preset)( SIXTRL_BE_ARGPTR_DEC
    NS(LineDensityProfileData)* SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LineDensityProfileData_clear)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)*
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t)
    NS(LineDensityProfileData_type_id)( void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(LineDensityProfileData_num_dataptrs)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(LineDensityProfileData_num_slots)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(LineDensityProfileData) *const SIXTRL_RESTRICT sc_elem,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(math_interpol_t) NS(LineDensityProfileData_method)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(math_abscissa_idx_t)
NS(LineDensityProfileData_num_values)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(math_abscissa_idx_t)
NS(LineDensityProfileData_capacity)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t)
NS(LineDensityProfileData_values_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t)
NS(LineDensityProfileData_derivatives_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const*
NS(LineDensityProfileData_const_values_begin)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const*
NS(LineDensityProfileData_const_values_end)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T*
NS(LineDensityProfileData_values_begin)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(LineDensityProfileData)* SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T*
NS(LineDensityProfileData_values_end)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(LineDensityProfileData)* SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(LineDensityProfileData_value_at_idx)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data,
    NS(math_abscissa_idx_t) const idx ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const*
NS(LineDensityProfileData_const_derivatives_begin)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const*
NS(LineDensityProfileData_const_derivatives_end)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T*
NS(LineDensityProfileData_derivatives_begin)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(LineDensityProfileData)* SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T*
NS(LineDensityProfileData_derivatives_end)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(LineDensityProfileData)* SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(LineDensityProfileData_derivatives_at_idx)( SIXTRL_BUFFER_DATAPTR_DEC const
        NS(LineDensityProfileData) *const SIXTRL_RESTRICT data,
    NS(math_abscissa_idx_t) const idx ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(LineDensityProfileData_z0)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(LineDensityProfileData_dz)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
            SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(LineDensityProfileData_z_min)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
            SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(LineDensityProfileData_z_max)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
            SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(math_abscissa_idx_t)
NS(LineDensityProfileData_find_idx)( SIXTRL_BUFFER_DATAPTR_DEC const
        NS(LineDensityProfileData) *const SIXTRL_RESTRICT data,
    SIXTRL_REAL_T const z ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(LineDensityProfileData_interpolate_value)( SIXTRL_BUFFER_DATAPTR_DEC const
        NS(LineDensityProfileData) *const SIXTRL_RESTRICT data,
    SIXTRL_REAL_T const z ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(LineDensityProfileData_interpolate_1st_derivative)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data, SIXTRL_REAL_T const z ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(LineDensityProfileData_interpolate_2nd_derivative)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data, SIXTRL_REAL_T const z ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(LineDensityProfileData_prepare_interpolation)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT data,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T*
        SIXTRL_RESTRICT temp_data ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(LineDensityProfileData_set_num_values)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT data,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(LineDensityProfileData_set_capacity)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT data,
    NS(math_abscissa_idx_t) const capacity ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(LineDensityProfileData_set_z0)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT data,
    SIXTRL_REAL_T const z0 ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(LineDensityProfileData_set_dz)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT data,
    SIXTRL_REAL_T const z0 ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(LineDensityProfileData_set_values_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT data,
    NS(buffer_addr_t) const values_addr ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(LineDensityProfileData_set_derivatives_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT data,
    NS(buffer_addr_t) const derivatives_addr ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(LineDensityProfileData_set_method)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT data,
    NS(math_interpol_t) const method ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(LineDensityProfileData_values_offset)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t)
NS(LineDensityProfileData_derivatives_offset)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(LineDensityProfileData_values_offset_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(LineDensityProfileData)* SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(LineDensityProfileData_derivatives_offset_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(LineDensityProfileData)* SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(LineDensityProfileData_prepare_interpolation_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(LineDensityProfileData)* SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(object_type_id_t)
NS(LineDensityProfileData_type_id_ext)( void ) SIXTRL_NOEXCEPT;


SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_REAL_T NS(LineDensityProfileData_z0_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_REAL_T NS(LineDensityProfileData_dz_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
            SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_REAL_T NS(LineDensityProfileData_z_min_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
            SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_REAL_T NS(LineDensityProfileData_z_max_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
            SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(math_abscissa_idx_t)
NS(LineDensityProfileData_find_idx_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
        NS(LineDensityProfileData) *const SIXTRL_RESTRICT data,
    SIXTRL_REAL_T const z ) SIXTRL_NOEXCEPT;


SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_REAL_T
NS(LineDensityProfileData_interpolate_value_ext)( SIXTRL_BUFFER_DATAPTR_DEC
        const NS(LineDensityProfileData) *const SIXTRL_RESTRICT data,
    SIXTRL_REAL_T const z ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_REAL_T
NS(LineDensityProfileData_interpolate_1st_derivative_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data, SIXTRL_REAL_T const z ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_REAL_T
NS(LineDensityProfileData_interpolate_2nd_derivative_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data, SIXTRL_REAL_T const z ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(LineDensityProfileData_set_z0_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT data,
    SIXTRL_REAL_T const z0 ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(LineDensityProfileData_set_dz_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT data,
    SIXTRL_REAL_T const z0 ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(LineDensityProfileData_set_values_addr_ext)( SIXTRL_BUFFER_DATAPTR_DEC
        NS(LineDensityProfileData)* SIXTRL_RESTRICT data,
    NS(buffer_addr_t) const values_addr ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(LineDensityProfileData_set_derivatives_addr_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(LineDensityProfileData)* SIXTRL_RESTRICT data, NS(buffer_addr_t) const
        derivatives_addr ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(LineDensityProfileData_set_method_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT data,
    NS(math_interpol_t) const method ) SIXTRL_NOEXCEPT;

#endif /* Host */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LineDensityProfileData) const*
NS(LineDensityProfileData_const_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)*
NS(LineDensityProfileData_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC
    NS(Object)* SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LineDensityProfileData) const*
NS(LineDensityProfileData_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)*
NS(LineDensityProfileData_from_managed_buffer)( SIXTRL_BUFFER_DATAPTR_DEC
        unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC
NS(LineDensityProfileData) const* NS(LineDensityProfileData_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(LineDensityProfileData)*
NS(LineDensityProfileData_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(LineDensityProfileData_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(LineDensityProfileData_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData)
        *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(LineDensityProfileData_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData)
        *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(LineDensityProfileData_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(math_abscissa_idx_t) const capacity,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC
NS(LineDensityProfileData)* NS(LineDensityProfileData_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(math_abscissa_idx_t) const capacity );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC
NS(LineDensityProfileData)* NS(LineDensityProfileData_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(math_interpol_t) const method, NS(math_abscissa_idx_t) num_values,
    NS(buffer_addr_t) const values_addr,
    NS(buffer_addr_t) const derivatives_addr,
    SIXTRL_REAL_T const z0, SIXTRL_REAL_T const dz,
    NS(math_abscissa_idx_t) capacity );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC
NS(LineDensityProfileData)* NS(LineDensityProfileData_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

#endif /* !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LineDensityProfileData_copy)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT dest,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData)
        *const SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT;

/* ************************************************************************* */

typedef struct NS(SpaceChargeInterpolatedProfile)
{
    SIXTRL_REAL_T     number_of_particles        SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T     sigma_x                    SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T     sigma_y                    SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T     length                     SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T     x_co                       SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T     y_co                       SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t) interpol_data_addr         SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T     line_density_prof_fallback SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T     min_sigma_diff             SIXTRL_ALIGN( 8 );
    SIXTRL_UINT64_T   enabled                    SIXTRL_ALIGN( 8 );
}
NS(SpaceChargeInterpolatedProfile);

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
NS(SpaceChargeInterpolatedProfile_preset)( SIXTRL_BE_ARGPTR_DEC
    NS(SpaceChargeInterpolatedProfile)* SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SpaceChargeInterpolatedProfile_clear)( SIXTRL_BE_ARGPTR_DEC
    NS(SpaceChargeInterpolatedProfile)* SIXTRL_RESTRICT
        sc_elem ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t)
NS(SpaceChargeInterpolatedProfile_type_id)( void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(SpaceChargeInterpolatedProfile_num_dataptrs)( SIXTRL_BE_ARGPTR_DEC const
    NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(SpaceChargeInterpolatedProfile_num_slots)( SIXTRL_BE_ARGPTR_DEC const
    NS(SpaceChargeInterpolatedProfile) *const SIXTRL_RESTRICT sc_elem,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(SpaceChargeInterpolatedProfile_interpol_data_addr_offset)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN bool
NS(SpaceChargeInterpolatedProfile_has_interpol_data)( SIXTRL_BE_ARGPTR_DEC
    const NS(SpaceChargeInterpolatedProfile) *const SIXTRL_RESTRICT
        sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t)
NS(SpaceChargeInterpolatedProfile_interpol_data_addr)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(LineDensityProfileData) const*
NS(SpaceChargeInterpolatedProfile_const_line_density_profile_data)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(LineDensityProfileData)*
NS(SpaceChargeInterpolatedProfile_line_density_profile_data)( SIXTRL_BE_ARGPTR_DEC
    NS(SpaceChargeInterpolatedProfile)* SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(SpaceChargeInterpolatedProfile_number_of_particles)( SIXTRL_BE_ARGPTR_DEC const
    NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(SpaceChargeInterpolatedProfile_line_density_prof_fallback)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(SpaceChargeInterpolatedProfile_sigma_x)( SIXTRL_BE_ARGPTR_DEC const
    NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(SpaceChargeInterpolatedProfile_sigma_y)( SIXTRL_BE_ARGPTR_DEC const
    NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(SpaceChargeInterpolatedProfile_length)( SIXTRL_BE_ARGPTR_DEC const
    NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(SpaceChargeInterpolatedProfile_x_co)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(SpaceChargeInterpolatedProfile_y_co)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(SpaceChargeInterpolatedProfile_min_sigma_diff)( SIXTRL_BE_ARGPTR_DEC const
    NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(SpaceChargeInterpolatedProfile_enabled)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SpaceChargeInterpolatedProfile_set_interpol_data_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem,
    NS(buffer_addr_t) const interpol_data_addr ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SpaceChargeInterpolatedProfile_set_number_of_particles)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const number_of_particles ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SpaceChargeInterpolatedProfile_set_line_density_prof_fallback)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const line_density_prof_fallback ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SpaceChargeInterpolatedProfile_set_sigma_x)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem, SIXTRL_REAL_T const sigma_x ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SpaceChargeInterpolatedProfile_set_sigma_y)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem, SIXTRL_REAL_T const sigma_y ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SpaceChargeInterpolatedProfile_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem, SIXTRL_REAL_T const length ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SpaceChargeInterpolatedProfile_set_x_co)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem, SIXTRL_REAL_T const x_co ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SpaceChargeInterpolatedProfile_set_y_co)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem, SIXTRL_REAL_T const y_co ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SpaceChargeInterpolatedProfile_set_min_sigma_diff)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const min_sigma_diff ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SpaceChargeInterpolatedProfile_set_enabled)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem, bool const is_enabled ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(SpaceChargeInterpolatedProfile_line_density_profile)( SIXTRL_BE_ARGPTR_DEC
    NS(SpaceChargeInterpolatedProfile)* SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const z ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(SpaceChargeInterpolatedProfile_line_density_profile_1st_derivative)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem, SIXTRL_REAL_T const z ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(SpaceChargeInterpolatedProfile_line_density_profile_2nd_derivative)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem, SIXTRL_REAL_T const z ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SpaceChargeInterpolatedProfile_prepare_interpolation)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem, SIXTRL_ARGPTR_DEC SIXTRL_REAL_T*
            SIXTRL_RESTRICT temp_data ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN
SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile) const*
NS(SpaceChargeInterpolatedProfile_const_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const
        SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN
SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
NS(SpaceChargeInterpolatedProfile_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
        SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT;


SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC
NS(SpaceChargeInterpolatedProfile) const*
NS(SpaceChargeInterpolatedProfile_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
NS(SpaceChargeInterpolatedProfile_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(object_type_id_t)
    NS(SpaceChargeInterpolatedProfile_type_id_ext)( void ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(SpaceChargeInterpolatedProfile_interpol_data_addr_offset_ext)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC
NS(SpaceChargeInterpolatedProfile) const*
NS(SpaceChargeInterpolatedProfile_const_from_buffer)( SIXTRL_BUFFER_ARGPTR_DEC
        const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC
NS(SpaceChargeInterpolatedProfile)* NS(SpaceChargeInterpolatedProfile_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(SpaceChargeInterpolatedProfile_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(SpaceChargeInterpolatedProfile_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(SpaceChargeInterpolatedProfile_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile)
        *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(SpaceChargeInterpolatedProfile_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC
NS(SpaceChargeInterpolatedProfile)* NS(SpaceChargeInterpolatedProfile_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC
NS(SpaceChargeInterpolatedProfile)* NS(SpaceChargeInterpolatedProfile_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_REAL_T const number_of_particles,
    SIXTRL_REAL_T const sigma_x, SIXTRL_REAL_T const sigma_y,
    SIXTRL_REAL_T const length, SIXTRL_REAL_T const x_co,
    SIXTRL_REAL_T const y_co,
    SIXTRL_REAL_T const line_density_prof_fallback_value,
    NS(buffer_addr_t) const interpol_data_addr,
    SIXTRL_REAL_T const min_sigma_diff, bool const enabled );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC
NS(SpaceChargeInterpolatedProfile)* NS(SpaceChargeInterpolatedProfile_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT;

#endif /* !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(SpaceChargeInterpolatedProfile_copy)( SIXTRL_BE_ARGPTR_DEC
        NS(SpaceChargeInterpolatedProfile)* SIXTRL_RESTRICT dest,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile)
        *const SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT;

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====             Implementation of inline functions                ===== */
/* ========================================================================= */

#if !defined( SIXTRL_NO_INCLUDES )
    #if !defined( _GPUCODE ) || defined( __CUDACC__ )
    #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) || defined( __CUDACC__ ) */

    #include "sixtracklib/common/internal/math_qgauss.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* BeamBeam4D: */

SIXTRL_INLINE NS(object_type_id_t)
    NS(BeamBeam4D_type_id)( void ) SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(OBJECT_TYPE_BEAM_BEAM_4D);
}

SIXTRL_INLINE NS(buffer_size_t) NS(BeamBeam4D_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( elem ) ) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )1u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(BeamBeam4D_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT elem,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) st_size_t;
    st_size_t num_slots = ( st_size_t )0u;

    if( ( elem != SIXTRL_NULLPTR ) && ( slot_size > ( st_size_t )0 ) )
    {
        st_size_t num_bytes = NS(ManagedBuffer_get_slot_based_length)(
            sizeof( NS(BeamBeam4D) ), slot_size );

        st_size_t const stored_data_size =
            NS(BeamBeam4D_data_size)( elem ) * sizeof( SIXTRL_REAL_T );

        SIXTRL_ASSERT( stored_data_size >=
            NS(ManagedBuffer_get_slot_based_length)( sizeof(
                NS(BB4D_data) ), slot_size ) );

        SIXTRL_ASSERT( ( stored_data_size % slot_size ) == ( st_size_t )0 );
        num_bytes += stored_data_size;
        num_slots  = num_bytes / slot_size;

        if( num_slots * slot_size < num_bytes ) ++num_slots;
    }

    return num_slots;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* NS(BeamBeam4D_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT bb_elem )
{
    if( bb_elem != SIXTRL_NULLPTR ) NS(BeamBeam4D_clear)( bb_elem );
    return bb_elem;
}

SIXTRL_INLINE void NS(BeamBeam4D_clear)( SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)*
    SIXTRL_RESTRICT bb_elem )
{
    SIXTRL_ASSERT( bb_elem != SIXTRL_NULLPTR );
    NS(arch_status_t) status = NS(BeamBeam4D_set_data_addr)(
        bb_elem, ( NS(buffer_addr_t) )0 );

    status |= NS(BeamBeam4D_set_data_size)(
        bb_elem, ( NS(buffer_size_t) )0 );

    SIXTRL_ASSERT( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_addr_t) NS(BeamBeam4D_data_addr)( SIXTRL_BE_ARGPTR_DEC
    const NS(BeamBeam4D) *const SIXTRL_RESTRICT bb_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( bb_elem != SIXTRL_NULLPTR );
    return bb_elem->data_addr;
}

SIXTRL_INLINE NS(arch_status_t) NS(BeamBeam4D_set_data_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT bb_elem,
    NS(buffer_addr_t) const data_addr ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( bb_elem != SIXTRL_NULLPTR );
    bb_elem->data_addr = data_addr;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(BB4D_data) const*
NS(BeamBeam4D_const_data)( SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const
    SIXTRL_RESTRICT bb_elem ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_DATAPTR_DEC NS(BB4D_data) const* )( uintptr_t
        )NS(BeamBeam4D_data_addr)( bb_elem );
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(BB4D_data)* NS(BeamBeam4D_data)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)*
        SIXTRL_RESTRICT bb_elem ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_DATAPTR_DEC NS(BB4D_data)* )( uintptr_t
        )NS(BeamBeam4D_data_addr)( bb_elem );
}

SIXTRL_INLINE NS(arch_size_t) NS(BeamBeam4D_data_size)( SIXTRL_BE_ARGPTR_DEC
    const NS(BeamBeam4D) *const SIXTRL_RESTRICT bb_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( bb_elem != SIXTRL_NULLPTR );
    return bb_elem->data_size;
}

SIXTRL_INLINE NS(arch_status_t) NS(BeamBeam4D_set_data_size)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT bb_elem,
    NS(arch_size_t) const data_size ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( bb_elem != SIXTRL_NULLPTR );
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    NS(buffer_size_t) min_bb4_data_size =
        NS(ManagedBuffer_get_slot_based_length)( sizeof( NS(BB4D_data) ),
            ( NS(buffer_size_t) )8u );
    min_bb4_data_size /= sizeof( SIXTRL_REAL_T );


    if( ( bb_elem != SIXTRL_NULLPTR ) &&
        ( ( data_size == ( NS(buffer_size_t) )0 ) ||
          ( data_size >= min_bb4_data_size ) ) )
    {
        bb_elem->data_size = data_size;
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D) const*
NS(BeamBeam4D_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
    NS(Object) *const SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
{
     return (
        ( NS(Object_get_type_id)( obj ) == NS(OBJECT_TYPE_BEAM_BEAM_4D) ) &&
        ( NS(Object_get_size)( obj ) >= sizeof( NS(BeamBeam4D) ) ) )
        ? ( SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D) const* )( uintptr_t
            )NS(Object_get_begin_addr)( obj )
        : SIXTRL_NULLPTR;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
    SIXTRL_RESTRICT obj_index ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)*
        )NS(BeamBeam4D_const_from_obj_index)( obj_index );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D) const*
NS(BeamBeam4D_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(BeamBeam4D_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(BeamBeam4D_from_obj_index)(
        NS(ManagedBuffer_get_object)( buffer_begin, index, slot_size ) );
}

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D) const*
NS(BeamBeam4D_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(BeamBeam4D_const_from_obj_index)(
        NS(Buffer_get_const_object)( buffer, index ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* NS(BeamBeam4D_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(BeamBeam4D_from_obj_index)(
        NS(Buffer_get_object)( buffer, index ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(BeamBeam4D_data_addr_offset)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( elem ) ) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )offsetof( NS(BeamBeam4D), data_addr );
}

#endif /* _GPUCODE */

SIXTRL_INLINE NS(arch_status_t) NS(BeamBeam4D_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT source )
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( destination != SIXTRL_NULLPTR ) && ( source != SIXTRL_NULLPTR ) )
    {
        if( destination != source )
        {
            SIXTRL_BE_DATAPTR_DEC NS(BB4D_data)* dst_data =
                NS(BeamBeam4D_data)( destination );

            SIXTRL_BE_DATAPTR_DEC NS(BB4D_data) const* src_data =
                NS(BeamBeam4D_const_data)( source );

            if( ( dst_data != SIXTRL_NULLPTR ) &&
                ( src_data != SIXTRL_NULLPTR ) &&
                ( NS(BeamBeam4D_data_size)( destination ) >=
                  NS(BeamBeam4D_data_size)( source ) ) &&
                ( dst_data != src_data ) )
            {
                *dst_data = *src_data;
                status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
            }

            if( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS )
            {
                status = NS(BeamBeam4D_set_data_size)( destination,
                    NS(BeamBeam4D_data_size)( source ) );
            }
        }
        else
        {
            status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
        }
    }

    return status;
}

/* ************************************************************************* */
/* BeamBeam6D: */

SIXTRL_INLINE NS(object_type_id_t)
    NS(BeamBeam6D_type_id)( void ) SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(OBJECT_TYPE_BEAM_BEAM_4D);
}

SIXTRL_INLINE NS(buffer_size_t) NS(BeamBeam6D_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( elem ) ) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )1u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(BeamBeam6D_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT elem,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) st_size_t;
    st_size_t num_slots = ( st_size_t )0u;

    if( ( elem != SIXTRL_NULLPTR ) && ( slot_size > ( st_size_t )0 ) )
    {
        st_size_t num_bytes = NS(ManagedBuffer_get_slot_based_length)(
            sizeof( NS(BeamBeam6D) ), slot_size );

        st_size_t const stored_data_size =
            NS(BeamBeam6D_data_size)( elem ) * sizeof( SIXTRL_REAL_T );

        SIXTRL_ASSERT( stored_data_size >=
            NS(ManagedBuffer_get_slot_based_length)( sizeof(
                NS(BB6D_data) ), slot_size ) );

        SIXTRL_ASSERT( ( stored_data_size % slot_size ) == ( st_size_t )0 );
        num_bytes += stored_data_size;
        num_slots  = num_bytes / slot_size;

        if( num_slots * slot_size < num_bytes ) ++num_slots;
    }

    return num_slots;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* NS(BeamBeam6D_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT bb_elem )
{
    if( bb_elem != SIXTRL_NULLPTR ) NS(BeamBeam6D_clear)( bb_elem );
    return bb_elem;
}

SIXTRL_INLINE void NS(BeamBeam6D_clear)( SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)*
    SIXTRL_RESTRICT bb_elem )
{
    SIXTRL_ASSERT( bb_elem != SIXTRL_NULLPTR );
    NS(arch_status_t) status = NS(BeamBeam6D_set_data_addr)(
        bb_elem, ( NS(buffer_addr_t) )0 );

    status |= NS(BeamBeam6D_set_data_size)(
        bb_elem, ( NS(buffer_size_t) )0 );

    SIXTRL_ASSERT( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_addr_t) NS(BeamBeam6D_data_addr)( SIXTRL_BE_ARGPTR_DEC
    const NS(BeamBeam6D) *const SIXTRL_RESTRICT bb_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( bb_elem != SIXTRL_NULLPTR );
    return bb_elem->data_addr;
}

SIXTRL_INLINE NS(arch_status_t) NS(BeamBeam6D_set_data_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT bb_elem,
    NS(buffer_addr_t) const data_addr ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( bb_elem != SIXTRL_NULLPTR );
    bb_elem->data_addr = data_addr;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(BB6D_data) const*
NS(BeamBeam6D_const_data)( SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const
    SIXTRL_RESTRICT bb_elem ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_DATAPTR_DEC NS(BB6D_data) const* )( uintptr_t
        )NS(BeamBeam6D_data_addr)( bb_elem );
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(BB6D_data)* NS(BeamBeam6D_data)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)*
        SIXTRL_RESTRICT bb_elem ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_DATAPTR_DEC NS(BB6D_data)* )( uintptr_t
        )NS(BeamBeam6D_data_addr)( bb_elem );
}

SIXTRL_INLINE NS(arch_size_t) NS(BeamBeam6D_data_size)( SIXTRL_BE_ARGPTR_DEC
    const NS(BeamBeam6D) *const SIXTRL_RESTRICT bb_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( bb_elem != SIXTRL_NULLPTR );
    return bb_elem->data_size;
}

SIXTRL_INLINE NS(arch_status_t) NS(BeamBeam6D_set_data_size)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT bb_elem,
    NS(arch_size_t) const data_size ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( bb_elem != SIXTRL_NULLPTR );
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    NS(buffer_size_t) min_bb4_data_size =
        NS(ManagedBuffer_get_slot_based_length)( sizeof( NS(BB6D_data) ),
            ( NS(buffer_size_t) )8u );
    min_bb4_data_size /= sizeof( SIXTRL_REAL_T );


    if( ( bb_elem != SIXTRL_NULLPTR ) &&
        ( ( data_size == ( NS(buffer_size_t) )0 ) ||
          ( data_size >= min_bb4_data_size ) ) )
    {
        bb_elem->data_size = data_size;
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D) const*
NS(BeamBeam6D_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
    NS(Object) *const SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
{
     return (
        ( NS(Object_get_type_id)( obj ) == NS(OBJECT_TYPE_BEAM_BEAM_6D) ) &&
        ( NS(Object_get_size)( obj ) >= sizeof( NS(BeamBeam6D) ) ) )
        ? ( SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D) const* )( uintptr_t
            )NS(Object_get_begin_addr)( obj )
        : SIXTRL_NULLPTR;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
    SIXTRL_RESTRICT obj_index ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)*
        )NS(BeamBeam6D_const_from_obj_index)( obj_index );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D) const*
NS(BeamBeam6D_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(BeamBeam6D_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(BeamBeam6D_from_obj_index)(
        NS(ManagedBuffer_get_object)( buffer_begin, index, slot_size ) );
}

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D) const*
NS(BeamBeam6D_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(BeamBeam6D_const_from_obj_index)(
        NS(Buffer_get_const_object)( buffer, index ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* NS(BeamBeam6D_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(BeamBeam6D_from_obj_index)(
        NS(Buffer_get_object)( buffer, index ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(BeamBeam6D_data_addr_offset)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( elem ) ) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )offsetof( NS(BeamBeam6D), data_addr );
}

#endif /* !defined( _GPUCODE ) */

SIXTRL_INLINE NS(arch_status_t) NS(BeamBeam6D_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT source )
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( destination != SIXTRL_NULLPTR ) && ( source != SIXTRL_NULLPTR ) )
    {
        if( destination != source )
        {
            SIXTRL_BE_DATAPTR_DEC NS(BB6D_data)* dst_data =
                NS(BeamBeam6D_data)( destination );

            SIXTRL_BE_DATAPTR_DEC NS(BB6D_data) const* src_data =
                NS(BeamBeam6D_const_data)( source );

            if( ( dst_data != SIXTRL_NULLPTR ) &&
                ( src_data != SIXTRL_NULLPTR ) &&
                ( NS(BeamBeam6D_data_size)( destination ) >=
                  NS(BeamBeam6D_data_size)( source ) ) &&
                ( dst_data != src_data ) )
            {
                *dst_data = *src_data;
                status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
            }

            if( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS )
            {
                status = NS(BeamBeam6D_set_data_size)( destination,
                    NS(BeamBeam6D_data_size)( source ) );
            }
        }
        else
        {
            status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
        }
    }

    return status;
}

/* ------------------------------------------------------------------------- */

#if !defined(mysign)
	#define mysign(a) (((a) >= 0) - ((a) < 0))
#endif

SIXTRL_INLINE void NS(BeamBeam6D_boost)(
    SIXTRL_BE_DATAPTR_DEC NS(BB6D_boost_data)* SIXTRL_RESTRICT data,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT x_star,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT px_star,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT y_star,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT py_star,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT sigma_star,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT delta_star)
{
    SIXTRL_REAL_T const sphi = data->sphi;
    SIXTRL_REAL_T const cphi = data->cphi;
    SIXTRL_REAL_T const tphi = data->tphi;
    SIXTRL_REAL_T const salpha = data->salpha;
    SIXTRL_REAL_T const calpha = data->calpha;


    SIXTRL_REAL_T const x = *x_star;
    SIXTRL_REAL_T const px = *px_star;
    SIXTRL_REAL_T const y = *y_star;
    SIXTRL_REAL_T const py = *py_star ;
    SIXTRL_REAL_T const sigma = *sigma_star;
    SIXTRL_REAL_T const delta = *delta_star ;

    SIXTRL_REAL_T const h = delta + 1. - sqrt((1.+delta)*(1.+delta)-px*px-py*py);


    SIXTRL_REAL_T const px_st = px/cphi-h*calpha*tphi/cphi;
    SIXTRL_REAL_T const py_st = py/cphi-h*salpha*tphi/cphi;
    SIXTRL_REAL_T const delta_st = delta -px*calpha*tphi-py*salpha*tphi+h*tphi*tphi;

    SIXTRL_REAL_T const pz_st =
        sqrt((1.+delta_st)*(1.+delta_st)-px_st*px_st-py_st*py_st);

    SIXTRL_REAL_T const hx_st = px_st/pz_st;
    SIXTRL_REAL_T const hy_st = py_st/pz_st;
    SIXTRL_REAL_T const hsigma_st = 1.-(delta_st+1)/pz_st;

    SIXTRL_REAL_T const L11 = 1.+hx_st*calpha*sphi;
    SIXTRL_REAL_T const L12 = hx_st*salpha*sphi;
    SIXTRL_REAL_T const L13 = calpha*tphi;

    SIXTRL_REAL_T const L21 = hy_st*calpha*sphi;
    SIXTRL_REAL_T const L22 = 1.+hy_st*salpha*sphi;
    SIXTRL_REAL_T const L23 = salpha*tphi;

    SIXTRL_REAL_T const L31 = hsigma_st*calpha*sphi;
    SIXTRL_REAL_T const L32 = hsigma_st*salpha*sphi;
    SIXTRL_REAL_T const L33 = 1./cphi;

    SIXTRL_REAL_T const x_st = L11*x + L12*y + L13*sigma;
    SIXTRL_REAL_T const y_st = L21*x + L22*y + L23*sigma;
    SIXTRL_REAL_T const sigma_st = L31*x + L32*y + L33*sigma;

    *x_star = x_st;
    *px_star = px_st;
    *y_star = y_st;
    *py_star = py_st;
    *sigma_star = sigma_st;
    *delta_star = delta_st;

}

SIXTRL_INLINE void NS(BeamBeam6D_inv_boost)(
        SIXTRL_BE_DATAPTR_DEC NS(BB6D_boost_data)* SIXTRL_RESTRICT data,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT x,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT px,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT y,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT py,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT sigma,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT delta )
{

    SIXTRL_REAL_T const sphi = data->sphi;
    SIXTRL_REAL_T const cphi = data->cphi;
    SIXTRL_REAL_T const tphi = data->tphi;
    SIXTRL_REAL_T const salpha = data->salpha;
    SIXTRL_REAL_T const calpha = data->calpha;

    SIXTRL_REAL_T const x_st = *x;
    SIXTRL_REAL_T const px_st = *px;
    SIXTRL_REAL_T const y_st = *y;
    SIXTRL_REAL_T const py_st = *py ;
    SIXTRL_REAL_T const sigma_st = *sigma;
    SIXTRL_REAL_T const delta_st = *delta ;

    SIXTRL_REAL_T const pz_st = sqrt((1.+delta_st)*(1.+delta_st)-px_st*px_st-py_st*py_st);
    SIXTRL_REAL_T const hx_st = px_st/pz_st;
    SIXTRL_REAL_T const hy_st = py_st/pz_st;
    SIXTRL_REAL_T const hsigma_st = 1.-(delta_st+1)/pz_st;

    SIXTRL_REAL_T const Det_L =
        1./cphi + (hx_st*calpha + hy_st*salpha-hsigma_st*sphi)*tphi;

    SIXTRL_REAL_T const Linv_11 =
        (1./cphi + salpha*tphi*(hy_st-hsigma_st*salpha*sphi))/Det_L;

    SIXTRL_REAL_T const Linv_12 =
        (salpha*tphi*(hsigma_st*calpha*sphi-hx_st))/Det_L;

    SIXTRL_REAL_T const Linv_13 =
        -tphi*(calpha - hx_st*salpha*salpha*sphi + hy_st*calpha*salpha*sphi)/Det_L;

    SIXTRL_REAL_T const Linv_21 =
        (calpha*tphi*(-hy_st + hsigma_st*salpha*sphi))/Det_L;

    SIXTRL_REAL_T const Linv_22 =
        (1./cphi + calpha*tphi*(hx_st-hsigma_st*calpha*sphi))/Det_L;

    SIXTRL_REAL_T const Linv_23 =
        -tphi*(salpha - hy_st*calpha*calpha*sphi + hx_st*calpha*salpha*sphi)/Det_L;

    SIXTRL_REAL_T const Linv_31 = -hsigma_st*calpha*sphi/Det_L;
    SIXTRL_REAL_T const Linv_32 = -hsigma_st*salpha*sphi/Det_L;
    SIXTRL_REAL_T const Linv_33 = (1. + hx_st*calpha*sphi + hy_st*salpha*sphi)/Det_L;

    SIXTRL_REAL_T const x_i = Linv_11*x_st + Linv_12*y_st + Linv_13*sigma_st;
    SIXTRL_REAL_T const y_i = Linv_21*x_st + Linv_22*y_st + Linv_23*sigma_st;
    SIXTRL_REAL_T const sigma_i = Linv_31*x_st + Linv_32*y_st + Linv_33*sigma_st;

    SIXTRL_REAL_T const h = (delta_st+1.-pz_st)*cphi*cphi;

    SIXTRL_REAL_T const px_i = px_st*cphi+h*calpha*tphi;
    SIXTRL_REAL_T const py_i = py_st*cphi+h*salpha*tphi;

    SIXTRL_REAL_T const delta_i = delta_st + px_i*calpha*tphi + py_i*salpha*tphi - h*tphi*tphi;


    *x = x_i;
    *px = px_i;
    *y = y_i;
    *py = py_i;
    *sigma = sigma_i;
    *delta = delta_i;

}

SIXTRL_INLINE void NS(BeamBeam6D_propagate_Sigma_matrix)(
        SIXTRL_BE_DATAPTR_DEC NS(BB6D_Sigmas)* SIXTRL_RESTRICT data,
        SIXTRL_REAL_T const S, SIXTRL_REAL_T const threshold_singular,
        SIXTRL_UINT64_T const handle_singularities,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT Sig_11_hat_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT Sig_33_hat_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT costheta_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT sintheta_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT dS_Sig_11_hat_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT dS_Sig_33_hat_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT dS_costheta_ptr,
        SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT dS_sintheta_ptr)
{
    SIXTRL_REAL_T const Sig_11_0 = data->Sig_11_0;
    SIXTRL_REAL_T const Sig_12_0 = data->Sig_12_0;
    SIXTRL_REAL_T const Sig_13_0 = data->Sig_13_0;
    SIXTRL_REAL_T const Sig_14_0 = data->Sig_14_0;
    SIXTRL_REAL_T const Sig_22_0 = data->Sig_22_0;
    SIXTRL_REAL_T const Sig_23_0 = data->Sig_23_0;
    SIXTRL_REAL_T const Sig_24_0 = data->Sig_24_0;
    SIXTRL_REAL_T const Sig_33_0 = data->Sig_33_0;
    SIXTRL_REAL_T const Sig_34_0 = data->Sig_34_0;
    SIXTRL_REAL_T const Sig_44_0 = data->Sig_44_0;

    // Propagate sigma matrix
    SIXTRL_REAL_T const Sig_11 = Sig_11_0 + 2.*Sig_12_0*S+Sig_22_0*S*S;
    SIXTRL_REAL_T const Sig_33 = Sig_33_0 + 2.*Sig_34_0*S+Sig_44_0*S*S;
    SIXTRL_REAL_T const Sig_13 = Sig_13_0 + (Sig_14_0+Sig_23_0)*S+Sig_24_0*S*S;

    SIXTRL_REAL_T const Sig_12 = Sig_12_0 + Sig_22_0*S;
    SIXTRL_REAL_T const Sig_14 = Sig_14_0 + Sig_24_0*S;
    SIXTRL_REAL_T const Sig_22 = Sig_22_0 + 0.*S;
    SIXTRL_REAL_T const Sig_23 = Sig_23_0 + Sig_24_0*S;
    SIXTRL_REAL_T const Sig_24 = Sig_24_0 + 0.*S;
    SIXTRL_REAL_T const Sig_34 = Sig_34_0 + Sig_44_0*S;
    SIXTRL_REAL_T const Sig_44 = Sig_44_0 + 0.*S;

    SIXTRL_REAL_T const R = Sig_11-Sig_33;
    SIXTRL_REAL_T const W = Sig_11+Sig_33;
    SIXTRL_REAL_T const T = R*R+4*Sig_13*Sig_13;

    //evaluate derivatives
    SIXTRL_REAL_T const dS_R = 2.*(Sig_12_0-Sig_34_0)+2*S*(Sig_22_0-Sig_44_0);
    SIXTRL_REAL_T const dS_W = 2.*(Sig_12_0+Sig_34_0)+2*S*(Sig_22_0+Sig_44_0);
    SIXTRL_REAL_T const dS_Sig_13 = Sig_14_0 + Sig_23_0 + 2*Sig_24_0*S;
    SIXTRL_REAL_T const dS_T = 2*R*dS_R+8.*Sig_13*dS_Sig_13;

    SIXTRL_REAL_T Sig_11_hat, Sig_33_hat, costheta, sintheta, dS_Sig_11_hat,
           dS_Sig_33_hat, dS_costheta, dS_sintheta, cos2theta, dS_cos2theta;

    SIXTRL_REAL_T const signR = mysign(R);

    //~ printf("handle: %ld\n",handle_singularities);

    if (T<threshold_singular && handle_singularities){

        SIXTRL_REAL_T const a = Sig_12-Sig_34;
        SIXTRL_REAL_T const b = Sig_22-Sig_44;
        SIXTRL_REAL_T const c = Sig_14+Sig_23;
        SIXTRL_REAL_T const d = Sig_24;

        SIXTRL_REAL_T sqrt_a2_c2 = sqrt(a*a+c*c);

        if (sqrt_a2_c2*sqrt_a2_c2*sqrt_a2_c2 < threshold_singular){
        //equivalent to: if np.abs(c)<threshold_singular and np.abs(a)<threshold_singular:

            if (fabs(d)> threshold_singular){
                cos2theta = fabs(b)/sqrt(b*b+4*d*d);
                }
            else{
                cos2theta = 1.;
                } // Decoupled beam

            costheta = sqrt(0.5*(1.+cos2theta));
            sintheta = mysign(b)*mysign(d)*sqrt(0.5*(1.-cos2theta));

            dS_costheta = 0.;
            dS_sintheta = 0.;

            Sig_11_hat = 0.5*W;
            Sig_33_hat = 0.5*W;

            dS_Sig_11_hat = 0.5*dS_W;
            dS_Sig_33_hat = 0.5*dS_W;
        }
        else{
            //~ printf("I am here\n");
            //~ printf("a=%.2e c=%.2e\n", a, c);
            sqrt_a2_c2 = sqrt(a*a+c*c); //repeated?
            cos2theta = fabs(2.*a)/(2*sqrt_a2_c2);
            costheta = sqrt(0.5*(1.+cos2theta));
            sintheta = mysign(a)*mysign(c)*sqrt(0.5*(1.-cos2theta));

            dS_cos2theta = mysign(a)*(0.5*b/sqrt_a2_c2-a*(a*b+2.*c*d)/(2.*sqrt_a2_c2*sqrt_a2_c2*sqrt_a2_c2));

            dS_costheta = 1./(4.*costheta)*dS_cos2theta;
            if (fabs(sintheta)>threshold_singular){
            //equivalent to: if np.abs(c)>threshold_singular:
                dS_sintheta = -1./(4.*sintheta)*dS_cos2theta;
            }
            else{
                dS_sintheta = d/(2.*a);
            }

            Sig_11_hat = 0.5*W;
            Sig_33_hat = 0.5*W;

            dS_Sig_11_hat = 0.5*dS_W + mysign(a)*sqrt_a2_c2;
            dS_Sig_33_hat = 0.5*dS_W - mysign(a)*sqrt_a2_c2;
        }
    }
    else{

        SIXTRL_REAL_T const sqrtT = sqrt(T);
        cos2theta = signR*R/sqrtT;
        costheta = sqrt(0.5*(1.+cos2theta));
        sintheta = signR*mysign(Sig_13)*sqrt(0.5*(1.-cos2theta));

        //in sixtrack this line seems to be different different
        // sintheta = -mysign((Sig_11-Sig_33))*np.sqrt(0.5*(1.-cos2theta))

        Sig_11_hat = 0.5*(W+signR*sqrtT);
        Sig_33_hat = 0.5*(W-signR*sqrtT);

        dS_cos2theta = signR*(dS_R/sqrtT - R/(2*sqrtT*sqrtT*sqrtT)*dS_T);
        dS_costheta = 1./(4.*costheta)*dS_cos2theta;

        if (fabs(sintheta)<threshold_singular && handle_singularities){
        //equivalent to to np.abs(Sig_13)<threshold_singular
            dS_sintheta = (Sig_14+Sig_23)/R;
        }
        else{
            dS_sintheta = -1./(4.*sintheta)*dS_cos2theta;
        }

        dS_Sig_11_hat = 0.5*(dS_W + signR*0.5/sqrtT*dS_T);
        dS_Sig_33_hat = 0.5*(dS_W - signR*0.5/sqrtT*dS_T);
    }

    *Sig_11_hat_ptr = Sig_11_hat;
    *Sig_33_hat_ptr = Sig_33_hat;
    *costheta_ptr = costheta;
    *sintheta_ptr = sintheta;
    *dS_Sig_11_hat_ptr = dS_Sig_11_hat;
    *dS_Sig_33_hat_ptr = dS_Sig_33_hat;
    *dS_costheta_ptr = dS_costheta;
    *dS_sintheta_ptr = dS_sintheta;

}

/* ************************************************************************* */
/* SCCoasting: */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)*
NS(SCCoasting_preset)( SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)*
    SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    if( sc_elem != SIXTRL_NULLPTR ) NS(SCCoasting_clear)( sc_elem );
    return sc_elem;
}

SIXTRL_INLINE NS(arch_status_t) NS(SCCoasting_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)*
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T real_t;

    NS(arch_status_t) status = NS(SCCoasting_set_number_of_particles)(
        sc_elem, ( real_t )0 );
    status |= NS(SCCoasting_set_circumference)( sc_elem, ( real_t )1 );
    status |= NS(SCCoasting_set_sigma_x)( sc_elem, ( real_t )1 );
    status |= NS(SCCoasting_set_sigma_y)( sc_elem, ( real_t )1 );
    status |= NS(SCCoasting_set_length)( sc_elem, ( real_t )0 );
    status |= NS(SCCoasting_set_x_co)( sc_elem, ( real_t )0 );
    status |= NS(SCCoasting_set_y_co)( sc_elem, ( real_t )0 );
    status |= NS(SCCoasting_set_min_sigma_diff)(
        sc_elem, ( real_t )1e-10 );
    status |= NS(SCCoasting_set_enabled)( sc_elem, true );

    return status;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(object_type_id_t) NS(SCCoasting_type_id)(
    void ) SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(OBJECT_TYPE_SC_COASTING);
}

SIXTRL_INLINE NS(buffer_size_t) NS(SCCoasting_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( sc_elem ) ) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )0;
}

SIXTRL_INLINE NS(buffer_size_t) NS(SCCoasting_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( sc_elem ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(buffer_size_t) num_slots = ( NS(buffer_size_t) )0u;

    if( num_slots > ( NS(buffer_size_t) )0u )
    {
        NS(buffer_size_t) const required_size =
            NS(ManagedBuffer_get_slot_based_length)(
                sizeof( NS(SCCoasting) ), slot_size );

        num_slots = required_size / slot_size;
        if( num_slots * slot_size < required_size ) ++num_slots;
    }

    return num_slots;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(SCCoasting_number_of_particles)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->number_of_particles;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SCCoasting_circumference)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->circumference;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SCCoasting_sigma_x)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->sigma_x;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SCCoasting_sigma_y)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->sigma_y;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SCCoasting_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->length;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SCCoasting_x_co)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->x_co;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SCCoasting_y_co)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->y_co;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SCCoasting_min_sigma_diff)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->min_sigma_diff;
}

SIXTRL_INLINE bool NS(SCCoasting_enabled)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return ( sc_elem->enabled == ( SIXTRL_UINT64_T )1 );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t) NS(SCCoasting_set_number_of_particles)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)* SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const number_of_particles ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->number_of_particles = number_of_particles;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SCCoasting_set_circumference)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)* SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const circumference ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->circumference = circumference;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SCCoasting_set_sigma_x)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)* SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const sigma_x ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->sigma_x = sigma_x;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SCCoasting_set_sigma_y)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)* SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const sigma_y ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->sigma_y = sigma_y;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SCCoasting_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)* SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const length ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->length = length;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SCCoasting_set_x_co)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)* SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const x_co ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->x_co = x_co;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SCCoasting_set_y_co)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)* SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const y_co ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->y_co = y_co;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SCCoasting_set_min_sigma_diff)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)* SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const min_sigma_diff ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->min_sigma_diff = min_sigma_diff;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SCCoasting_set_enabled)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)* SIXTRL_RESTRICT sc_elem,
    bool const is_enabled ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->enabled = ( is_enabled )
        ? ( SIXTRL_UINT64_T )1 : ( SIXTRL_UINT64_T )0;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SCCoasting) const*
NS(SCCoasting_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC
    const NS(Object) *const SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
{
    return (
        ( NS(Object_get_type_id)( obj ) == NS(OBJECT_TYPE_SC_COASTING) ) &&
        ( NS(Object_get_size)( obj ) >= sizeof( NS(SCCoasting) ) ) )
        ? ( SIXTRL_BE_ARGPTR_DEC NS(SCCoasting) const* )( uintptr_t
            )NS(Object_get_begin_addr)( obj )
        : SIXTRL_NULLPTR;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)*
NS(SCCoasting_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC
    NS(Object)* SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)*
        )NS(SCCoasting_const_from_obj_index)( obj );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SCCoasting) const*
NS(SCCoasting_const_from_managed_buffer)( SIXTRL_BUFFER_DATAPTR_DEC
        unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(SCCoasting_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)*
NS(SCCoasting_from_managed_buffer)( SIXTRL_BUFFER_DATAPTR_DEC
        unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
   return NS(SCCoasting_from_obj_index)(
        NS(ManagedBuffer_get_object)( buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE NS(arch_status_t) NS(SCCoasting_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(SCCoasting)* SIXTRL_RESTRICT dest,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(SCCoasting)
        *const SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( (  dest != SIXTRL_NULLPTR ) && ( src != SIXTRL_NULLPTR ) )
    {
        if( dest != src )
        {
            status = NS(SCCoasting_set_number_of_particles)( dest,
                NS(SCCoasting_number_of_particles)( src ) );

            status |= NS(SCCoasting_set_circumference)( dest,
                NS(SCCoasting_circumference)( src ) );

            status |= NS(SCCoasting_set_sigma_x)( dest,
                NS(SCCoasting_sigma_x)( src ) );

            status |= NS(SCCoasting_set_sigma_y)( dest,
                NS(SCCoasting_sigma_y)( src ) );

            status |= NS(SCCoasting_set_length)( dest,
                NS(SCCoasting_length)( src ) );

            status |= NS(SCCoasting_set_x_co)( dest,
                NS(SCCoasting_x_co)( src ) );

            status |= NS(SCCoasting_set_y_co)( dest,
                NS(SCCoasting_y_co)( src ) );

            status |= NS(SCCoasting_set_min_sigma_diff)( dest,
                NS(SCCoasting_min_sigma_diff)( src ) );

            status |= NS(SCCoasting_set_enabled)( dest,
                NS(SCCoasting_enabled)( src ) );
        }
        else
        {
            status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
        }
    }

    return status;
}

/* ************************************************************************* */
/* SCQGaussProfile: */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)*
NS(SCQGaussProfile_preset)( SIXTRL_BE_ARGPTR_DEC
    NS(SCQGaussProfile)* SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    if( sc_elem != SIXTRL_NULLPTR )
    {
        NS(SCQGaussProfile_clear)( sc_elem );
    }

    return sc_elem;
}

SIXTRL_INLINE NS(arch_status_t) NS(SCQGaussProfile_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)*
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T real_t;
    NS(arch_status_t) status =
        NS(SCQGaussProfile_set_number_of_particles)(
            sc_elem, ( real_t )0 );

    status |= NS(SCQGaussProfile_set_bunchlength_rms)(
        sc_elem, ( real_t )1 );

    status |= NS(SCQGaussProfile_set_sigma_x)(
        sc_elem, ( real_t )1 );

    status |= NS(SCQGaussProfile_set_sigma_y)(
        sc_elem, ( real_t )1 );

    status |= NS(SCQGaussProfile_set_length)(
        sc_elem, ( real_t )0 );

    status |= NS(SCQGaussProfile_set_x_co)( sc_elem, ( real_t )0 );
    status |= NS(SCQGaussProfile_set_y_co)( sc_elem, ( real_t )0 );
    status |= NS(SCQGaussProfile_set_q_param)(
        sc_elem, ( real_t )1 );

    status |= NS(SCQGaussProfile_set_enabled)( sc_elem, true );
    return status;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(object_type_id_t) NS(SCQGaussProfile_type_id)(
    void ) SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(OBJECT_TYPE_SC_QGAUSSIAN_PROF);
}

SIXTRL_INLINE NS(buffer_size_t) NS(SCQGaussProfile_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( sc_elem ) ) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(SCQGaussProfile_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( sc_elem ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(buffer_size_t) num_slots = ( NS(buffer_size_t) )0u;
    if( slot_size > ( NS(buffer_size_t) )0u )
    {
        NS(buffer_size_t) const extent =
            NS(ManagedBuffer_get_slot_based_length)(
                sizeof( NS(SCQGaussProfile) ), slot_size );
        num_slots = ( extent / slot_size );
        if( ( num_slots * slot_size ) < extent ) ++num_slots;
        SIXTRL_ASSERT( ( num_slots * slot_size ) >= extent );
    }

    return num_slots;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(SCQGaussProfile_number_of_particles)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->number_of_particles;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SCQGaussProfile_bunchlength_rms)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->bunchlength_rms;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SCQGaussProfile_sigma_x)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->sigma_x;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SCQGaussProfile_sigma_y)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->sigma_y;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SCQGaussProfile_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->length;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SCQGaussProfile_x_co)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->x_co;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SCQGaussProfile_y_co)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->y_co;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SCQGaussProfile_q_param)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->q_param;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SCQGaussProfile_cq)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->cq;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SCQGaussProfile_min_sigma_diff)(
    SIXTRL_BE_ARGPTR_DEC const NS(SCQGaussProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->min_sigma_diff;
}

SIXTRL_INLINE bool NS(SCQGaussProfile_enabled)( SIXTRL_BE_ARGPTR_DEC
    const NS(SCQGaussProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return ( sc_elem->enabled == ( SIXTRL_UINT64_T )1 );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t)
NS(SCQGaussProfile_set_number_of_particles)( SIXTRL_BE_ARGPTR_DEC
        NS(SCQGaussProfile)* SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const number_of_particles ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->number_of_particles = number_of_particles;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t)
NS(SCQGaussProfile_set_bunchlength_rms)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)* SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const bunchlength_rms ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->bunchlength_rms = bunchlength_rms;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SCQGaussProfile_set_sigma_x)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)* SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const sigma_x ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->sigma_x = sigma_x;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SCQGaussProfile_set_sigma_y)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)* SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const sigma_y ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->sigma_y = sigma_y;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SCQGaussProfile_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)* SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const length ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->length = length;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SCQGaussProfile_set_x_co)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)* SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const x_co ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->x_co = x_co;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SCQGaussProfile_set_y_co)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)* SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const y_co ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->y_co = y_co;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SCQGaussProfile_set_min_sigma_diff)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)*
        SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const min_sigma_diff ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->min_sigma_diff = min_sigma_diff;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SCQGaussProfile_set_q_param)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)*
        SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const q_param ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( sc_elem != SIXTRL_NULLPTR ) && ( q_param < ( SIXTRL_REAL_T )3 ) )
    {
        sc_elem->q_param = q_param;
        sc_elem->cq = NS(Math_q_gauss_cq)( q_param );
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(SCQGaussProfile_set_enabled)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)*
        SIXTRL_RESTRICT sc_elem, bool const is_enabled ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->enabled = ( is_enabled )
        ? ( SIXTRL_UINT64_T )1 : ( SIXTRL_UINT64_T )0;
        return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile) const*
NS(SCQGaussProfile_const_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const
        SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
{
    return ( ( NS(Object_get_type_id)( obj ) ==
               NS(OBJECT_TYPE_SC_QGAUSSIAN_PROF) ) &&
             ( NS(Object_get_size)( obj ) >=
               sizeof( NS(SCQGaussProfile) ) ) )
             ? ( SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile) const* )(
                 uintptr_t )NS(Object_get_begin_addr)( obj )
             : SIXTRL_NULLPTR;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)*
NS(SCQGaussProfile_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC
    NS(Object)* SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)*
        )NS(SCQGaussProfile_const_from_obj_index)( obj );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile) const*
NS(SCQGaussProfile_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(SCQGaussProfile_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)*
NS(SCQGaussProfile_from_managed_buffer)( SIXTRL_BUFFER_DATAPTR_DEC
        unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(SCQGaussProfile_from_obj_index)(
        NS(ManagedBuffer_get_object)( buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE NS(arch_status_t) NS(SCQGaussProfile_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(SCQGaussProfile)* SIXTRL_RESTRICT dest,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(SCQGaussProfile)
        *const SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( (  dest != SIXTRL_NULLPTR ) && ( src != SIXTRL_NULLPTR ) )
    {
        if( dest != src )
        {
            status = NS(SCQGaussProfile_set_number_of_particles)( dest,
                NS(SCQGaussProfile_number_of_particles)( src ) );

            status |= NS(SCQGaussProfile_set_bunchlength_rms)( dest,
                NS(SCQGaussProfile_bunchlength_rms)( src ) );

            status |= NS(SCQGaussProfile_set_sigma_x)( dest,
                NS(SCQGaussProfile_sigma_x)( src ) );

            status |= NS(SCQGaussProfile_set_sigma_y)( dest,
                NS(SCQGaussProfile_sigma_y)( src ) );

            status |= NS(SCQGaussProfile_set_length)( dest,
                NS(SCQGaussProfile_length)( src ) );

            status |= NS(SCQGaussProfile_set_x_co)( dest,
                NS(SCQGaussProfile_x_co)( src ) );

            status |= NS(SCQGaussProfile_set_y_co)( dest,
                NS(SCQGaussProfile_y_co)( src ) );

            status |= NS(SCQGaussProfile_set_min_sigma_diff)( dest,
                NS(SCQGaussProfile_min_sigma_diff)( src ) );

            status |= NS(SCQGaussProfile_set_q_param)( dest,
                NS(SCQGaussProfile_q_param)( src ) );

            status |= NS(SCQGaussProfile_set_enabled)( dest,
                NS(SCQGaussProfile_enabled)( src ) );
        }
        else
        {
            status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
        }
    }

    return status;
}

/* ************************************************************************* */
/* SpaceChargeInterpolatedProfile: */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)*
NS(LineDensityProfileData_preset)( SIXTRL_BE_ARGPTR_DEC
    NS(LineDensityProfileData)* SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    if( data != SIXTRL_NULLPTR ) NS(LineDensityProfileData_clear)( data );
    return data;
}

SIXTRL_INLINE NS(arch_status_t)
NS(LineDensityProfileData_clear)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(LineDensityProfileData)* SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T real_t;
    typedef NS(math_abscissa_idx_t) absc_t;
    typedef NS(buffer_addr_t) addr_t;

    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    status = NS(LineDensityProfileData_set_method)(
        data, NS(MATH_INTERPOL_NONE) );

    status |= NS(LineDensityProfileData_set_capacity)( data, ( absc_t )2 );
    status |= NS(LineDensityProfileData_set_values_addr)( data, ( addr_t )0 );
    status |= NS(LineDensityProfileData_set_derivatives_addr)( data, ( addr_t )0 );
    status |= NS(LineDensityProfileData_set_z0)( data, ( real_t )0 );
    status |= NS(LineDensityProfileData_set_dz)( data, ( real_t )1 );
    status |= NS(LineDensityProfileData_set_num_values)( data, ( absc_t )0 );

    return status;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(object_type_id_t) NS(LineDensityProfileData_type_id)(
    void ) SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(OBJECT_TYPE_LINE_DENSITY_PROF_DATA);
}

SIXTRL_INLINE NS(buffer_size_t) NS(LineDensityProfileData_num_dataptrs)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( data ) ) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )2u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(LineDensityProfileData_num_slots)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) st_size_t;
    typedef NS(math_abscissa_idx_t) absc_t;

    st_size_t num_slots = ( st_size_t )0u;
    if( slot_size > ( st_size_t )0u )
    {
        st_size_t required_size = NS(ManagedBuffer_get_slot_based_length)(
            sizeof( NS(LineDensityProfileData) ), slot_size );

        SIXTRL_ASSERT( NS(LineDensityProfileData_num_dataptrs)( data ) ==
            ( st_size_t )2u );

        if( ( data != SIXTRL_NULLPTR ) && ( data->capacity > ( absc_t )0 ) )
        {
            st_size_t const capacity = ( st_size_t
                )NS(LineDensityProfileData_capacity)( data );

            st_size_t const field_size =
            NS(ManagedBuffer_get_slot_based_length)(
                sizeof( SIXTRL_REAL_T ) * capacity, slot_size );

            required_size += ( st_size_t )2 * field_size;
        }

        num_slots = required_size / slot_size;

        if( ( num_slots * slot_size ) < required_size ) ++num_slots;
    }

    return num_slots;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(math_interpol_t) NS(LineDensityProfileData_method)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    typedef NS(math_interpol_t) method_t;
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );

    SIXTRL_ASSERT(
        ( ( method_t )data->method == NS(MATH_INTERPOL_NONE) ) ||
        ( ( method_t )data->method == NS(MATH_INTERPOL_LINEAR) ) ||
        ( ( method_t )data->method == NS(MATH_INTERPOL_CUBIC) ) );

    return ( method_t )data->method;
}

SIXTRL_INLINE NS(math_abscissa_idx_t) NS(LineDensityProfileData_num_values)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->num_values;
}

SIXTRL_INLINE NS(math_abscissa_idx_t) NS(LineDensityProfileData_capacity)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->capacity;
}

SIXTRL_INLINE NS(buffer_addr_t) NS(LineDensityProfileData_values_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->values_addr;
}

SIXTRL_INLINE NS(buffer_addr_t) NS(LineDensityProfileData_derivatives_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->derivatives_addr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const*
NS(LineDensityProfileData_const_values_begin)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(LineDensityProfileData) *const SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return ( SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* )( uintptr_t
        )NS(LineDensityProfileData_values_addr)( data );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const*
NS(LineDensityProfileData_const_values_end)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(LineDensityProfileData) *const SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* ptr_end =
        NS(LineDensityProfileData_const_values_begin)( data );

    NS(math_abscissa_idx_t) const num_values =
        NS(LineDensityProfileData_num_values)( data );

    if( ( ptr_end != SIXTRL_NULLPTR ) &&
        ( num_values >= ( NS(math_abscissa_idx_t) )0 ) )
    {
        ptr_end = ptr_end + num_values;
    }

    return ptr_end;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T*
NS(LineDensityProfileData_values_begin)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(LineDensityProfileData)* SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return ( SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T* )( uintptr_t
        )NS(LineDensityProfileData_values_addr)( data );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T*
NS(LineDensityProfileData_values_end)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(LineDensityProfileData)* SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T*
        )NS(LineDensityProfileData_const_values_end)( data );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(LineDensityProfileData_value_at_idx)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data,
    NS(math_abscissa_idx_t) const idx ) SIXTRL_NOEXCEPT
{
    SIXTRL_BE_ARGPTR_DEC SIXTRL_REAL_T const* ptr_begin =
        NS(LineDensityProfileData_const_values_begin)( data );

    SIXTRL_ASSERT( ptr_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( idx < NS(LineDensityProfileData_num_values)( data ) );

    return ptr_begin[ idx ];
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const*
NS(LineDensityProfileData_const_derivatives_begin)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return ( SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* )( uintptr_t
        )NS(LineDensityProfileData_derivatives_addr)( data );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const*
NS(LineDensityProfileData_const_derivatives_end)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* ptr_end =
        NS(LineDensityProfileData_const_derivatives_begin)( data );

    NS(math_abscissa_idx_t) const num_values =
        NS(LineDensityProfileData_num_values)( data );

    if( ( ptr_end != SIXTRL_NULLPTR ) &&
        ( num_values >= ( NS(math_abscissa_idx_t) )0 ) )
    {
        ptr_end = ptr_end + num_values;
    }

    return ptr_end;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T*
NS(LineDensityProfileData_derivatives_begin)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(LineDensityProfileData)* SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T*
        )NS(LineDensityProfileData_const_derivatives_begin)( data );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T*
NS(LineDensityProfileData_derivatives_end)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(LineDensityProfileData)* SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T*
        )NS(LineDensityProfileData_const_derivatives_end)( data );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(LineDensityProfileData_derivatives_at_idx)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data,
    NS(math_abscissa_idx_t) const idx ) SIXTRL_NOEXCEPT
{
    SIXTRL_BE_ARGPTR_DEC SIXTRL_REAL_T const* ptr_begin =
        NS(LineDensityProfileData_const_derivatives_begin)( data );

    SIXTRL_ASSERT( ptr_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( idx < NS(LineDensityProfileData_num_values)( data ) );

    return ptr_begin[ idx ];
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_REAL_T NS(LineDensityProfileData_z0)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->z0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(LineDensityProfileData_dz)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
            SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->dz;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(LineDensityProfileData_z_min)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
            SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    return data->z0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(LineDensityProfileData_z_max)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
            SIXTRL_RESTRICT data ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( data->dz > ( SIXTRL_REAL_T )0 );
    SIXTRL_ASSERT( data->num_values > ( NS(math_abscissa_idx_t) )0 );
    return data->z0 + data->dz * data->num_values;
}

SIXTRL_INLINE NS(math_abscissa_idx_t) NS(LineDensityProfileData_find_idx)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data, SIXTRL_REAL_T const z ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( z >= NS(LineDensityProfileData_z_min)( data ) );
    SIXTRL_ASSERT( z <= NS(LineDensityProfileData_z_max)( data ) );
    SIXTRL_ASSERT( NS(LineDensityProfileData_values_addr)( data ) !=
                   ( NS(buffer_addr_t) )0 );
    SIXTRL_ASSERT( NS(LineDensityProfileData_derivatives_addr)( data ) !=
                   ( NS(buffer_addr_t) )0 );

    return NS(Math_abscissa_index_equ)(
        z, data->z0, data->dz, data->num_values );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_REAL_T NS(LineDensityProfileData_interpolate_value)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data, SIXTRL_REAL_T const z ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( z >= NS(LineDensityProfileData_z_min)( data ) );
    SIXTRL_ASSERT( z <= NS(LineDensityProfileData_z_max)( data ) );
    SIXTRL_ASSERT( NS(LineDensityProfileData_values_addr)( data ) !=
                   ( NS(buffer_addr_t) )0 );
    SIXTRL_ASSERT( NS(LineDensityProfileData_derivatives_addr)( data ) !=
                   ( NS(buffer_addr_t) )0 );

    return NS(Math_interpol_y_equ)( z,
        NS(LineDensityProfileData_z0)( data ),
        NS(LineDensityProfileData_dz)( data ),
        NS(LineDensityProfileData_const_values_begin)( data ),
        NS(LineDensityProfileData_const_derivatives_begin)( data ),
        NS(LineDensityProfileData_num_values)( data ),
        NS(LineDensityProfileData_method)( data ) );
}

SIXTRL_INLINE SIXTRL_REAL_T
NS(LineDensityProfileData_interpolate_1st_derivative)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data, SIXTRL_REAL_T const z ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( z >= NS(LineDensityProfileData_z_min)( data ) );
    SIXTRL_ASSERT( z <= NS(LineDensityProfileData_z_max)( data ) );
    SIXTRL_ASSERT( NS(LineDensityProfileData_values_addr)( data ) !=
                   ( NS(buffer_addr_t) )0 );
    SIXTRL_ASSERT( NS(LineDensityProfileData_derivatives_addr)( data ) !=
                   ( NS(buffer_addr_t) )0 );

    return NS(Math_interpol_yp_equ)( z,
        NS(LineDensityProfileData_z0)( data ),
        NS(LineDensityProfileData_dz)( data ),
        NS(LineDensityProfileData_const_values_begin)( data ),
        NS(LineDensityProfileData_const_derivatives_begin)( data ),
        NS(LineDensityProfileData_num_values)( data ),
        NS(LineDensityProfileData_method)( data ) );
}

SIXTRL_INLINE SIXTRL_REAL_T
NS(LineDensityProfileData_interpolate_2nd_derivative)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT data, SIXTRL_REAL_T const z ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( z >= NS(LineDensityProfileData_z_min)( data ) );
    SIXTRL_ASSERT( z <= NS(LineDensityProfileData_z_max)( data ) );
    SIXTRL_ASSERT( NS(LineDensityProfileData_values_addr)( data ) !=
                   ( NS(buffer_addr_t) )0 );
    SIXTRL_ASSERT( NS(LineDensityProfileData_derivatives_addr)( data ) !=
                   ( NS(buffer_addr_t) )0 );

    return NS(Math_interpol_ypp_equ)( z,
        NS(LineDensityProfileData_z0)( data ),
        NS(LineDensityProfileData_dz)( data ),
        NS(LineDensityProfileData_const_values_begin)( data ),
        NS(LineDensityProfileData_const_derivatives_begin)( data ),
        NS(LineDensityProfileData_num_values)( data ),
        NS(LineDensityProfileData_method)( data ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t)
NS(LineDensityProfileData_prepare_interpolation)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT data,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T*
        SIXTRL_RESTRICT temp_data ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_addr_t) addr_t;
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_DEBUGGING_GENERAL_FAILURE;

    if( ( data != SIXTRL_NULLPTR ) &&
        ( NS(LineDensityProfileData_values_addr)( data ) != ( addr_t )0 ) &&
        ( NS(LineDensityProfileData_derivatives_addr)( data ) != ( addr_t )0 ) &&
        ( NS(LineDensityProfileData_num_values)( data ) >
          ( NS(math_abscissa_idx_t) )0 ) )
    {
        status = NS(Math_interpol_prepare_equ)(
            NS(LineDensityProfileData_derivatives_begin)( data ), temp_data,
            NS(LineDensityProfileData_const_values_begin)( data ),
            NS(LineDensityProfileData_z0)( data ),
            NS(LineDensityProfileData_dz)( data ),
            NS(LineDensityProfileData_num_values)( data ),
            NS(LineDensityProfileData_method)( data ) );
    }

    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t) NS(LineDensityProfileData_set_z0)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT data,
    SIXTRL_REAL_T const z0 ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    data->z0 = z0;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(LineDensityProfileData_set_dz)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT data,
    SIXTRL_REAL_T const dz ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( dz >= ( SIXTRL_REAL_T )0 );
    data->dz = dz;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(LineDensityProfileData_set_num_values)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT data,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    data->num_values = num_values;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(LineDensityProfileData_set_capacity)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT data,
    NS(math_abscissa_idx_t) const capacity ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    data->capacity = capacity;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(LineDensityProfileData_set_values_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT data,
    NS(buffer_addr_t) const values_addr ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    data->values_addr = values_addr;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(LineDensityProfileData_set_derivatives_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT data,
    NS(buffer_addr_t) const derivatives_addr ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    data->derivatives_addr = derivatives_addr;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(LineDensityProfileData_set_method)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT data,
    NS(math_interpol_t) const method ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( data != SIXTRL_NULLPTR );
    data->method = ( NS(math_interpol_int_t) )method;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LineDensityProfileData) const*
NS(LineDensityProfileData_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC
    const NS(Object) *const SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
{
    return ( ( NS(Object_get_type_id)( obj ) ==
               NS(OBJECT_TYPE_LINE_DENSITY_PROF_DATA) ) &&
             ( NS(Object_get_size)( obj ) >=
               sizeof( NS(LineDensityProfileData) ) ) )
        ? ( SIXTRL_BE_ARGPTR_DEC NS(LineDensityProfileData) const* )( uintptr_t
            )NS(Object_get_begin_addr)( obj )
        : SIXTRL_NULLPTR;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)*
NS(LineDensityProfileData_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC
    NS(Object)* SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)*
        )NS(LineDensityProfileData_const_from_obj_index)( obj );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LineDensityProfileData) const*
NS(LineDensityProfileData_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(LineDensityProfileData_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( buffer_begin, index, slot_size) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)*
NS(LineDensityProfileData_from_managed_buffer)( SIXTRL_BUFFER_DATAPTR_DEC
        unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(LineDensityProfileData_from_obj_index)(
        NS(ManagedBuffer_get_object)( buffer_begin, index, slot_size) );
}

SIXTRL_INLINE NS(arch_status_t) NS(LineDensityProfileData_copy)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* SIXTRL_RESTRICT dest,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData)
        *const SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( dest != SIXTRL_NULLPTR ) && ( src != SIXTRL_NULLPTR ) &&
        ( NS(LineDensityProfileData_capacity)( dest ) >=
          NS(LineDensityProfileData_num_values)( src ) ) &&
        ( ( ( NS(LineDensityProfileData_values_addr)( dest ) !=
              NS(LineDensityProfileData_values_addr)( src ) ) &&
            ( NS(LineDensityProfileData_values_addr)( dest ) !=
              ( NS(buffer_addr_t) )0 ) &&
            ( NS(LineDensityProfileData_values_addr)( src ) !=
              ( NS(buffer_addr_t) )0 ) &&
            ( NS(LineDensityProfileData_derivatives_addr)( dest ) !=
              NS(LineDensityProfileData_derivatives_addr)( src ) ) &&
            ( NS(LineDensityProfileData_derivatives_addr)( dest ) !=
              ( NS(buffer_addr_t) )0 ) &&
            ( NS(LineDensityProfileData_derivatives_addr)( src ) !=
              ( NS(buffer_addr_t) )0 ) ) ||
          ( dest == src ) ) )
    {
        if( dest != src )
        {
            typedef NS(math_abscissa_idx_t) absc_t;
            absc_t ii = ( absc_t )0;
            absc_t const num_values =
                NS(LineDensityProfileData_num_values)( src );

            SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* src_values =
                NS(LineDensityProfileData_const_values_begin)( src );

            SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T* dest_values =
                NS(LineDensityProfileData_values_begin)( dest );

            SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* src_derivatives =
                NS(LineDensityProfileData_const_derivatives_begin)( src );

            SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T* dest_derivatives =
                NS(LineDensityProfileData_derivatives_begin)( dest );

            NS(LineDensityProfileData_set_method)( dest,
                NS(LineDensityProfileData_method)( src ) );

            NS(LineDensityProfileData_set_num_values)( dest, num_values );

            NS(LineDensityProfileData_set_z0)( dest,
                NS(LineDensityProfileData_z0)( src ) );

            NS(LineDensityProfileData_set_dz)( dest,
                NS(LineDensityProfileData_dz)( src ) );

            SIXTRL_ASSERT( src_values  != SIXTRL_NULLPTR );
            SIXTRL_ASSERT( dest_values != SIXTRL_NULLPTR );
            SIXTRL_ASSERT( dest_values != src_values );

            SIXTRL_ASSERT( src_derivatives  != SIXTRL_NULLPTR );
            SIXTRL_ASSERT( dest_derivatives != SIXTRL_NULLPTR );
            SIXTRL_ASSERT( dest_derivatives != src_derivatives );

            for( ; ii < num_values ; ++ii )
            {
                dest_values[ ii ] = src_values[ ii ];
                dest_derivatives[ ii ] = src_derivatives[ ii ];
            }
        }

        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

#if !defined( _GPUCODE )

SIXTRL_INLINE NS(buffer_size_t) NS(LineDensityProfileData_values_offset)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( data ) ) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )offsetof(
        NS(LineDensityProfileData), values_addr );
}

SIXTRL_INLINE NS(buffer_addr_t) NS(LineDensityProfileData_derivatives_offset)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(LineDensityProfileData) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( data ) ) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )offsetof(
        NS(LineDensityProfileData), derivatives_addr );
}

#endif /* !defined( _GPUCODE ) */

/* ************************************************************************* */
/* NS(SpaceChargeInterpolatedProfile) */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
NS(SpaceChargeInterpolatedProfile_preset)( SIXTRL_BE_ARGPTR_DEC
    NS(SpaceChargeInterpolatedProfile)* SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    if( sc_elem != SIXTRL_NULLPTR )
    {
        NS(SpaceChargeInterpolatedProfile_clear)( sc_elem );
    }

    return sc_elem;
}

SIXTRL_INLINE NS(arch_status_t) NS(SpaceChargeInterpolatedProfile_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T real_t;

    NS(arch_status_t) status =
        NS(SpaceChargeInterpolatedProfile_set_number_of_particles)(
            sc_elem, ( real_t )0 );

    status |= NS(SpaceChargeInterpolatedProfile_set_sigma_x)(
        sc_elem, ( real_t )1 );

    status |= NS(SpaceChargeInterpolatedProfile_set_sigma_y)(
        sc_elem, ( real_t )1 );

    status |= NS(SpaceChargeInterpolatedProfile_set_length)(
        sc_elem, ( real_t )0 );

    status |= NS(SpaceChargeInterpolatedProfile_set_x_co)(
        sc_elem, ( real_t )0 );

    status |= NS(SpaceChargeInterpolatedProfile_set_y_co)(
        sc_elem, ( real_t )0 );

    status |= NS(SpaceChargeInterpolatedProfile_set_interpol_data_addr)(
        sc_elem, ( NS(buffer_addr_t) )0 );

    status |= NS(SpaceChargeInterpolatedProfile_set_line_density_prof_fallback)(
        sc_elem, ( real_t )1 );

    status |= NS(SpaceChargeInterpolatedProfile_set_min_sigma_diff)(
        sc_elem, ( real_t )1e-10 );

    status |= NS(SpaceChargeInterpolatedProfile_set_enabled)( sc_elem, true );
    return status;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(object_type_id_t)
    NS(SpaceChargeInterpolatedProfile_type_id)() SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(OBJECT_TYPE_SC_INTERPOLATED_PROF);
}

SIXTRL_INLINE NS(buffer_size_t) NS(SpaceChargeInterpolatedProfile_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( sc_elem ) ) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(SpaceChargeInterpolatedProfile_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( sc_elem ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(buffer_size_t) num_slots = ( NS(buffer_size_t) )0u;

    if( num_slots > ( NS(buffer_size_t) )0u )
    {
        NS(buffer_size_t) const required_size =
            NS(ManagedBuffer_get_slot_based_length)(
                sizeof( NS(SpaceChargeInterpolatedProfile) ), slot_size );

        num_slots = required_size / slot_size;
        if( num_slots * slot_size < required_size ) ++num_slots;
    }

    return num_slots;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE bool NS(SpaceChargeInterpolatedProfile_has_interpol_data)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData) const* interpol_data =
        NS(SpaceChargeInterpolatedProfile_const_line_density_profile_data)(
            sc_elem );

    return ( ( interpol_data != SIXTRL_NULLPTR ) &&
             ( NS(LineDensityProfileData_method)( interpol_data ) !=
               NS(MATH_INTERPOL_NONE) ) &&
             ( NS(LineDensityProfileData_num_values)( interpol_data ) >=
              ( NS(math_abscissa_idx_t) )0 ) &&
             ( NS(LineDensityProfileData_const_values_begin)(
                 interpol_data ) != SIXTRL_NULLPTR ) &&
             ( NS(LineDensityProfileData_const_derivatives_begin)(
                 interpol_data ) != SIXTRL_NULLPTR ) );
}

SIXTRL_INLINE NS(buffer_addr_t)
NS(SpaceChargeInterpolatedProfile_interpol_data_addr)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->interpol_data_addr;
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(LineDensityProfileData) const*
NS(SpaceChargeInterpolatedProfile_const_line_density_profile_data)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData) const*
        )( uintptr_t )sc_elem->interpol_data_addr;
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(LineDensityProfileData)*
NS(SpaceChargeInterpolatedProfile_line_density_profile_data)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)* SIXTRL_RESTRICT
        sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)*
        )( uintptr_t )sc_elem->interpol_data_addr;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SpaceChargeInterpolatedProfile_number_of_particles)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->number_of_particles;
}

SIXTRL_INLINE SIXTRL_REAL_T
NS(SpaceChargeInterpolatedProfile_line_density_prof_fallback)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->line_density_prof_fallback;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SpaceChargeInterpolatedProfile_sigma_x)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->sigma_x;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SpaceChargeInterpolatedProfile_sigma_y)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->sigma_y;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SpaceChargeInterpolatedProfile_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->length;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SpaceChargeInterpolatedProfile_x_co)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->x_co;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SpaceChargeInterpolatedProfile_y_co)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->y_co;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SpaceChargeInterpolatedProfile_min_sigma_diff)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return sc_elem->min_sigma_diff;
}

SIXTRL_INLINE bool NS(SpaceChargeInterpolatedProfile_enabled)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT sc_elem ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    return ( sc_elem->enabled == ( SIXTRL_UINT64_T )1 );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t)
NS(SpaceChargeInterpolatedProfile_set_interpol_data_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem,
    NS(buffer_addr_t) const interpol_data_addr ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->interpol_data_addr = interpol_data_addr;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SpaceChargeInterpolatedProfile_set_number_of_particles)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const number_of_particles ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->number_of_particles = number_of_particles;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SpaceChargeInterpolatedProfile_set_sigma_x)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem, SIXTRL_REAL_T const sigma_x ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->sigma_x = sigma_x;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SpaceChargeInterpolatedProfile_set_sigma_y)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem, SIXTRL_REAL_T const sigma_y ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->sigma_y = sigma_y;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SpaceChargeInterpolatedProfile_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem, SIXTRL_REAL_T const length ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->length = length;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SpaceChargeInterpolatedProfile_set_x_co)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem, SIXTRL_REAL_T const x_co ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->x_co = x_co;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SpaceChargeInterpolatedProfile_set_y_co)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem, SIXTRL_REAL_T const y_co ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->y_co = y_co;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t)
NS(SpaceChargeInterpolatedProfile_set_line_density_prof_fallback)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const line_density_prof_fallback ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->line_density_prof_fallback = line_density_prof_fallback;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t)
NS(SpaceChargeInterpolatedProfile_set_min_sigma_diff)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const min_sigma_diff ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->min_sigma_diff = min_sigma_diff;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SpaceChargeInterpolatedProfile_set_enabled)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem, bool const is_enabled ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( sc_elem != SIXTRL_NULLPTR );
    sc_elem->enabled = ( is_enabled )
        ? ( SIXTRL_UINT64_T )1 : ( SIXTRL_UINT64_T )0;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_REAL_T
NS(SpaceChargeInterpolatedProfile_line_density_profile)( SIXTRL_BE_ARGPTR_DEC
    NS(SpaceChargeInterpolatedProfile)* SIXTRL_RESTRICT sc_elem,
    SIXTRL_REAL_T const z ) SIXTRL_NOEXCEPT
{
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData) const* interpol_data =
        NS(SpaceChargeInterpolatedProfile_const_line_density_profile_data)(
            sc_elem );

    SIXTRL_ASSERT( interpol_data != SIXTRL_NULLPTR );
    return NS(LineDensityProfileData_interpolate_value)( interpol_data, z );
}

SIXTRL_INLINE SIXTRL_REAL_T
NS(SpaceChargeInterpolatedProfile_line_density_profile_1st_derivative)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem, SIXTRL_REAL_T const z ) SIXTRL_NOEXCEPT
{
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData) const* interpol_data =
        NS(SpaceChargeInterpolatedProfile_const_line_density_profile_data)(
            sc_elem );

    SIXTRL_ASSERT( interpol_data != SIXTRL_NULLPTR );
    return NS(LineDensityProfileData_interpolate_1st_derivative)(
        interpol_data, z );
}

SIXTRL_INLINE SIXTRL_REAL_T
NS(SpaceChargeInterpolatedProfile_line_density_profile_2nd_derivative)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem, SIXTRL_REAL_T const z ) SIXTRL_NOEXCEPT
{
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData) const* interpol_data =
        NS(SpaceChargeInterpolatedProfile_const_line_density_profile_data)(
            sc_elem );

    SIXTRL_ASSERT( interpol_data != SIXTRL_NULLPTR );
    return NS(LineDensityProfileData_interpolate_2nd_derivative)(
        interpol_data, z );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t)
NS(SpaceChargeInterpolatedProfile_prepare_interpolation)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT sc_elem, SIXTRL_ARGPTR_DEC SIXTRL_REAL_T*
            SIXTRL_RESTRICT temp_data ) SIXTRL_NOEXCEPT
{
    SIXTRL_BUFFER_DATAPTR_DEC NS(LineDensityProfileData)* interpol_data =
        NS(SpaceChargeInterpolatedProfile_line_density_profile_data)( sc_elem );

    SIXTRL_ASSERT( interpol_data != SIXTRL_NULLPTR );
    return NS(LineDensityProfileData_prepare_interpolation)(
        interpol_data, temp_data );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile) const*
NS(SpaceChargeInterpolatedProfile_const_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const
        SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
{
    return ( ( NS(Object_get_type_id)( obj ) ==
               NS(OBJECT_TYPE_SC_INTERPOLATED_PROF) ) &&
             ( NS(Object_get_size)( obj ) >= sizeof(
               NS(SpaceChargeInterpolatedProfile) ) ) )
        ? ( SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile) const* )(
            uintptr_t )NS(Object_get_begin_addr)( obj )
        : SIXTRL_NULLPTR;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
NS(SpaceChargeInterpolatedProfile_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
        SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        )NS(SpaceChargeInterpolatedProfile_const_from_obj_index)( obj );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile) const*
NS(SpaceChargeInterpolatedProfile_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(SpaceChargeInterpolatedProfile_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
NS(SpaceChargeInterpolatedProfile_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(SpaceChargeInterpolatedProfile_from_obj_index)(
        NS(ManagedBuffer_get_object)( buffer_begin, index, slot_size ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t) NS(SpaceChargeInterpolatedProfile_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeInterpolatedProfile)*
        SIXTRL_RESTRICT dest,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile)
        *const SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( dest != SIXTRL_NULLPTR ) && ( src != SIXTRL_NULLPTR ) )
    {
        if( dest != src )
        {
            NS(SpaceChargeInterpolatedProfile_set_number_of_particles)( dest,
                NS(SpaceChargeInterpolatedProfile_number_of_particles)( src ) );

            NS(SpaceChargeInterpolatedProfile_set_sigma_x)( dest,
                NS(SpaceChargeInterpolatedProfile_sigma_x)( src ) );

            NS(SpaceChargeInterpolatedProfile_set_sigma_y)( dest,
                NS(SpaceChargeInterpolatedProfile_sigma_y)( src ) );

            NS(SpaceChargeInterpolatedProfile_set_length)( dest,
                NS(SpaceChargeInterpolatedProfile_length)( src ) );

            NS(SpaceChargeInterpolatedProfile_set_x_co)( dest,
                NS(SpaceChargeInterpolatedProfile_x_co)( src ) );

            NS(SpaceChargeInterpolatedProfile_set_y_co)( dest,
                NS(SpaceChargeInterpolatedProfile_y_co)( src ) );

            NS(SpaceChargeInterpolatedProfile_set_interpol_data_addr)( dest,
                NS(SpaceChargeInterpolatedProfile_interpol_data_addr)( src ) );

            NS(SpaceChargeInterpolatedProfile_set_line_density_prof_fallback)(
                dest, NS(SpaceChargeInterpolatedProfile_line_density_prof_fallback)(
                    src ) );

            NS(SpaceChargeInterpolatedProfile_set_min_sigma_diff)( dest,
                NS(SpaceChargeInterpolatedProfile_min_sigma_diff)( src ) );

            NS(SpaceChargeInterpolatedProfile_set_enabled)( dest,
                NS(SpaceChargeInterpolatedProfile_enabled)( src ) );
        }

        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

#if !defined( _GPUCODE )

SIXTRL_INLINE NS(buffer_size_t)
NS(SpaceChargeInterpolatedProfile_interpol_data_addr_offset)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeInterpolatedProfile) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( sc_elem ) ) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )offsetof(
        NS(SpaceChargeInterpolatedProfile), interpol_data_addr );
}

#endif /* _GPUCODE */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_BE_BEAMFIELDS_BE_BEAMFIELDS_C99_H__ */

/* sixtracklib/common/be_beamfields/be_beamfields.h */
