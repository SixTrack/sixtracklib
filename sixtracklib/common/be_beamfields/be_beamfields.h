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

typedef SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* NS(beambeam4d_real_ptr_t);
typedef SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const*
        NS(beambeam4d_real_const_ptr_t);

typedef struct NS(BeamBeam4D)
{
    SIXTRL_UINT64_T  size SIXTRL_ALIGN( 8 );
    NS(beambeam4d_real_ptr_t) SIXTRL_RESTRICT data SIXTRL_ALIGN( 8 );
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
    SIXTRL_INT64_T enabled          SIXTRL_ALIGN( 8 );
}NS(BB4D_data);

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BeamBeam4D_get_required_num_dataptrs_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BeamBeam4D_get_required_num_slots_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN  SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam );

SIXTRL_STATIC SIXTRL_FN void NS(BeamBeam4D_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BeamBeam4D_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const beam_beam );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BeamBeam4D_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const beam_beam );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(BeamBeam4D_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_size,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const data_size );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const  data_size,
    NS(beambeam4d_real_ptr_t) SIXTRL_RESTRICT input_data );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT orig );

#endif /* !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN NS(beambeam4d_real_const_ptr_t)
NS(BeamBeam4D_get_const_data)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_STATIC SIXTRL_FN NS(beambeam4d_real_ptr_t) NS(BeamBeam4D_get_data)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam );

SIXTRL_STATIC SIXTRL_FN SIXTRL_UINT64_T NS(BeamBeam4D_get_data_size)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_STATIC SIXTRL_FN void NS(BeamBeam4D_set_data)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam,
    SIXTRL_BE_ARGPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_data );

SIXTRL_STATIC SIXTRL_FN void NS(BeamBeam4D_set_data_size)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const data_size );

SIXTRL_STATIC SIXTRL_FN void NS(BeamBeam4D_assign_data_ptr)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam,
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT ptr_data );

SIXTRL_STATIC SIXTRL_FN int NS(BeamBeam4D_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT source );

/* ************************************************************************* */
/* SpaceChargeCoasting: */

typedef SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* NS(sc_coasting_real_ptr_t);
typedef SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const*
        NS(sc_coasting_real_const_ptr_t);

typedef struct NS(SpaceChargeCoasting)
{
    SIXTRL_UINT64_T size SIXTRL_ALIGN( 8 );
    NS(sc_coasting_real_ptr_t) SIXTRL_RESTRICT data SIXTRL_ALIGN( 8 );
}
NS(SpaceChargeCoasting);

/*
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
    SIXTRL_INT64_T enabled          SIXTRL_ALIGN( 8 );
}NS(SpaceChargeCoasting_data);
*/

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(SpaceChargeCoasting_get_required_num_dataptrs_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(SpaceChargeCoasting_get_required_num_slots_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN  SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeCoasting)*
NS(SpaceChargeCoasting_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeCoasting)* SIXTRL_RESTRICT beam_beam );

SIXTRL_STATIC SIXTRL_FN void NS(SpaceChargeCoasting_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeCoasting)* SIXTRL_RESTRICT beam_beam );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(SpaceChargeCoasting_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const beam_beam );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(SpaceChargeCoasting_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const beam_beam );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(SpaceChargeCoasting_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_size,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(SpaceChargeCoasting)*
NS(SpaceChargeCoasting_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const data_size );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(SpaceChargeCoasting)*
NS(SpaceChargeCoasting_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const  data_size,
    NS(sc_coasting_real_ptr_t) SIXTRL_RESTRICT input_data );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(SpaceChargeCoasting)*
NS(SpaceChargeCoasting_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT orig );

#endif /* !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN NS(sc_coasting_real_const_ptr_t)
NS(SpaceChargeCoasting_get_const_data)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_STATIC SIXTRL_FN NS(sc_coasting_real_ptr_t) NS(SpaceChargeCoasting_get_data)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeCoasting)* SIXTRL_RESTRICT beam_beam );

SIXTRL_STATIC SIXTRL_FN SIXTRL_UINT64_T NS(SpaceChargeCoasting_get_data_size)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_STATIC SIXTRL_FN void NS(SpaceChargeCoasting_set_data)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeCoasting)* SIXTRL_RESTRICT beam_beam,
    SIXTRL_BE_ARGPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_data );

SIXTRL_STATIC SIXTRL_FN void NS(SpaceChargeCoasting_set_data_size)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeCoasting)* SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const data_size );

SIXTRL_STATIC SIXTRL_FN void NS(SpaceChargeCoasting_assign_data_ptr)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeCoasting)* SIXTRL_RESTRICT beam_beam,
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT ptr_data );

SIXTRL_STATIC SIXTRL_FN int NS(SpaceChargeCoasting_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeCoasting)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT source );

/* ************************************************************************* */
/* SpaceChargeBunched: */

typedef SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* NS(sc_bunched_real_ptr_t);
typedef SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const*
        NS(sc_bunched_real_const_ptr_t);

typedef struct NS(SpaceChargeBunched)
{
    SIXTRL_UINT64_T size SIXTRL_ALIGN( 8 );
    NS(sc_bunched_real_ptr_t) SIXTRL_RESTRICT data SIXTRL_ALIGN( 8 );
}
NS(SpaceChargeBunched);

/*
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
    SIXTRL_INT64_T enabled          SIXTRL_ALIGN( 8 );
}NS(SpaceChargeBunched_data);
*/

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(SpaceChargeBunched_get_required_num_dataptrs_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(SpaceChargeBunched_get_required_num_slots_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN  SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeBunched)*
NS(SpaceChargeBunched_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeBunched)* SIXTRL_RESTRICT beam_beam );

SIXTRL_STATIC SIXTRL_FN void NS(SpaceChargeBunched_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeBunched)* SIXTRL_RESTRICT beam_beam );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(SpaceChargeBunched_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const beam_beam );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(SpaceChargeBunched_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const beam_beam );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(SpaceChargeBunched_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_size,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(SpaceChargeBunched)*
NS(SpaceChargeBunched_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const data_size );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(SpaceChargeBunched)*
NS(SpaceChargeBunched_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const  data_size,
    NS(sc_bunched_real_ptr_t) SIXTRL_RESTRICT input_data );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(SpaceChargeBunched)*
NS(SpaceChargeBunched_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const SIXTRL_RESTRICT orig );

#endif /* !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN NS(sc_bunched_real_const_ptr_t)
NS(SpaceChargeBunched_get_const_data)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_STATIC SIXTRL_FN NS(sc_bunched_real_ptr_t) NS(SpaceChargeBunched_get_data)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeBunched)* SIXTRL_RESTRICT beam_beam );

SIXTRL_STATIC SIXTRL_FN SIXTRL_UINT64_T NS(SpaceChargeBunched_get_data_size)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_STATIC SIXTRL_FN void NS(SpaceChargeBunched_set_data)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeBunched)* SIXTRL_RESTRICT beam_beam,
    SIXTRL_BE_ARGPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_data );

SIXTRL_STATIC SIXTRL_FN void NS(SpaceChargeBunched_set_data_size)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeBunched)* SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const data_size );

SIXTRL_STATIC SIXTRL_FN void NS(SpaceChargeBunched_assign_data_ptr)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeBunched)* SIXTRL_RESTRICT beam_beam,
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT ptr_data );

SIXTRL_STATIC SIXTRL_FN int NS(SpaceChargeBunched_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeBunched)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const SIXTRL_RESTRICT source );

/* ************************************************************************* */
/* BeamBeam6D: */

typedef SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* NS(beambeam6d_real_ptr_t);
typedef SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const*
        NS(beambeam6d_real_const_ptr_t);

typedef struct NS(BeamBeam6D)
{
    SIXTRL_UINT64_T                           size      SIXTRL_ALIGN( 8 );
    NS(beambeam6d_real_ptr_t) SIXTRL_RESTRICT data      SIXTRL_ALIGN( 8 );
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
    SIXTRL_INT64_T enabled           SIXTRL_ALIGN( 8 );

    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* N_part_per_slice   SIXTRL_ALIGN( 8 );
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* x_slices_star      SIXTRL_ALIGN( 8 );
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* y_slices_star      SIXTRL_ALIGN( 8 );
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* sigma_slices_star  SIXTRL_ALIGN( 8 );
}NS(BB6D_data);


SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BeamBeam6D_get_required_num_dataptrs_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BeamBeam6D_get_required_num_slots_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT beam_beam );

SIXTRL_STATIC SIXTRL_FN void NS(BeamBeam6D_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT beam_beam );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BeamBeam6D_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const beam_beam );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BeamBeam6D_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const beam_beam );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(BeamBeam6D_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_size,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const data_size );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const  data_size,
    NS(beambeam6d_real_ptr_t) SIXTRL_RESTRICT input_data );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT orig );

#endif /* !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN NS(beambeam6d_real_const_ptr_t)
NS(BeamBeam6D_get_const_data)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_STATIC SIXTRL_FN NS(beambeam6d_real_ptr_t) NS(BeamBeam6D_get_data)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT beam_beam );

SIXTRL_STATIC SIXTRL_FN SIXTRL_UINT64_T NS(BeamBeam6D_get_data_size)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_STATIC SIXTRL_FN void NS(BeamBeam6D_set_data)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT beam_beam,
    SIXTRL_BE_ARGPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_data );

SIXTRL_STATIC SIXTRL_FN void NS(BeamBeam6D_set_data_size)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const data_size );

SIXTRL_STATIC SIXTRL_FN void NS(BeamBeam6D_assign_data_ptr)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT beam_beam,
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT ptr_data );

SIXTRL_STATIC SIXTRL_FN int NS(BeamBeam6D_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT source );

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
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* BeamBeam4D: */

SIXTRL_INLINE NS(buffer_size_t)
NS(BeamBeam4D_get_required_num_dataptrs_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t num_dataptrs = ( buf_size_t )0u;

    if( ( buffer != SIXTRL_NULLPTR ) && ( slot_size > ( buf_size_t )0u ) &&
        ( beam_beam != SIXTRL_NULLPTR ) &&
        ( NS(BeamBeam4D_get_data_size)( beam_beam ) > ( buf_size_t )0u ) )
    {
        num_dataptrs = ( buf_size_t )1u;
    }

    return num_dataptrs;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(BeamBeam4D_get_required_num_slots_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t num_slots = ( buf_size_t )0u;

    if( ( buffer != SIXTRL_NULLPTR ) && ( slot_size > ( buf_size_t )0u ) &&
        ( beam_beam != SIXTRL_NULLPTR ) &&
        ( NS(BeamBeam4D_get_data_size)( beam_beam ) > ( buf_size_t )0u ) )
    {
        num_slots = NS(ManagedBuffer_get_slot_based_length)(
            NS(BeamBeam4D_get_data_size)( beam_beam ) * sizeof( SIXTRL_REAL_T ),
                slot_size );

        num_slots /= slot_size;
    }

    return num_slots;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* NS(BeamBeam4D_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam )
{
    if( beam_beam != SIXTRL_NULLPTR )
    {
        beam_beam->size = ( SIXTRL_UINT64_T )0u;
        beam_beam->data   = SIXTRL_NULLPTR;

        NS(BeamBeam4D_clear)( beam_beam );
    }

    return beam_beam;
}

SIXTRL_INLINE void NS(BeamBeam4D_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam )
{
    typedef NS(buffer_size_t ) buf_size_t;

    buf_size_t const data_size = NS(BeamBeam4D_get_data_size)( beam_beam );
    NS(beambeam4d_real_ptr_t) ptr_data = NS(BeamBeam4D_get_data)( beam_beam );

    if( ( data_size > ( NS(buffer_size_t) )0u ) &&
        ( ptr_data != SIXTRL_NULLPTR ) )
    {
        SIXTRL_REAL_T const Z = ( SIXTRL_REAL_T )0;
        SIXTRACKLIB_SET_VALUES( SIXTRL_REAL_T, ptr_data, data_size, Z );
    }

    return;
}

SIXTRL_INLINE NS(beambeam4d_real_const_ptr_t)
NS(BeamBeam4D_get_const_data)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT beam_beam )
{
    SIXTRL_ASSERT( beam_beam != SIXTRL_NULLPTR );
    return beam_beam->data;
}

SIXTRL_INLINE NS(beambeam4d_real_ptr_t) NS(BeamBeam4D_get_data)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam )
{
    return ( NS(beambeam4d_real_ptr_t) )NS(BeamBeam4D_get_const_data)( beam_beam );
}

SIXTRL_INLINE SIXTRL_UINT64_T NS(BeamBeam4D_get_data_size)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT beam_beam )
{
    SIXTRL_ASSERT( beam_beam != SIXTRL_NULLPTR );
    return beam_beam->size;
}



SIXTRL_INLINE void NS(BeamBeam4D_set_data)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam,
    SIXTRL_BE_ARGPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_data )
{
    typedef SIXTRL_REAL_T real_t;

    NS(buffer_size_t) const size =
        NS(BeamBeam4D_get_data_size)( beam_beam );

    NS(beambeam4d_real_ptr_t) ptr_dest_data =
        NS(BeamBeam4D_get_data)( beam_beam );

    SIXTRL_ASSERT( ptr_dest_data != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ptr_data      != SIXTRL_NULLPTR );
    SIXTRACKLIB_COPY_VALUES( real_t, ptr_dest_data, ptr_data, size );

    return;
}

SIXTRL_INLINE void NS(BeamBeam4D_set_data_size)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const data_size )
{
    SIXTRL_ASSERT( beam_beam != SIXTRL_NULLPTR );
    beam_beam->size = data_size;
    return;
}

SIXTRL_INLINE void NS(BeamBeam4D_assign_data_ptr)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam,
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT ptr_data )
{
    SIXTRL_ASSERT( beam_beam != SIXTRL_NULLPTR );
    beam_beam->data = ptr_data;
    return;
}

SIXTRL_INLINE int NS(BeamBeam4D_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT source )
{
    int success = -1;

    if( ( destination != SIXTRL_NULLPTR ) && ( source != SIXTRL_NULLPTR ) &&
        ( destination != source ) &&
        ( NS(BeamBeam4D_get_const_data)( destination ) != SIXTRL_NULLPTR ) &&
        ( NS(BeamBeam4D_get_const_data)( source      ) != SIXTRL_NULLPTR ) &&
        ( NS(BeamBeam4D_get_data_size)( destination ) ==
          NS(BeamBeam4D_get_data_size)( source ) ) )
    {
        SIXTRL_ASSERT( NS(BeamBeam4D_get_const_data)( destination ) !=
                       NS(BeamBeam4D_get_const_data)( source ) );

        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
            NS(BeamBeam4D_get_data)( destination ),
            NS(BeamBeam4D_get_const_data)( source ),
            NS(BeamBeam4D_get_data_size)( source ) );

        success = 0;
    }

    return success;
}

/* ************************************************************************* */
/* SpaceChargeBunched: */

SIXTRL_INLINE NS(buffer_size_t)
NS(SpaceChargeBunched_get_required_num_dataptrs_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const SIXTRL_RESTRICT sc,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t num_dataptrs = ( buf_size_t )0u;

    if( ( slot_size > ( buf_size_t )0u ) && ( buffer != SIXTRL_NULLPTR ) &&
        ( sc != SIXTRL_NULLPTR ) &&
        ( NS(SpaceChargeBunched_get_data_size)( sc ) > ( buf_size_t )0u ) )
    {
        num_dataptrs = ( buf_size_t )1u;
    }

    return num_dataptrs;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(SpaceChargeBunched_get_required_num_slots_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const SIXTRL_RESTRICT sc,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t num_slots = ( buf_size_t )0u;

    if( ( slot_size > ( buf_size_t )0u ) && ( buffer != SIXTRL_NULLPTR ) &&
        ( sc != SIXTRL_NULLPTR ) &&
        ( NS(SpaceChargeBunched_get_data_size)( sc ) > ( buf_size_t )0u ) )
    {
        num_slots = NS(ManagedBuffer_get_slot_based_length)(
            NS(SpaceChargeBunched_get_data_size)( sc ) * sizeof( SIXTRL_REAL_T ),
                slot_size );

        num_slots /= slot_size;
    }

    return num_slots;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeBunched)*
NS(SpaceChargeBunched_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeBunched)* SIXTRL_RESTRICT sc )
{
    if( sc != SIXTRL_NULLPTR )
    {
        sc->size = ( SIXTRL_UINT64_T )0u;
        sc->data   = SIXTRL_NULLPTR;

        NS(SpaceChargeBunched_clear)( sc );
    }

    return sc;
}

SIXTRL_INLINE void NS(SpaceChargeBunched_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeBunched)* SIXTRL_RESTRICT sc )
{
    typedef NS(buffer_size_t ) buf_size_t;

    buf_size_t const data_size = NS(SpaceChargeBunched_get_data_size)( sc );
    NS(sc_bunched_real_ptr_t) ptr_data = NS(SpaceChargeBunched_get_data)( sc );

    if( ( data_size > ( NS(buffer_size_t) )0u ) &&
        ( ptr_data != SIXTRL_NULLPTR ) )
    {
        SIXTRL_REAL_T const Z = ( SIXTRL_REAL_T )0;
        SIXTRACKLIB_SET_VALUES( SIXTRL_REAL_T, ptr_data, data_size, Z );
    }

    return;
}

SIXTRL_INLINE NS(sc_bunched_real_const_ptr_t)
NS(SpaceChargeBunched_get_const_data)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const SIXTRL_RESTRICT sc )
{
    SIXTRL_ASSERT( sc != SIXTRL_NULLPTR );
    return sc->data;
}

SIXTRL_INLINE NS(sc_bunched_real_ptr_t)
NS(SpaceChargeBunched_get_data)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeBunched)* SIXTRL_RESTRICT sc )
{
    return ( NS(sc_bunched_real_ptr_t) )NS(SpaceChargeBunched_get_const_data)( sc );
}

SIXTRL_INLINE SIXTRL_UINT64_T NS(SpaceChargeBunched_get_data_size)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const SIXTRL_RESTRICT sc )
{
    SIXTRL_ASSERT( sc != SIXTRL_NULLPTR );
    return sc->size;
}



SIXTRL_INLINE void NS(SpaceChargeBunched_set_data)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeBunched)* SIXTRL_RESTRICT sc,
    SIXTRL_BE_ARGPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_data )
{
    typedef SIXTRL_REAL_T real_t;

    NS(buffer_size_t) const size =
        NS(SpaceChargeBunched_get_data_size)( sc );

    NS(sc_bunched_real_ptr_t) ptr_dest_data =
        NS(SpaceChargeBunched_get_data)( sc );

    SIXTRL_ASSERT( ptr_dest_data != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ptr_data      != SIXTRL_NULLPTR );
    SIXTRACKLIB_COPY_VALUES( real_t, ptr_dest_data, ptr_data, size );

    return;
}

SIXTRL_INLINE void NS(SpaceChargeBunched_set_data_size)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeBunched)* SIXTRL_RESTRICT sc,
    NS(buffer_size_t) const data_size )
{
    SIXTRL_ASSERT( sc != SIXTRL_NULLPTR );
    sc->size = data_size;
    return;
}

SIXTRL_INLINE void NS(SpaceChargeBunched_assign_data_ptr)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeBunched)* SIXTRL_RESTRICT sc,
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT ptr_data )
{
    SIXTRL_ASSERT( sc != SIXTRL_NULLPTR );
    sc->data = ptr_data;
    return;
}

SIXTRL_INLINE int NS(SpaceChargeBunched_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeBunched)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeBunched) *const SIXTRL_RESTRICT source )
{
    int success = -1;

    if( ( destination != SIXTRL_NULLPTR ) && ( source != SIXTRL_NULLPTR ) &&
        ( destination != source ) &&
        ( NS(SpaceChargeBunched_get_const_data)( destination ) != SIXTRL_NULLPTR ) &&
        ( NS(SpaceChargeBunched_get_const_data)( source      ) != SIXTRL_NULLPTR ) &&
        ( NS(SpaceChargeBunched_get_data_size)( destination ) ==
          NS(SpaceChargeBunched_get_data_size)( source ) ) )
    {
        SIXTRL_ASSERT( NS(SpaceChargeBunched_get_const_data)( destination ) !=
                       NS(SpaceChargeBunched_get_const_data)( source ) );

        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
            NS(SpaceChargeBunched_get_data)( destination ),
            NS(SpaceChargeBunched_get_const_data)( source ),
            NS(SpaceChargeBunched_get_data_size)( source ) );

        success = 0;
    }

    return success;
}

/* ************************************************************************* */
/* SpaceChargeCoasting: */

SIXTRL_INLINE NS(buffer_size_t)
NS(SpaceChargeCoasting_get_required_num_dataptrs_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT sc,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t num_dataptrs = ( buf_size_t )0u;

    if( ( slot_size > ( buf_size_t )0u ) &&
        ( sc != SIXTRL_NULLPTR ) &&
        ( NS(SpaceChargeCoasting_get_data_size)( sc ) > ( buf_size_t )0u ) )
    {
        num_dataptrs = ( buf_size_t )1u;
    }

    ( void )buffer;

    return num_dataptrs;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(SpaceChargeCoasting_get_required_num_slots_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT sc,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t num_slots = ( buf_size_t )0u;

    if( ( slot_size > ( buf_size_t )0u ) && ( sc != SIXTRL_NULLPTR ) &&
        ( NS(SpaceChargeCoasting_get_data_size)( sc ) > ( buf_size_t )0u ) )
    {
        num_slots = NS(ManagedBuffer_get_slot_based_length)(
            NS(SpaceChargeCoasting_get_data_size)( sc ) * sizeof( SIXTRL_REAL_T ),
                slot_size );

        num_slots /= slot_size;
    }

    ( void )buffer;

    return num_slots;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeCoasting)*
NS(SpaceChargeCoasting_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeCoasting)* SIXTRL_RESTRICT sc )
{
    if( sc != SIXTRL_NULLPTR )
    {
        sc->size = ( SIXTRL_UINT64_T )0u;
        sc->data   = SIXTRL_NULLPTR;

        NS(SpaceChargeCoasting_clear)( sc );
    }

    return sc;
}

SIXTRL_INLINE void NS(SpaceChargeCoasting_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeCoasting)* SIXTRL_RESTRICT sc )
{
    typedef NS(buffer_size_t ) buf_size_t;

    buf_size_t const data_size = NS(SpaceChargeCoasting_get_data_size)( sc );
    NS(sc_coasting_real_ptr_t) ptr_data = NS(SpaceChargeCoasting_get_data)( sc );

    if( ( data_size > ( NS(buffer_size_t) )0u ) &&
        ( ptr_data != SIXTRL_NULLPTR ) )
    {
        SIXTRL_REAL_T const Z = ( SIXTRL_REAL_T )0;
        SIXTRACKLIB_SET_VALUES( SIXTRL_REAL_T, ptr_data, data_size, Z );
    }

    return;
}

SIXTRL_INLINE NS(sc_coasting_real_const_ptr_t)
NS(SpaceChargeCoasting_get_const_data)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT sc )
{
    SIXTRL_ASSERT( sc != SIXTRL_NULLPTR );
    return sc->data;
}

SIXTRL_INLINE NS(sc_coasting_real_ptr_t)
NS(SpaceChargeCoasting_get_data)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeCoasting)* SIXTRL_RESTRICT sc )
{
    return ( NS(sc_coasting_real_ptr_t) )NS(SpaceChargeCoasting_get_const_data)( sc );
}

SIXTRL_INLINE SIXTRL_UINT64_T NS(SpaceChargeCoasting_get_data_size)(
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT sc )
{
    SIXTRL_ASSERT( sc != SIXTRL_NULLPTR );
    return sc->size;
}



SIXTRL_INLINE void NS(SpaceChargeCoasting_set_data)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeCoasting)* SIXTRL_RESTRICT sc,
    SIXTRL_BE_ARGPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_data )
{
    typedef SIXTRL_REAL_T real_t;

    NS(buffer_size_t) const size =
        NS(SpaceChargeCoasting_get_data_size)( sc );

    NS(sc_coasting_real_ptr_t) ptr_dest_data =
        NS(SpaceChargeCoasting_get_data)( sc );

    SIXTRL_ASSERT( ptr_dest_data != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ptr_data      != SIXTRL_NULLPTR );
    SIXTRACKLIB_COPY_VALUES( real_t, ptr_dest_data, ptr_data, size );

    return;
}

SIXTRL_INLINE void NS(SpaceChargeCoasting_set_data_size)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeCoasting)* SIXTRL_RESTRICT sc,
    NS(buffer_size_t) const data_size )
{
    SIXTRL_ASSERT( sc != SIXTRL_NULLPTR );
    sc->size = data_size;
    return;
}

SIXTRL_INLINE void NS(SpaceChargeCoasting_assign_data_ptr)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeCoasting)* SIXTRL_RESTRICT sc,
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT ptr_data )
{
    SIXTRL_ASSERT( sc != SIXTRL_NULLPTR );
    sc->data = ptr_data;
    return;
}

SIXTRL_INLINE int NS(SpaceChargeCoasting_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(SpaceChargeCoasting)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(SpaceChargeCoasting) *const SIXTRL_RESTRICT source )
{
    int success = -1;

    if( ( destination != SIXTRL_NULLPTR ) && ( source != SIXTRL_NULLPTR ) &&
        ( destination != source ) &&
        ( NS(SpaceChargeCoasting_get_const_data)( destination ) != SIXTRL_NULLPTR ) &&
        ( NS(SpaceChargeCoasting_get_const_data)( source      ) != SIXTRL_NULLPTR ) &&
        ( NS(SpaceChargeCoasting_get_data_size)( destination ) ==
          NS(SpaceChargeCoasting_get_data_size)( source ) ) )
    {
        SIXTRL_ASSERT( NS(SpaceChargeCoasting_get_const_data)( destination ) !=
                       NS(SpaceChargeCoasting_get_const_data)( source ) );

        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
            NS(SpaceChargeCoasting_get_data)( destination ),
            NS(SpaceChargeCoasting_get_const_data)( source ),
            NS(SpaceChargeCoasting_get_data_size)( source ) );

        success = 0;
    }

    return success;
}

/* ************************************************************************* */
/* BeamBeam6D: */

SIXTRL_INLINE NS(buffer_size_t)
NS(BeamBeam6D_get_required_num_dataptrs_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t num_dataptrs = ( buf_size_t )0u;

    if( ( slot_size > ( buf_size_t )0u ) && ( beam_beam != SIXTRL_NULLPTR ) &&
        ( NS(BeamBeam6D_get_data_size)( beam_beam ) > ( buf_size_t )0u ) )
    {
        num_dataptrs = ( buf_size_t )1u;
    }

    ( void )buffer;

    return num_dataptrs;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(BeamBeam6D_get_required_num_slots_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t num_slots = ( buf_size_t )0u;

    if( ( slot_size > ( buf_size_t )0u ) && ( beam_beam != SIXTRL_NULLPTR ) &&
        ( NS(BeamBeam6D_get_data_size)( beam_beam ) > ( buf_size_t )0u ) )
    {
        num_slots = NS(ManagedBuffer_get_slot_based_length)(
            NS(BeamBeam6D_get_data_size)( beam_beam ) * sizeof( SIXTRL_REAL_T ),
                slot_size );

        num_slots /= slot_size;
    }

    ( void )buffer;

    return num_slots;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT beam_beam )
{
    if( beam_beam != SIXTRL_NULLPTR )
    {
        beam_beam->size = ( NS(buffer_size_t) )0u;
        beam_beam->data = SIXTRL_NULLPTR;
    }

    return beam_beam;
}

SIXTRL_INLINE void NS(BeamBeam6D_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT beam_beam )
{
    typedef NS(buffer_size_t ) buf_size_t;

    buf_size_t const data_size = NS(BeamBeam6D_get_data_size)( beam_beam );
    NS(beambeam6d_real_ptr_t) ptr_data = NS(BeamBeam6D_get_data)( beam_beam );

    if( ( data_size > ( NS(buffer_size_t) )0u ) &&
        ( ptr_data != SIXTRL_NULLPTR ) )
    {
        SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO = ( SIXTRL_REAL_T )0u;
        SIXTRACKLIB_SET_VALUES( SIXTRL_REAL_T, ptr_data, data_size, ZERO );
    }

    return;
}

SIXTRL_INLINE NS(beambeam6d_real_const_ptr_t)
NS(BeamBeam6D_get_const_data)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT beam_beam )
{
    SIXTRL_ASSERT( beam_beam != SIXTRL_NULLPTR );
    return beam_beam->data;
}

SIXTRL_INLINE NS(beambeam6d_real_ptr_t)
NS(BeamBeam6D_get_data)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT beam_beam )
{
    return ( NS(beambeam6d_real_ptr_t) )NS(BeamBeam6D_get_const_data)( beam_beam );
}

SIXTRL_INLINE SIXTRL_UINT64_T NS(BeamBeam6D_get_data_size)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT beam_beam )
{
    SIXTRL_ASSERT( beam_beam != SIXTRL_NULLPTR );
    return beam_beam->size;
}

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

SIXTRL_INLINE void NS(BeamBeam6D_set_data)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT beam_beam,
    SIXTRL_BE_ARGPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_data )
{
    typedef SIXTRL_REAL_T real_t;

    NS(buffer_size_t) const size =
        NS(BeamBeam6D_get_data_size)( beam_beam );

    NS(beambeam6d_real_ptr_t) ptr_dest_data =
        NS(BeamBeam6D_get_data)( beam_beam );

    SIXTRL_ASSERT( ptr_dest_data != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ptr_data      != SIXTRL_NULLPTR );
    SIXTRACKLIB_COPY_VALUES( real_t, ptr_dest_data, ptr_data, size );

    return;
}

SIXTRL_INLINE void NS(BeamBeam6D_set_data_size)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const data_size )
{
    SIXTRL_ASSERT( beam_beam != SIXTRL_NULLPTR );
    beam_beam->size = data_size;
    return;
}

SIXTRL_INLINE void NS(BeamBeam6D_assign_data_ptr)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT beam_beam,
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT ptr_data )
{
    SIXTRL_ASSERT( beam_beam != SIXTRL_NULLPTR );
    beam_beam->data = ptr_data;
    return;
}

SIXTRL_INLINE int NS(BeamBeam6D_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam6D)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT source )
{
    int success = -1;

    if( ( destination != SIXTRL_NULLPTR ) && ( source != SIXTRL_NULLPTR ) &&
        ( destination != source ) &&
        ( NS(BeamBeam6D_get_const_data)( destination ) != SIXTRL_NULLPTR ) &&
        ( NS(BeamBeam6D_get_const_data)( source      ) != SIXTRL_NULLPTR ) &&
        ( NS(BeamBeam6D_get_data_size)( destination ) ==
          NS(BeamBeam6D_get_data_size)( source ) ) )
    {
        SIXTRL_ASSERT( NS(BeamBeam6D_get_const_data)( destination ) !=
                       NS(BeamBeam6D_get_const_data)( source ) );

        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
            NS(BeamBeam6D_get_data)( destination ),
            NS(BeamBeam6D_get_const_data)( source ),
            NS(BeamBeam6D_get_data_size)( source ) );

        success = 0;
    }

    return success;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_BE_BEAMFIELDS_BE_BEAMFIELDS_C99_H__ */

/* sixtracklib/common/be_beamfields/be_beamfields.h */
