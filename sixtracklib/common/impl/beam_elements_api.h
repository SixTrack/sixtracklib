#ifndef SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENTS_API_H__
#define SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENTS_API_H__

#if !defined( _GPUCODE )

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/impl/beam_elements_type.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/blocks.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(Drift)* NS(Drift_preset)( NS(Drift)* drift );

SIXTRL_FN SIXTRL_STATIC NS(BlockType) NS(Drift_get_type_id)(
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC NS(block_type_num_t) NS(Drift_get_type_id_num)(
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Drift_get_length)(
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC void NS(Drift_set_length)(
    NS(Drift)* SIXTRL_RESTRICT drift, SIXTRL_REAL_T const length );
    
SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC 
NS(Drift) const* NS(Blocks_get_const_drift)( 
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(Drift)* NS(Blocks_get_drift)( 
    NS(BlockInfo)* SIXTRL_RESTRICT block_info );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(DriftExact)* NS(DriftExact_preset)(
    NS(DriftExact)* drift_exact );

SIXTRL_FN SIXTRL_STATIC NS(BlockType) NS(DriftExact_get_type_id)(
    const NS(DriftExact) *const SIXTRL_RESTRICT drift_exact );

SIXTRL_FN SIXTRL_STATIC NS(block_type_num_t) NS(DriftExact_get_type_id_num)(
    const NS(DriftExact) *const SIXTRL_RESTRICT drift_exact );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(DriftExact_get_length)(
    const NS(DriftExact) *const SIXTRL_RESTRICT drift_exact );

SIXTRL_FN SIXTRL_STATIC void NS(DriftExact_set_length)(
    NS(DriftExact)* SIXTRL_RESTRICT drift_exact, SIXTRL_REAL_T const length );
    
SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(DriftExact) const* 
NS(Blocks_get_const_drift_exact)( 
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC 
NS(DriftExact)* NS(Blocks_get_drift_exact)( 
    NS(BlockInfo)* SIXTRL_RESTRICT block_info );

/* ------------------------------------------------------------------------- */
    
SIXTRL_FN SIXTRL_STATIC NS(MultiPole)* 
NS(MultiPole_preset)( NS(MultiPole)* multipole );

SIXTRL_FN SIXTRL_STATIC NS(BlockType) NS(MultiPole_get_type_id)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_FN SIXTRL_STATIC NS(block_type_num_t) NS(MultiPole_get_type_id_num)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(MultiPole_get_length)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(MultiPole_get_hxl)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(MultiPole_get_hyl)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_FN SIXTRL_STATIC SIXTRL_INT64_T NS(MultiPole_get_order)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(MultiPole_get_const_bal)( const NS(MultiPole) *const SIXTRL_RESTRICT mp );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* 
NS(MultiPole_get_bal)( NS(MultiPole)* SIXTRL_RESTRICT mp );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(MultiPole_get_bal_value)(
    const NS(MultiPole) *const SIXTRL_RESTRICT mp, 
    SIXTRL_INT64_T const index );

SIXTRL_FN SIXTRL_STATIC void NS(MultiPole_set_length)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, SIXTRL_REAL_T const length );

SIXTRL_FN SIXTRL_STATIC void NS(MultiPole_set_hxl)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, SIXTRL_REAL_T const hxl );

SIXTRL_FN SIXTRL_STATIC void NS(MultiPole_set_hyl)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, SIXTRL_REAL_T const hyl );

SIXTRL_FN SIXTRL_STATIC void NS(MultiPole_set_order)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, SIXTRL_INT64_T const order );

SIXTRL_FN SIXTRL_STATIC void NS(MultiPole_set_bal_value)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, 
    SIXTRL_INT64_T const index, SIXTRL_REAL_T const value );

SIXTRL_FN SIXTRL_STATIC void NS(MultiPole_set_bal)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, 
    SIXTRL_INT64_T const order, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT bal );

SIXTRL_FN SIXTRL_STATIC void NS(MultiPole_assign_ptr_to_bal)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT ptr_to_bal );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(MultiPole) const* 
NS(Blocks_get_const_multipole)( 
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC 
NS(MultiPole)* NS(Blocks_get_multipole)( 
    NS(BlockInfo)* SIXTRL_RESTRICT block_info );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(BeamBeamBoostData) const* 
NS(BeamBeam_get_const_ptr_boost_data)( 
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(BeamBeamBoostData)*
NS(BeamBeam_get_ptr_boost_data)( NS(BeamBeam)* SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_set_boost_data)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC const NS(BeamBeamBoostData) *const 
        SIXTRL_RESTRICT ptr_boost_data );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_set_boost_data_value)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    const NS(BeamBeamBoostData) *const SIXTRL_RESTRICT ptr_boost_data );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_assign_boost_data_ptr)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC NS(BeamBeamBoostData)* SIXTRL_RESTRICT ptr_boost_data );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(BeamBeamSigmas) const* 
NS(BeamBeam_get_const_ptr_sigmas_matrix)( 
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(BeamBeamSigmas)*
NS(BeamBeam_get_ptr_sigmas_matrix)( NS(BeamBeam)* SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_set_sigmas_matrix)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC const NS(BeamBeamSigmas) *const 
        SIXTRL_RESTRICT ptr_sigmas );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_set_sigmas_matrix_value)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT ptr_sigmas );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_assign_sigmas_matrix_ptr)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC NS(BeamBeamSigmas)* SIXTRL_RESTRICT ptr_sigmas );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC NS(BeamBeam)* NS(BeamBeam_preset)( 
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(BeamBeam) const* 
NS(Blocks_get_const_beam_beam)( 
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(BeamBeam)* 
NS(Blocks_get_beam_beam)( NS(BlockInfo)* SIXTRL_RESTRICT block_info );

SIXTRL_FN SIXTRL_STATIC NS(BlockType) NS(BeamBeam_get_type_id)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC NS(block_type_num_t) NS(BeamBeam_get_type_id_num)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(BeamBeam_get_const_n_part_per_slice)( 
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* 
NS(BeamBeam_get_n_part_per_slice)( NS(BeamBeam)* SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(BeamBeam_get_n_part_per_slice_value)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam, 
    NS(block_num_elements_t) const index );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_set_n_part_per_slice_value)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam, 
    NS(block_num_elements_t) const index,
    SIXTRL_REAL_T const value );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_set_n_part_per_slice)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* n_part_per_slice_begin );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_assign_n_part_per_slice_ptr)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* n_part_per_slice_begin );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(BeamBeam_get_const_x_slices_star)( 
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* 
NS(BeamBeam_get_x_slices_star)( NS(BeamBeam)* SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(BeamBeam_get_x_slices_star_value)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam, 
    NS(block_num_elements_t) const index );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_set_x_slices_star_value)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam, 
    NS(block_num_elements_t) const index,
    SIXTRL_REAL_T const value );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_set_x_slices_star)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* x_slices_star_begin );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_assign_x_slices_star_ptr)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* x_slices_star_begin );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(BeamBeam_get_const_y_slices_star)( 
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* 
NS(BeamBeam_get_y_slices_star)( NS(BeamBeam)* SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(BeamBeam_get_y_slices_star_value)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam, 
    NS(block_num_elements_t) const index );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_set_y_slices_star_value)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam, 
    NS(block_num_elements_t) const index,
    SIXTRL_REAL_T const value );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_set_y_slices_star)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* y_slices_star_begin );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_assign_y_slices_star_ptr)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* y_slices_star_begin );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */


SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(BeamBeam_get_const_sigma_slices_star)( 
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* 
NS(BeamBeam_get_sigma_slices_star)( NS(BeamBeam)* SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(BeamBeam_get_sigma_slices_star_value)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam, 
    NS(block_num_elements_t) const index );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_set_sigma_slices_star_value)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam, 
    NS(block_num_elements_t) const index,
    SIXTRL_REAL_T const value );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_set_sigma_slices_star)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* sigma_slices_star_begin );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_assign_sigma_slices_star_ptr)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* sigma_slices_star_begin );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC NS(block_num_elements_t) NS(BeamBeam_get_num_of_slices)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_set_num_of_slices)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    NS(block_num_elements_t) const num_of_slices );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(BeamBeam_get_q_part)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_set_q_part)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, SIXTRL_REAL_T const q_part );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(BeamBeam_get_min_sigma_diff)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_set_min_sigma_diff)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_REAL_T const min_sigma_diff );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(BeamBeam_get_treshold_singular)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_set_treshold_singular)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_REAL_T const treshold_singular );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(BeamBeamSigmas)* NS(BeamBeamSigmas_preset)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(BeamBeamSigmas_get_sigma11)( 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT sigma_matrix );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeamSigmas_set_sigma11)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix, 
    SIXTRL_REAL_T const sigma_11 );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(BeamBeamSigmas_get_sigma12)( 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT sigma_matrix );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeamSigmas_set_sigma12)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix, 
    SIXTRL_REAL_T const sigma_12 );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(BeamBeamSigmas_get_sigma13)( 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT sigma_matrix );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeamSigmas_set_sigma13)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix, 
    SIXTRL_REAL_T const sigma_13 );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T  NS(BeamBeamSigmas_get_sigma14)( 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT sigma_matrix );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeamSigmas_set_sigma14)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix, 
    SIXTRL_REAL_T const sigma_14 );


SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T  NS(BeamBeamSigmas_get_sigma22)( 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT sigma_matrix );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeamSigmas_set_sigma22)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix, 
    SIXTRL_REAL_T const sigma_22 );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T  NS(BeamBeamSigmas_get_sigma23)( 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT sigma_matrix );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeamSigmas_set_sigma23)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix, 
    SIXTRL_REAL_T const sigma_23 );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T  NS(BeamBeamSigmas_get_sigma24)( 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT sigma_matrix );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeamSigmas_set_sigma24)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix, 
    SIXTRL_REAL_T const sigma_24 );


SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T  NS(BeamBeamSigmas_get_sigma33)( 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT sigma_matrix );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeamSigmas_set_sigma33)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix, 
    SIXTRL_REAL_T const sigma_33 );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T  NS(BeamBeamSigmas_get_sigma34)( 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT sigma_matrix );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeamSigmas_set_sigma34)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix, 
    SIXTRL_REAL_T const sigma_34 );



SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T  NS(BeamBeamSigmas_get_sigma44)( 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT sigma_matrix );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeamSigmas_set_sigma44)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix, 
    SIXTRL_REAL_T const sigma_44 );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(BeamBeamPropagatedSigmasResult)* 
NS(BeamBeamPropagatedSigmasResult_preset)(
    NS(BeamBeamPropagatedSigmasResult)* SIXTRL_RESTRICT result );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T 
NS(BeamBeamPropagatedSigmasResult_get_cos_theta)( 
    const NS(BeamBeamPropagatedSigmasResult) *const SIXTRL_RESTRICT result );
    
SIXTRL_FN SIXTRL_STATIC void 
NS(BeamBeamPropagatedSigmasResult_set_cos_theta)(
    NS(BeamBeamPropagatedSigmasResult)* SIXTRL_RESTRICT result,
    SIXTRL_REAL_T const cos_theta );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T 
NS(BeamBeamPropagatedSigmasResult_get_sin_theta)( 
    const NS(BeamBeamPropagatedSigmasResult) *const SIXTRL_RESTRICT result );
    
SIXTRL_FN SIXTRL_STATIC void 
NS(BeamBeamPropagatedSigmasResult_set_sin_theta)(
    NS(BeamBeamPropagatedSigmasResult)* SIXTRL_RESTRICT result,
    SIXTRL_REAL_T const sin_theta );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T 
NS(BeamBeamPropagatedSigmasResult_get_sigma_11_hat)( 
    const NS(BeamBeamPropagatedSigmasResult) *const SIXTRL_RESTRICT result );
    
SIXTRL_FN SIXTRL_STATIC void 
NS(BeamBeamPropagatedSigmasResult_set_sigma_11_hat)(
    NS(BeamBeamPropagatedSigmasResult)* SIXTRL_RESTRICT result,
    SIXTRL_REAL_T const sigma_11_hat );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T 
NS(BeamBeamPropagatedSigmasResult_get_sigma_33_hat)( 
    const NS(BeamBeamPropagatedSigmasResult) *const SIXTRL_RESTRICT result );
    
SIXTRL_FN SIXTRL_STATIC void 
NS(BeamBeamPropagatedSigmasResult_set_sigma_33_hat)(
    NS(BeamBeamPropagatedSigmasResult)* SIXTRL_RESTRICT result,
    SIXTRL_REAL_T const sigma_11_hat );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(BeamBeamBoostData)* NS(BeamBeamBoostData_preset)(
    NS(BeamBeamBoostData)* SIXTRL_RESTRICT boost_data );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T 
NS(BeamBeamBoostData_get_sphi)( SIXTRL_GLOBAL_DEC const 
    NS(BeamBeamBoostData) *const SIXTRL_RESTRICT boost_data );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeamBoostData_set_sphi)(
    NS(BeamBeamBoostData)* SIXTRL_RESTRICT boost_data, 
    SIXTRL_REAL_T const sphi );



SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T 
NS(BeamBeamBoostData_get_cphi)( 
    const NS(BeamBeamBoostData) *const SIXTRL_RESTRICT boost_data );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeamBoostData_set_cphi)(
    NS(BeamBeamBoostData)* SIXTRL_RESTRICT boost_data, 
    SIXTRL_REAL_T const cphi );



SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T 
NS(BeamBeamBoostData_get_tphi)( 
    const NS(BeamBeamBoostData) *const SIXTRL_RESTRICT boost_data );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeamBoostData_set_tphi)(
    NS(BeamBeamBoostData)* SIXTRL_RESTRICT boost_data, 
    SIXTRL_REAL_T const tphi );


SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T 
NS(BeamBeamBoostData_get_salpha)( 
    const NS(BeamBeamBoostData) *const SIXTRL_RESTRICT boost_data );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeamBoostData_set_salpha)(
    NS(BeamBeamBoostData)* SIXTRL_RESTRICT boost_data, 
    SIXTRL_REAL_T const salpha );
    

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T 
NS(BeamBeamBoostData_get_calpha)( 
    const NS(BeamBeamBoostData) *const SIXTRL_RESTRICT boost_data );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeamBoostData_set_calpha)(
    NS(BeamBeamBoostData)* SIXTRL_RESTRICT boost_data, 
    SIXTRL_REAL_T const calpha );

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC NS(Cavity)* NS(Cavity_preset)( 
    NS(Cavity)* SIXTRL_RESTRICT cavity );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(Cavity) const* 
NS(Blocks_get_const_cavity)( 
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info );

SIXTRL_FN SIXTRL_STATIC  SIXTRL_GLOBAL_DEC NS(Cavity)* NS(Blocks_get_cavity)( 
    NS(BlockInfo)* SIXTRL_RESTRICT block_info );

SIXTRL_FN SIXTRL_STATIC  NS(BlockType) NS(Cavity_get_type_id)( 
    const NS(Cavity) *const SIXTRL_RESTRICT cavity );

SIXTRL_FN SIXTRL_STATIC  NS(block_type_num_t) NS(Cavity_get_type_id_num)(
    const NS(Cavity) *const SIXTRL_RESTRICT cavity );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Cavity_get_voltage)(
    const NS(Cavity) *const SIXTRL_RESTRICT cavity );

SIXTRL_FN SIXTRL_STATIC void NS(Cavity_set_voltage)(
    NS(Cavity)* SIXTRL_RESTRICT cavity, SIXTRL_REAL_T const voltage );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Cavity_get_frequency)(
    const NS(Cavity) *const SIXTRL_RESTRICT cavity );

SIXTRL_FN SIXTRL_STATIC void NS(Cavity_set_frequency)(
    NS(Cavity)* SIXTRL_RESTRICT cavity, SIXTRL_REAL_T const frequency );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Cavity_get_lag)(
    const NS(Cavity) *const SIXTRL_RESTRICT cavity );

SIXTRL_FN SIXTRL_STATIC void NS(Cavity_set_lag)(
    NS(Cavity)* SIXTRL_RESTRICT cavity, SIXTRL_REAL_T const lag );

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC NS(Align)* NS(Align_preset)( 
    NS(Align)* SIXTRL_RESTRICT align );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(Align) const* 
NS(Blocks_get_const_align)( 
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(Align)* 
NS(Blocks_get_align)( NS(BlockInfo)* SIXTRL_RESTRICT block_info );

SIXTRL_FN SIXTRL_STATIC NS(BlockType) NS(Align_get_type_id)(
    const NS(Align) *const SIXTRL_RESTRICT align );

SIXTRL_FN SIXTRL_STATIC NS(block_type_num_t) NS(Align_get_type_id_num)(
    const NS(Align) *const SIXTRL_RESTRICT align );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Align_get_tilt)(
    const NS(Align) *const SIXTRL_RESTRICT align );

SIXTRL_FN SIXTRL_STATIC void NS(Align_set_tilt)(
    NS(Align)* SIXTRL_RESTRICT align, SIXTRL_REAL_T const tilt );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Align_get_cz)(
    const NS(Align) *const SIXTRL_RESTRICT align );

SIXTRL_FN SIXTRL_STATIC void NS(Align_set_cz)(
    NS(Align)* SIXTRL_RESTRICT align, SIXTRL_REAL_T const cz );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Align_get_cz)(
    const NS(Align) *const SIXTRL_RESTRICT align );

SIXTRL_FN SIXTRL_STATIC void NS(Align_set_sz)(
    NS(Align)* SIXTRL_RESTRICT align, SIXTRL_REAL_T const sz );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Align_get_dx)(
    const NS(Align) *const SIXTRL_RESTRICT align );

SIXTRL_FN SIXTRL_STATIC void NS(Align_set_dx)(
    NS(Align)* SIXTRL_RESTRICT align, SIXTRL_REAL_T const dx );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Align_get_dy)(
    const NS(Align) *const SIXTRL_RESTRICT align );

SIXTRL_FN SIXTRL_STATIC void NS(Align_set_dy)(
    NS(Align)* SIXTRL_RESTRICT align, SIXTRL_REAL_T const dy );

/* ========================================================================= */
/* ======             Implementation of inline functions            ======== */
/* ========================================================================= */

#if !defined( _GPUCODE )
#include "sixtracklib/common/beam_elements.h"
#endif /* !defined( _GPUCODE ) */

SIXTRL_INLINE NS(Drift)* NS(Drift_preset)( NS(Drift)* drift )
{
    NS(Drift_set_length)( drift, ( SIXTRL_REAL_T )0.0 );
    return drift;
}

SIXTRL_INLINE NS(BlockType) NS(Drift_get_type_id)(
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    return ( drift != 0 ) ? NS(BLOCK_TYPE_DRIFT) : NS(BLOCK_TYPE_INVALID);
}

SIXTRL_INLINE NS(block_type_num_t) NS(Drift_get_type_id_num)(
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    return NS(BlockType_to_number)( NS(Drift_get_type_id)( drift ) );
        
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Drift_get_length)(
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    return ( drift ) ? drift->length : ( SIXTRL_REAL_T )0.0;
}

SIXTRL_INLINE void NS(Drift_set_length)(
    NS(Drift)* SIXTRL_RESTRICT drift, SIXTRL_REAL_T const length )
{
    if( ( drift != 0 ) && ( length >= ( SIXTRL_REAL_T )0.0 ) )
    {
        drift->length = length;
    }
    
    return;
}
    
SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(Drift) const* NS(Blocks_get_const_drift)( 
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info )
{
    #if !defined( _GPUCODE )
    
    NS(BlockType) const type_id = 
        NS(BlockInfo_get_type_id)( block_info );
    
    SIXTRL_GLOBAL_DEC void const* ptr_begin = 
        NS(BlockInfo_get_const_ptr_begin)( block_info );    
    
    #else
    
    SIXTRL_GLOBAL_DEC void const* ptr_begin = 0;
    NS(BlockType) type_id = NS(BLOCK_TYPE_INVALID);    
    
    if( block_info != 0 )
    {
        NS(BlockInfo) const info = *block_info;
        ptr_begin = NS(BlockInfo_get_const_ptr_begin)( &info );
        type_id   = NS(BlockInfo_get_type_id)( &info );
    }
    
    #endif /* !defined( _GPUCODE ) */
    
    SIXTRL_ASSERT( ( ptr_begin == 0 ) ||
                   ( ( ( ( uintptr_t )ptr_begin ) % 8u ) == 0u ) );
    
    return ( type_id == NS(BLOCK_TYPE_DRIFT) )
        ? ( SIXTRL_GLOBAL_DEC NS(Drift) const* )ptr_begin : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(Drift)* NS(Blocks_get_drift)( 
    NS(BlockInfo)* SIXTRL_RESTRICT block_info )
{
    return ( SIXTRL_GLOBAL_DEC NS(Drift)* 
        )NS(Blocks_get_const_drift)( block_info );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(DriftExact)* NS(DriftExact_preset)( NS(DriftExact)* drift )
{
    NS(DriftExact_set_length)( drift, ( SIXTRL_REAL_T )0.0 );
    return drift;
}

SIXTRL_INLINE NS(BlockType) NS(DriftExact_get_type_id)(
    const NS(DriftExact) *const SIXTRL_RESTRICT drift )
{
    return ( drift != 0 ) ? NS(BLOCK_TYPE_DRIFT_EXACT) : NS(BLOCK_TYPE_INVALID);
}

SIXTRL_INLINE NS(block_type_num_t) NS(DriftExact_get_type_id_num)(
    const NS(DriftExact) *const SIXTRL_RESTRICT drift )
{
    return NS(BlockType_to_number)( NS(DriftExact_get_type_id)( drift ) );
        
}

SIXTRL_INLINE SIXTRL_REAL_T NS(DriftExact_get_length)(
    const NS(DriftExact) *const SIXTRL_RESTRICT drift )
{
    return ( drift ) ? drift->length : ( SIXTRL_REAL_T )0.0;
}

SIXTRL_INLINE void NS(DriftExact_set_length)(
    NS(DriftExact)* SIXTRL_RESTRICT drift, SIXTRL_REAL_T const length )
{
    if( ( drift != 0 ) && ( length >= ( SIXTRL_REAL_T )0.0 ) )
    {
        drift->length = length;
    }
    
    return;
}
    
SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(DriftExact) const* 
NS(Blocks_get_const_drift_exact)( 
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info )
{
    #if !defined( _GPUCODE )
    
    NS(BlockType) const type_id = 
        NS(BlockInfo_get_type_id)( block_info );
    
    SIXTRL_GLOBAL_DEC void const* ptr_begin = 
        NS(BlockInfo_get_const_ptr_begin)( block_info );    
    
    #else
    
    SIXTRL_GLOBAL_DEC void const* ptr_begin = 0;
    NS(BlockType) type_id = NS(BLOCK_TYPE_INVALID);    
    
    if( block_info != 0 )
    {
        NS(BlockInfo) const info = *block_info;
        ptr_begin = NS(BlockInfo_get_const_ptr_begin)( &info );
        type_id   = NS(BlockInfo_get_type_id)( &info );
    }
    
    #endif /* !defined( _GPUCODE ) */
    
    SIXTRL_ASSERT( ( ptr_begin == 0 ) ||
                   ( ( ( ( uintptr_t )ptr_begin ) % 8u ) == 0u ) );
    
    return ( type_id == NS(BLOCK_TYPE_DRIFT_EXACT) )
        ? ( SIXTRL_GLOBAL_DEC NS(DriftExact) const* )ptr_begin : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(DriftExact)* NS(Blocks_get_drift_exact)( 
    NS(BlockInfo)* SIXTRL_RESTRICT block_info )
{
    return ( SIXTRL_GLOBAL_DEC NS(DriftExact)* 
        )NS(Blocks_get_const_drift_exact)( block_info );
}

/* ------------------------------------------------------------------------- */
    
SIXTRL_INLINE NS(MultiPole)* NS(MultiPole_preset)( NS(MultiPole)* multipole )
{
    NS(MultiPole_set_order)(  multipole, ( SIXTRL_INT64_T )0 );
    NS(MultiPole_set_hxl)(    multipole, ( SIXTRL_REAL_T )0.0 );
    NS(MultiPole_set_hyl)(    multipole, ( SIXTRL_REAL_T )0.0 );
    NS(MultiPole_set_length)( multipole, ( SIXTRL_REAL_T )0.0 );
    NS(MultiPole_assign_ptr_to_bal)( multipole, 0 );
    
    return multipole;
}

SIXTRL_INLINE NS(BlockType) NS(MultiPole_get_type_id)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole )
{
    return NS(BLOCK_TYPE_MULTIPOLE);
}

SIXTRL_INLINE NS(block_type_num_t) NS(MultiPole_get_type_id_num)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole )
{
    return NS(BlockType_to_number)( NS(BLOCK_TYPE_MULTIPOLE) );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(MultiPole_get_length)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole )
{
    SIXTRL_ASSERT(  ( multipole != 0 ) && 
                    ( multipole->length >= ( SIXTRL_REAL_T )0.0 ) );
    
    return multipole->length;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(MultiPole_get_hxl)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole )
{
    SIXTRL_ASSERT(  ( multipole != 0 ) && 
                    ( multipole->hyl > ( SIXTRL_REAL_T )0.0 ) );
    
    return multipole->hxl;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(MultiPole_get_hyl)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole )
{
    SIXTRL_ASSERT(  ( multipole != 0 ) && 
                    ( multipole->hyl > ( SIXTRL_REAL_T )0.0 ) );
    
    return multipole->hyl;
}

SIXTRL_INLINE SIXTRL_INT64_T NS(MultiPole_get_order)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole )
{
    SIXTRL_ASSERT( ( multipole != 0 ) && ( multipole->order >= 0 ) );
    return multipole->order;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(MultiPole_get_const_bal)( const NS(MultiPole) *const SIXTRL_RESTRICT mp )
{
    SIXTRL_ASSERT( mp != 0 );
    return mp->bal;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* 
NS(MultiPole_get_bal)( NS(MultiPole)* SIXTRL_RESTRICT mp )
{
    return ( SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* )NS(MultiPole_get_bal)( mp );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(MultiPole_get_bal_value)(
    const NS(MultiPole) *const SIXTRL_RESTRICT mp, SIXTRL_INT64_T const index )
{
    SIXTRL_ASSERT( ( mp != 0 ) && ( index >= 0 ) && 
                   ( index < ( 2 * mp->order + 2 ) ) );
     
    return ( mp->bal != 0 ) ? mp->bal[ index ] : ( SIXTRL_REAL_T )0.0;
}

SIXTRL_INLINE void NS(MultiPole_set_length)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, SIXTRL_REAL_T const length )
{
    SIXTRL_ASSERT( multipole != 0 );
    multipole->length = length;
    
    return;
}

SIXTRL_INLINE void NS(MultiPole_set_hxl)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, SIXTRL_REAL_T const hxl )
{
    SIXTRL_ASSERT( multipole != 0 );
    multipole->hxl = hxl;
    
    return;
}

SIXTRL_INLINE void NS(MultiPole_set_hyl)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, SIXTRL_REAL_T const hyl )
{
    SIXTRL_ASSERT( multipole != 0 );
    multipole->hyl = hyl;
    
    return;
}

SIXTRL_INLINE void NS(MultiPole_set_order)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, SIXTRL_INT64_T const order )
{
    SIXTRL_ASSERT( ( multipole != 0 ) && ( order >= 0 ) );
    multipole->order = order;
    
    return;
}

SIXTRL_INLINE void NS(MultiPole_set_bal_value)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, 
    SIXTRL_INT64_T const index, SIXTRL_REAL_T const value )
{
    SIXTRL_ASSERT( ( multipole != 0 ) && ( index >= 0 ) &&
                   ( index < ( 2 * multipole->order + 2 ) ) );
    
    multipole->bal[ index ] = value;
    return;
}

SIXTRL_INLINE void NS(MultiPole_set_bal)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, SIXTRL_INT64_T const order, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT bal )
{
    SIXTRL_ASSERT( ( multipole != 0 ) && ( order > 0 ) && ( bal != 0 ) );    
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, multipole->bal, bal, 
                             ( 2 * order + 2 ) );
    
    return;
}

SIXTRL_INLINE void NS(MultiPole_assign_ptr_to_bal)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT ptr_to_bal )
{
    SIXTRL_ASSERT( multipole != 0 );
    multipole->bal = ptr_to_bal;
    
    return;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(MultiPole) const* 
NS(Blocks_get_const_multipole)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info )
{
    #if !defined( _GPUCODE )
    
    NS(BlockType) const type_id = 
        NS(BlockInfo_get_type_id)( block_info );
    
    SIXTRL_GLOBAL_DEC void const* ptr_begin = 
        NS(BlockInfo_get_const_ptr_begin)( block_info );    
    
    #else
    
    SIXTRL_GLOBAL_DEC void const* ptr_begin = 0;
    NS(BlockType) type_id = NS(BLOCK_TYPE_INVALID);    
    
    if( block_info != 0 )
    {
        NS(BlockInfo) const info = *block_info;
        ptr_begin = NS(BlockInfo_get_const_ptr_begin)( &info );
        type_id   = NS(BlockInfo_get_type_id)( &info );
    }
    
    #endif /* !defined( _GPUCODE ) */
    
    SIXTRL_ASSERT( ( ptr_begin == 0 ) ||
                   ( ( ( ( uintptr_t )ptr_begin ) % 8u ) == 0u ) );
    
    return ( type_id == NS(BLOCK_TYPE_MULTIPOLE) )
        ? ( SIXTRL_GLOBAL_DEC NS(MultiPole) const* )ptr_begin : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(MultiPole)*
NS(Blocks_get_multipole)( NS(BlockInfo)* SIXTRL_RESTRICT block_info )
{
    return ( SIXTRL_GLOBAL_DEC NS(MultiPole)* )NS(Blocks_get_const_multipole)( 
        block_info );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(BeamBeamBoostData) const* 
NS(BeamBeam_get_const_ptr_boost_data)( 
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam )
{
    return ( beam_beam != 0 ) ? beam_beam->boost : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(BeamBeamBoostData)*
NS(BeamBeam_get_ptr_boost_data)( NS(BeamBeam)* SIXTRL_RESTRICT beam_beam )
{
    return ( SIXTRL_GLOBAL_DEC NS(BeamBeamBoostData)* 
        )NS(BeamBeam_get_const_ptr_boost_data)( beam_beam );
}

SIXTRL_INLINE void NS(BeamBeam_set_boost_data)( NS(BeamBeam)* SIXTRL_RESTRICT 
    beam_beam, SIXTRL_GLOBAL_DEC const NS(BeamBeamBoostData) *const 
        SIXTRL_RESTRICT ptr_boost_data )
{
    SIXTRL_ASSERT( ( beam_beam != 0 ) && ( beam_beam->boost != 0 ) );
    SIXTRL_ASSERT( ptr_boost_data != 0 );
    
    SIXTRACKLIB_COPY_VALUES( NS(BeamBeamBoostData), beam_beam->boost, 
                             ptr_boost_data, 1u );
    
    return;
}

SIXTRL_INLINE void NS(BeamBeam_set_boost_data_value)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    const NS(BeamBeamBoostData) *const SIXTRL_RESTRICT ptr_boost_data )
{
    SIXTRL_ASSERT( ( beam_beam != 0 ) && ( beam_beam->boost != 0 ) );
    SIXTRL_ASSERT( ptr_boost_data != 0 );
    
    *beam_beam->boost = *ptr_boost_data;
    
    return;
}

SIXTRL_INLINE void NS(BeamBeam_assign_boost_data_ptr)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC NS(BeamBeamBoostData)* SIXTRL_RESTRICT ptr_boost_data )
{
    SIXTRL_ASSERT( beam_beam != 0 );
    beam_beam->boost = ptr_boost_data;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(BeamBeamSigmas) const* 
NS(BeamBeam_get_const_ptr_sigmas_matrix)( 
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam )
{
    return ( beam_beam != 0 ) ? beam_beam->sigmas : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(BeamBeamSigmas)*
NS(BeamBeam_get_ptr_sigmas_matrix)( NS(BeamBeam)* SIXTRL_RESTRICT beam_beam )
{
    return ( SIXTRL_GLOBAL_DEC NS(BeamBeamSigmas)* 
        )NS(BeamBeam_get_const_ptr_sigmas_matrix)( beam_beam );
}

SIXTRL_INLINE void NS(BeamBeam_set_sigmas_matrix)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC const NS(BeamBeamSigmas) *const 
        SIXTRL_RESTRICT ptr_sigmas )
{
    SIXTRL_ASSERT( ( beam_beam  != 0 ) && ( beam_beam->sigmas != 0 ) );
    SIXTRL_ASSERT(   ptr_sigmas != 0 );
    
    SIXTRACKLIB_COPY_VALUES( NS(BeamBeamSigmas), beam_beam->sigmas, 
                             ptr_sigmas, 1u );
    
    return;
}

SIXTRL_INLINE void NS(BeamBeam_set_sigmas_matrix_value)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT ptr_sigmas )
{
    SIXTRL_ASSERT( ( beam_beam != 0 ) && ( beam_beam->sigmas != 0 ) );
    SIXTRL_ASSERT(  ptr_sigmas != 0 );
    
    *beam_beam->sigmas = *ptr_sigmas;    
    return;
}

SIXTRL_INLINE void NS(BeamBeam_assign_sigmas_matrix_ptr)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC NS(BeamBeamSigmas)* SIXTRL_RESTRICT ptr_sigmas )
{
    SIXTRL_ASSERT( beam_beam != 0 );
    beam_beam->sigmas = ptr_sigmas;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(BeamBeam)* NS(BeamBeam_preset)( 
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam )
{
    if( beam_beam != 0 )
    {
        SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO = ( SIXTRL_REAL_T )0.0;
        
        beam_beam->boost             = 0;
        beam_beam->sigmas            = 0;
        beam_beam->n_part_per_slice  = 0;
        beam_beam->x_slices_star     = 0;
        beam_beam->y_slices_star     = 0;
        beam_beam->sigma_slices_star = 0;
        
        beam_beam->num_of_slices     = ( NS(block_num_elements_t) )0u;
        beam_beam->q_part            = ZERO;
        beam_beam->min_sigma_diff    = ZERO;
        beam_beam->treshold_sing     = ZERO;        
    }
    
    return beam_beam;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(BeamBeam) const* 
NS(Blocks_get_const_beam_beam)( 
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info )
{
    #if !defined( _GPUCODE )
    
    NS(BlockType) const type_id = 
        NS(BlockInfo_get_type_id)( block_info );
    
    SIXTRL_GLOBAL_DEC void const* ptr_begin = 
        NS(BlockInfo_get_const_ptr_begin)( block_info );    
    
    #else
    
    SIXTRL_GLOBAL_DEC void const* ptr_begin = 0;
    NS(BlockType) type_id = NS(BLOCK_TYPE_INVALID);    
    
    if( block_info != 0 )
    {
        NS(BlockInfo) const info = *block_info;
        ptr_begin = NS(BlockInfo_get_const_ptr_begin)( &info );
        type_id   = NS(BlockInfo_get_type_id)( &info );
    }
    
    #endif /* !defined( _GPUCODE ) */
    
    SIXTRL_ASSERT( ( ptr_begin == 0 ) ||
                   ( ( ( ( uintptr_t )ptr_begin ) % 8u ) == 0u ) );
    
    return ( type_id == NS(BLOCK_TYPE_BEAM_BEAM) )
        ? ( SIXTRL_GLOBAL_DEC NS(BeamBeam) const* )ptr_begin : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(BeamBeam)* 
NS(Blocks_get_beam_beam)( NS(BlockInfo)* SIXTRL_RESTRICT block_info )
{
    return ( SIXTRL_GLOBAL_DEC NS(BeamBeam)* )NS(Blocks_get_const_beam_beam)(
        block_info );
}

SIXTRL_INLINE NS(BlockType) NS(BeamBeam_get_type_id)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam )
{
    return ( beam_beam != 0 ) ? NS(BLOCK_TYPE_BEAM_BEAM) 
                              : NS(BLOCK_TYPE_INVALID);
}

SIXTRL_INLINE NS(block_type_num_t) NS(BeamBeam_get_type_id_num)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam )
{
    return NS(BlockType_to_number)( NS(BeamBeam_get_type_id)( beam_beam ) );
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(BeamBeam_get_const_n_part_per_slice)( 
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam )
{
    return ( beam_beam != 0 ) ? beam_beam->n_part_per_slice : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* 
NS(BeamBeam_get_n_part_per_slice)( NS(BeamBeam)* SIXTRL_RESTRICT beam_beam )
{
    return ( SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* 
        )NS(BeamBeam_get_const_n_part_per_slice)( beam_beam );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(BeamBeam_get_n_part_per_slice_value)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam, 
    NS(block_num_elements_t) const index )
{
    SIXTRL_ASSERT( ( beam_beam != 0 ) && 
                   ( beam_beam->n_part_per_slice != 0 ) &&
                   ( beam_beam->num_of_slices > index ) );
    
    return beam_beam->n_part_per_slice[ index ];
}

SIXTRL_INLINE void NS(BeamBeam_set_n_part_per_slice_value)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam, 
    NS(block_num_elements_t) const index,
    SIXTRL_REAL_T const value )
{
    SIXTRL_ASSERT( ( beam_beam != 0 ) && 
                   ( beam_beam->n_part_per_slice != 0 ) &&
                   ( beam_beam->num_of_slices > index ) );
    
    beam_beam->n_part_per_slice[ index ] = value;
    
    return;
}

SIXTRL_INLINE void NS(BeamBeam_set_n_part_per_slice)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* n_part_per_slice_begin )
{
    SIXTRL_ASSERT( ( beam_beam != 0 ) && 
                   ( beam_beam->n_part_per_slice != 0 ) &&
                   ( beam_beam->num_of_slices > 0 ) );
    
    SIXTRL_ASSERT( n_part_per_slice_begin != 0 );
    
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, beam_beam->n_part_per_slice, 
                             n_part_per_slice_begin, 
                             beam_beam->num_of_slices );
    
    return;
}

SIXTRL_INLINE void NS(BeamBeam_assign_n_part_per_slice_ptr)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* n_part_per_slice_begin )
{
    SIXTRL_ASSERT( beam_beam != 0 );
    beam_beam->n_part_per_slice = n_part_per_slice_begin;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(BeamBeam_get_const_x_slices_star)( 
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam )
{
    return ( beam_beam != 0 ) ? beam_beam->x_slices_star : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* 
NS(BeamBeam_get_x_slices_star)( NS(BeamBeam)* SIXTRL_RESTRICT beam_beam )
{
    return ( SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* 
        )NS(BeamBeam_get_const_x_slices_star)( beam_beam );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(BeamBeam_get_x_slices_star_value)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam, 
    NS(block_num_elements_t) const index )
{
    SIXTRL_ASSERT( ( beam_beam != 0 ) && ( beam_beam->x_slices_star != 0 ) &&
                   ( beam_beam->num_of_slices > index ) );
    
    return beam_beam->x_slices_star[ index ];
}

SIXTRL_INLINE void NS(BeamBeam_set_x_slices_star_value)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam, 
    NS(block_num_elements_t) const index,
    SIXTRL_REAL_T const value )
{
    SIXTRL_ASSERT( ( beam_beam != 0 ) && ( beam_beam->x_slices_star != 0 ) &&
                   ( beam_beam->num_of_slices > index ) );
    
    beam_beam->x_slices_star[ index ] = value;
    return;
}

SIXTRL_INLINE void NS(BeamBeam_set_x_slices_star)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* x_slices_star_begin )
{
    SIXTRL_ASSERT( ( beam_beam != 0 ) && ( beam_beam->x_slices_star != 0 ) &&
                   ( beam_beam->num_of_slices > 0 ) );
    
    SIXTRL_ASSERT( x_slices_star_begin != 0 );
    
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, beam_beam->x_slices_star, 
                             x_slices_star_begin, beam_beam->num_of_slices );
    
    return;
}

SIXTRL_INLINE void NS(BeamBeam_assign_x_slices_star_ptr)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* x_slices_star_begin )
{
    SIXTRL_ASSERT( beam_beam != 0 );
    beam_beam->x_slices_star = x_slices_star_begin;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(BeamBeam_get_const_y_slices_star)( 
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam )
{
    return ( beam_beam != 0 ) ? beam_beam->y_slices_star : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* 
NS(BeamBeam_get_y_slices_star)( NS(BeamBeam)* SIXTRL_RESTRICT beam_beam )
{
    return ( SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* 
        )NS(BeamBeam_get_const_y_slices_star)( beam_beam );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(BeamBeam_get_y_slices_star_value)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam, 
    NS(block_num_elements_t) const index )
{
    SIXTRL_ASSERT( ( beam_beam != 0 ) && ( beam_beam->y_slices_star != 0 ) &&
                   ( beam_beam->num_of_slices > index ) );
    
    return beam_beam->y_slices_star[ index ];
}

SIXTRL_INLINE void NS(BeamBeam_set_y_slices_star_value)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam, 
    NS(block_num_elements_t) const index,
    SIXTRL_REAL_T const value )
{
    SIXTRL_ASSERT( ( beam_beam != 0 ) && ( beam_beam->y_slices_star != 0 ) &&
                   ( beam_beam->num_of_slices > index ) );
    
    beam_beam->y_slices_star[ index ] = value;
    return;
}

SIXTRL_INLINE void NS(BeamBeam_set_y_slices_star)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* y_slices_star_begin )
{
    SIXTRL_ASSERT( ( beam_beam != 0 ) && ( beam_beam->y_slices_star != 0 ) &&
                   ( beam_beam->num_of_slices > 0 ) );
    
    SIXTRL_ASSERT( y_slices_star_begin != 0 );
    
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, beam_beam->y_slices_star, 
                             y_slices_star_begin, beam_beam->num_of_slices );
    
    return;
}

SIXTRL_INLINE void NS(BeamBeam_assign_y_slices_star_ptr)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* y_slices_star_begin )
{
    SIXTRL_ASSERT( beam_beam != 0 );
    beam_beam->y_slices_star = y_slices_star_begin;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */


SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(BeamBeam_get_const_sigma_slices_star)( 
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam )
{
    return ( beam_beam != 0 ) ? beam_beam->sigma_slices_star : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* 
NS(BeamBeam_get_sigma_slices_star)( NS(BeamBeam)* SIXTRL_RESTRICT beam_beam )
{
    return ( SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* 
        )NS(BeamBeam_get_const_sigma_slices_star)( beam_beam );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(BeamBeam_get_sigma_slices_star_value)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam, 
    NS(block_num_elements_t) const index )
{
    SIXTRL_ASSERT( ( beam_beam != 0 ) && 
                   ( beam_beam->sigma_slices_star != 0 ) &&
                   ( beam_beam->num_of_slices > index ) );
    
    return beam_beam->sigma_slices_star[ index ];
}

SIXTRL_INLINE void NS(BeamBeam_set_sigma_slices_star_value)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam, 
    NS(block_num_elements_t) const index,
    SIXTRL_REAL_T const value )
{
    SIXTRL_ASSERT( ( beam_beam != 0 ) && 
                   ( beam_beam->sigma_slices_star != 0 ) &&
                   ( beam_beam->num_of_slices > index ) );
    
    beam_beam->sigma_slices_star[ index ] = value;
    return;
}

SIXTRL_INLINE void NS(BeamBeam_set_sigma_slices_star)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* sigma_slices_star_begin )
{
    SIXTRL_ASSERT( ( beam_beam != 0 ) && ( beam_beam->sigma_slices_star != 0 ) &&
                   ( beam_beam->num_of_slices > 0 ) );
    
    SIXTRL_ASSERT( sigma_slices_star_begin != 0 );
    
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, beam_beam->sigma_slices_star, 
                             sigma_slices_star_begin, 
                             beam_beam->num_of_slices );
    
    return;
}

SIXTRL_INLINE void NS(BeamBeam_assign_sigma_slices_star_ptr)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* sigma_slices_star_begin )
{
    SIXTRL_ASSERT( beam_beam != 0 );
    beam_beam->sigma_slices_star = sigma_slices_star_begin;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(block_num_elements_t) NS(BeamBeam_get_num_of_slices)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam )
{
    return ( beam_beam != 0 ) 
        ? beam_beam->num_of_slices : ( NS(block_num_elements_t) )0u;    
}

SIXTRL_INLINE void NS(BeamBeam_set_num_of_slices)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    NS(block_num_elements_t) const num_of_slices )
{
    SIXTRL_ASSERT( beam_beam != 0 );    
    beam_beam->num_of_slices = num_of_slices;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_REAL_T NS(BeamBeam_get_q_part)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam )
{
    SIXTRL_ASSERT( beam_beam != 0 );
    return beam_beam->q_part;
}

SIXTRL_INLINE void NS(BeamBeam_set_q_part)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, SIXTRL_REAL_T const q_part )
{
    SIXTRL_ASSERT( beam_beam != 0 );
    beam_beam->q_part = q_part;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_REAL_T NS(BeamBeam_get_min_sigma_diff)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam )
{
    SIXTRL_ASSERT( beam_beam != 0 );
    return beam_beam->min_sigma_diff;
}

SIXTRL_INLINE void NS(BeamBeam_set_min_sigma_diff)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_REAL_T const min_sigma_diff )
{
    SIXTRL_ASSERT( beam_beam != 0 );
    beam_beam->min_sigma_diff = min_sigma_diff;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_REAL_T NS(BeamBeam_get_treshold_singular)(
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam )
{
    SIXTRL_ASSERT( beam_beam != 0 );
    return beam_beam->treshold_sing;
}

SIXTRL_INLINE void NS(BeamBeam_set_treshold_singular)(
    NS(BeamBeam)* SIXTRL_RESTRICT beam_beam, 
    SIXTRL_REAL_T const treshold_singular )
{
    SIXTRL_ASSERT( beam_beam != 0 );
    beam_beam->treshold_sing = treshold_singular;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(BeamBeamSigmas)* NS(BeamBeamSigmas_preset)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix )
{
    if( sigma_matrix != 0 )
    {
        SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO = ( SIXTRL_REAL_T )0.0L;
        
        sigma_matrix->sigma_11 = ZERO;
        sigma_matrix->sigma_12 = ZERO;
        sigma_matrix->sigma_13 = ZERO;
        sigma_matrix->sigma_14 = ZERO;
        
        sigma_matrix->sigma_22 = ZERO;
        sigma_matrix->sigma_23 = ZERO;
        sigma_matrix->sigma_24 = ZERO;
        
        sigma_matrix->sigma_33 = ZERO;
        sigma_matrix->sigma_34 = ZERO;
        
        sigma_matrix->sigma_44 = ZERO;
    }
    
    return sigma_matrix;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(BeamBeamSigmas_get_sigma11)( 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT sigma_matrix )
{
    SIXTRL_ASSERT( sigma_matrix != 0 );
    return sigma_matrix->sigma_11;
}

SIXTRL_INLINE void NS(BeamBeamSigmas_set_sigma11)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix, 
    SIXTRL_REAL_T const sigma_11 )
{
    SIXTRL_ASSERT( sigma_matrix != 0 );
    sigma_matrix->sigma_11 = sigma_11;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(BeamBeamSigmas_get_sigma12)( 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT sigma_matrix )
{
    SIXTRL_ASSERT( sigma_matrix != 0 );
    return sigma_matrix->sigma_12;
}

SIXTRL_INLINE void NS(BeamBeamSigmas_set_sigma12)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix, 
    SIXTRL_REAL_T const sigma_12 )
{
    SIXTRL_ASSERT( sigma_matrix != 0 );
    sigma_matrix->sigma_12 = sigma_12;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(BeamBeamSigmas_get_sigma13)( 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT sigma_matrix )
{
    SIXTRL_ASSERT( sigma_matrix != 0 );
    return sigma_matrix->sigma_13;
}

SIXTRL_INLINE void NS(BeamBeamSigmas_set_sigma13)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix, 
    SIXTRL_REAL_T const sigma_13 )
{
    SIXTRL_ASSERT( sigma_matrix != 0 );
    sigma_matrix->sigma_13 = sigma_13;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(BeamBeamSigmas_get_sigma14)( 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT sigma_matrix )
{
    SIXTRL_ASSERT( sigma_matrix != 0 );
    return sigma_matrix->sigma_14;
}

SIXTRL_INLINE void NS(BeamBeamSigmas_set_sigma14)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix, 
    SIXTRL_REAL_T const sigma_14 )
{
    SIXTRL_ASSERT( sigma_matrix != 0 );
    sigma_matrix->sigma_14 = sigma_14;
}


SIXTRL_INLINE SIXTRL_REAL_T NS(BeamBeamSigmas_get_sigma22)( 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT sigma_matrix )
{
    SIXTRL_ASSERT( sigma_matrix != 0 );
    return sigma_matrix->sigma_22;
}

SIXTRL_INLINE void  NS(BeamBeamSigmas_set_sigma22)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix, 
    SIXTRL_REAL_T const sigma_22 )
{
    SIXTRL_ASSERT( sigma_matrix != 0 );
    sigma_matrix->sigma_22 = sigma_22;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(BeamBeamSigmas_get_sigma23)( 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT sigma_matrix )
{
    SIXTRL_ASSERT( sigma_matrix != 0 );
    return sigma_matrix->sigma_23;
}

SIXTRL_INLINE void NS(BeamBeamSigmas_set_sigma23)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix, 
    SIXTRL_REAL_T const sigma_23 )
{
    SIXTRL_ASSERT( sigma_matrix != 0 );
    sigma_matrix->sigma_23 = sigma_23;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(BeamBeamSigmas_get_sigma24)( 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT sigma_matrix )
{
    SIXTRL_ASSERT( sigma_matrix != 0 );
    return sigma_matrix->sigma_24;
}

SIXTRL_INLINE void NS(BeamBeamSigmas_set_sigma24)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix, 
    SIXTRL_REAL_T const sigma_24 )
{
    SIXTRL_ASSERT( sigma_matrix != 0 );
    sigma_matrix->sigma_24 = sigma_24;
}


SIXTRL_INLINE SIXTRL_REAL_T NS(BeamBeamSigmas_get_sigma33)( 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT sigma_matrix )
{
    SIXTRL_ASSERT( sigma_matrix != 0 );
    return sigma_matrix->sigma_33;
}

SIXTRL_INLINE void NS(BeamBeamSigmas_set_sigma33)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix, 
    SIXTRL_REAL_T const sigma_33 )
{
    SIXTRL_ASSERT( sigma_matrix != 0 );
    sigma_matrix->sigma_33 = sigma_33;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(BeamBeamSigmas_get_sigma34)( 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT sigma_matrix )
{
    SIXTRL_ASSERT( sigma_matrix != 0 );
    return sigma_matrix->sigma_34;
}

SIXTRL_INLINE void NS(BeamBeamSigmas_set_sigma34)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix, 
    SIXTRL_REAL_T const sigma_34 )
{
    SIXTRL_ASSERT( sigma_matrix != 0 );
    sigma_matrix->sigma_34 = sigma_34;
}


SIXTRL_INLINE SIXTRL_REAL_T NS(BeamBeamSigmas_get_sigma44)( 
    const NS(BeamBeamSigmas) *const SIXTRL_RESTRICT sigma_matrix )
{
    SIXTRL_ASSERT( sigma_matrix != 0 );
    return sigma_matrix->sigma_44;
}

SIXTRL_INLINE void NS(BeamBeamSigmas_set_sigma44)( 
    NS(BeamBeamSigmas)* SIXTRL_RESTRICT sigma_matrix, 
    SIXTRL_REAL_T const sigma_44 )
{
    SIXTRL_ASSERT( sigma_matrix != 0 );
    sigma_matrix->sigma_44 = sigma_44;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(BeamBeamPropagatedSigmasResult)* 
NS(BeamBeamPropagatedSigmasResult_preset)(
    NS(BeamBeamPropagatedSigmasResult)* SIXTRL_RESTRICT result )
{
    if( result != 0 )
    {
        SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO = ( SIXTRL_REAL_T )0.0L;
        
        result->cos_theta    = ZERO;
        result->sin_theta    = ZERO;
        result->sigma_11_hat = ZERO;
        result->sigma_33_hat = ZERO;
    }
    
    return result;
}

SIXTRL_INLINE SIXTRL_REAL_T 
NS(BeamBeamPropagatedSigmasResult_get_cos_theta)( 
    const NS(BeamBeamPropagatedSigmasResult) *const SIXTRL_RESTRICT result )
{
    SIXTRL_ASSERT( result != 0 );
    return result->cos_theta;
}
    
SIXTRL_INLINE void 
NS(BeamBeamPropagatedSigmasResult_set_cos_theta)(
    NS(BeamBeamPropagatedSigmasResult)* SIXTRL_RESTRICT result,
    SIXTRL_REAL_T const cos_theta )
{
    SIXTRL_ASSERT( result != 0 );
    result->cos_theta = cos_theta;
    return;
}

SIXTRL_INLINE SIXTRL_REAL_T 
NS(BeamBeamPropagatedSigmasResult_get_sin_theta)( 
    const NS(BeamBeamPropagatedSigmasResult) *const SIXTRL_RESTRICT result )
{
    SIXTRL_ASSERT( result != 0 );
    return result->sin_theta;
}
    
SIXTRL_INLINE void 
NS(BeamBeamPropagatedSigmasResult_set_sin_theta)(
    NS(BeamBeamPropagatedSigmasResult)* SIXTRL_RESTRICT result,
    SIXTRL_REAL_T const sin_theta )
{
    SIXTRL_ASSERT( result != 0 );
    result->sin_theta = sin_theta;
    return;
}

SIXTRL_INLINE SIXTRL_REAL_T 
NS(BeamBeamPropagatedSigmasResult_get_sigma_11_hat)( 
    const NS(BeamBeamPropagatedSigmasResult) *const SIXTRL_RESTRICT result )
{
    SIXTRL_ASSERT( result != 0 );
    return result->sigma_11_hat;
}
    
SIXTRL_INLINE void 
NS(BeamBeamPropagatedSigmasResult_set_sigma_11_hat)(
    NS(BeamBeamPropagatedSigmasResult)* SIXTRL_RESTRICT result,
    SIXTRL_REAL_T const sigma_11_hat )
{
    SIXTRL_ASSERT( result != 0 );
    result->sigma_11_hat = sigma_11_hat;
    return;
}

SIXTRL_INLINE SIXTRL_REAL_T 
NS(BeamBeamPropagatedSigmasResult_get_sigma_33_hat)( 
    const NS(BeamBeamPropagatedSigmasResult) *const SIXTRL_RESTRICT result )
{
    SIXTRL_ASSERT( result != 0 );
    return result->sigma_33_hat;
}
    
SIXTRL_INLINE void 
NS(BeamBeamPropagatedSigmasResult_set_sigma_33_hat)(
    NS(BeamBeamPropagatedSigmasResult)* SIXTRL_RESTRICT result,
    SIXTRL_REAL_T const sigma_33_hat )
{
    SIXTRL_ASSERT( result != 0 );
    result->sigma_33_hat = sigma_33_hat;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(BeamBeamBoostData)* NS(BeamBeamBoostData_preset)(
    NS(BeamBeamBoostData)* SIXTRL_RESTRICT boost_data )
{
    if( boost_data != 0 )
    {
        SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO = ( SIXTRL_REAL_T )0.0L;
        boost_data->sphi   = ZERO;
        boost_data->cphi   = ZERO;
        boost_data->tphi   = ZERO;
        boost_data->salpha = ZERO;
        boost_data->calpha = ZERO;
    }
    
    return boost_data;
}

SIXTRL_INLINE SIXTRL_REAL_T 
NS(BeamBeamBoostData_get_sphi)( SIXTRL_GLOBAL_DEC const 
    NS(BeamBeamBoostData) *const SIXTRL_RESTRICT boost_data )
{
    SIXTRL_ASSERT( boost_data != 0 );
    return boost_data->sphi;
}

SIXTRL_INLINE void NS(BeamBeamBoostData_set_sphi)(
    NS(BeamBeamBoostData)* SIXTRL_RESTRICT boost_data, 
    SIXTRL_REAL_T const sphi )
{
    SIXTRL_ASSERT( boost_data != 0 );
    boost_data->sphi = sphi;
    return;
}



SIXTRL_INLINE SIXTRL_REAL_T 
NS(BeamBeamBoostData_get_cphi)( 
    const NS(BeamBeamBoostData) *const SIXTRL_RESTRICT boost_data )
{
    SIXTRL_ASSERT( boost_data != 0 );
    return boost_data->cphi;
}

SIXTRL_INLINE void NS(BeamBeamBoostData_set_cphi)(
    NS(BeamBeamBoostData)* SIXTRL_RESTRICT boost_data, 
    SIXTRL_REAL_T const cphi )
{
    SIXTRL_ASSERT( boost_data != 0 );
    boost_data->cphi = cphi;
    return;
}


SIXTRL_INLINE SIXTRL_REAL_T 
NS(BeamBeamBoostData_get_tphi)( 
    const NS(BeamBeamBoostData) *const SIXTRL_RESTRICT boost_data )
{
    SIXTRL_ASSERT( boost_data != 0 );
    return boost_data->tphi;
}

SIXTRL_INLINE void NS(BeamBeamBoostData_set_tphi)(
    NS(BeamBeamBoostData)* SIXTRL_RESTRICT boost_data, 
    SIXTRL_REAL_T const tphi )
{
    SIXTRL_ASSERT( boost_data != 0 );
    boost_data->tphi = tphi;
    return;
}


SIXTRL_INLINE SIXTRL_REAL_T 
NS(BeamBeamBoostData_get_salpha)( 
    const NS(BeamBeamBoostData) *const SIXTRL_RESTRICT boost_data )
{
    SIXTRL_ASSERT( boost_data != 0 );
    return boost_data->salpha;
}

SIXTRL_INLINE void NS(BeamBeamBoostData_set_salpha)(
    NS(BeamBeamBoostData)* SIXTRL_RESTRICT boost_data, 
    SIXTRL_REAL_T const salpha )
{
    SIXTRL_ASSERT( boost_data != 0 );
    boost_data->salpha = salpha;
    return;
}
    

SIXTRL_INLINE SIXTRL_REAL_T 
NS(BeamBeamBoostData_get_calpha)( 
    const NS(BeamBeamBoostData) *const SIXTRL_RESTRICT boost_data )
{
    SIXTRL_ASSERT( boost_data != 0 );
    return boost_data->calpha;
}

SIXTRL_INLINE void NS(BeamBeamBoostData_set_calpha)(
    NS(BeamBeamBoostData)* SIXTRL_RESTRICT boost_data, 
    SIXTRL_REAL_T const calpha )
{
    SIXTRL_ASSERT( boost_data != 0 );
    boost_data->calpha = calpha;
    return;
}

/* ========================================================================= */

SIXTRL_INLINE NS(Cavity)* NS(Cavity_preset)( 
    NS(Cavity)* SIXTRL_RESTRICT cavity )
{
    if( cavity != 0 )
    {
        SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO = ( SIXTRL_REAL_T )0.0L;
        
        cavity->voltage   = ZERO;
        cavity->frequency = ZERO;
        cavity->lag       = ZERO;
    }
    
    return cavity;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(Cavity) const* NS(Blocks_get_const_cavity)( 
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info )
{
    return ( NS(BlockInfo_get_type_id)( block_info ) == NS(BLOCK_TYPE_CAVITY) )
        ? ( SIXTRL_GLOBAL_DEC NS(Cavity) const* 
            )NS(BlockInfo_get_const_ptr_begin)( block_info )
        : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(Cavity)* NS(Blocks_get_cavity)( 
    NS(BlockInfo)* SIXTRL_RESTRICT block_info )
{
    return ( SIXTRL_GLOBAL_DEC NS(Cavity)* 
        )NS(Blocks_get_const_align)( block_info );
}

SIXTRL_INLINE NS(BlockType) NS(Cavity_get_type_id)( 
    const NS(Cavity) *const SIXTRL_RESTRICT cavity )
{
    return ( cavity != 0 ) ? NS(BLOCK_TYPE_ALIGN) : NS(BLOCK_TYPE_INVALID);
}

SIXTRL_INLINE NS(block_type_num_t) NS(Cavity_get_type_id_num)(
    const NS(Cavity) *const SIXTRL_RESTRICT cavity )
{
    return NS(BlockType_to_number)( NS(Cavity_get_type_id)( cavity ) );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Cavity_get_voltage)(
    const NS(Cavity) *const SIXTRL_RESTRICT cavity )
{
    SIXTRL_ASSERT( cavity != 0 );
    return cavity->voltage;
}

SIXTRL_INLINE void NS(Cavity_set_voltage)(
    NS(Cavity)* SIXTRL_RESTRICT cavity, SIXTRL_REAL_T const voltage )
{
    SIXTRL_ASSERT( cavity != 0 );
    cavity->voltage = voltage;
    return;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Cavity_get_frequency)(
    const NS(Cavity) *const SIXTRL_RESTRICT cavity )
{
    SIXTRL_ASSERT( cavity != 0 );
    return cavity->frequency;
}

SIXTRL_INLINE void NS(Cavity_set_frequency)(
    NS(Cavity)* SIXTRL_RESTRICT cavity, SIXTRL_REAL_T const frequency )
{
    SIXTRL_ASSERT( cavity != 0 );
    cavity->frequency = frequency;
    return;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Cavity_get_lag)(
    const NS(Cavity) *const SIXTRL_RESTRICT cavity )
{
    SIXTRL_ASSERT( cavity != 0 );
    return cavity->lag;
}

SIXTRL_INLINE void NS(Cavity_set_lag)( 
    NS(Cavity)* SIXTRL_RESTRICT cavity, SIXTRL_REAL_T const lag )
{
    SIXTRL_ASSERT( cavity != 0 );
    cavity->lag = lag;
    return;
}

/* ========================================================================= */

SIXTRL_INLINE NS(Align)* NS(Align_preset)( NS(Align)* SIXTRL_RESTRICT align )
{
    if( align != 0 )
    {
        SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO = ( SIXTRL_REAL_T )0.0L;
        
        align->tilt     = ZERO;
        align->cz       = ZERO;
        align->sz       = ZERO;
        align->dx       = ZERO;
        align->dy       = ZERO;
    }
    
    return align;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(Align) const* NS(Blocks_get_const_align)( 
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info )
{
    return ( NS(BlockInfo_get_type_id)( block_info ) == NS(BLOCK_TYPE_ALIGN ) )
        ? ( SIXTRL_GLOBAL_DEC NS(Align) const* 
            )NS(BlockInfo_get_const_ptr_begin)( block_info )
        : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(Align)* NS(Blocks_get_align)( 
    NS(BlockInfo)* SIXTRL_RESTRICT block_info )
{
    return ( SIXTRL_GLOBAL_DEC NS(Align)* 
        )NS(Blocks_get_const_align)( block_info );
}

SIXTRL_INLINE NS(BlockType) NS(Align_get_type_id)( 
    const NS(Align) *const SIXTRL_RESTRICT align )
{
    return ( align != 0 ) ? NS(BLOCK_TYPE_ALIGN) : NS(BLOCK_TYPE_INVALID);
}

SIXTRL_INLINE NS(block_type_num_t) NS(Align_get_type_id_num)(
    const NS(Align) *const SIXTRL_RESTRICT align )
{
    return NS(BlockType_to_number)( NS(Align_get_type_id)( align ) );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Align_get_tilt)(
    const NS(Align) *const SIXTRL_RESTRICT align )
{
    SIXTRL_ASSERT( align != 0 );
    return align->tilt;
}

SIXTRL_INLINE void NS(Align_set_tilt)(
    NS(Align)* SIXTRL_RESTRICT align, SIXTRL_REAL_T const tilt )
{
    SIXTRL_ASSERT( align != 0 );
    align->tilt = tilt;
    return;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Align_get_cz)(
    const NS(Align) *const SIXTRL_RESTRICT align )
{
    SIXTRL_ASSERT( align != 0 );
    return align->cz;
}

SIXTRL_INLINE void NS(Align_set_cz)(
    NS(Align)* SIXTRL_RESTRICT align, SIXTRL_REAL_T const cz )
{
    SIXTRL_ASSERT( align != 0 );
    align->cz = cz;
    return;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Align_get_sz)(
    const NS(Align) *const SIXTRL_RESTRICT align )
{
    SIXTRL_ASSERT( align != 0 );
    return align->sz;
}

SIXTRL_INLINE void NS(Align_set_sz)(
    NS(Align)* SIXTRL_RESTRICT align, SIXTRL_REAL_T const sz )
{
    SIXTRL_ASSERT( align != 0 );
    align->sz = sz;
    return;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Align_get_dx)( 
    const NS(Align) *const SIXTRL_RESTRICT align )
{
    SIXTRL_ASSERT( align != 0 );
    return align->dx;
}

SIXTRL_INLINE void NS(Align_set_dx)(
    NS(Align)* SIXTRL_RESTRICT align, SIXTRL_REAL_T const dx )
{
    SIXTRL_ASSERT( align != 0 );
    align->dx = dx;
    return;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Align_get_dy)(
    const NS(Align) *const SIXTRL_RESTRICT align )
{
    SIXTRL_ASSERT( align != 0 );
    return align->dy;
}

SIXTRL_INLINE void NS(Align_set_dy)(
    NS(Align)* SIXTRL_RESTRICT align, SIXTRL_REAL_T const dy )
{
    SIXTRL_ASSERT( align != 0 );
    align->dy = dy;
    return;
}

/* ========================================================================= */

#if !defined( _GPUCODE )
#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENTS_API_H__ */

/* end: sixtracklib/common/impl/beam_elements_api.h  */
