#ifndef SIXTRACKLIB_COMMON_BEAM_ELEMENT_DRIFT_H__
#define SIXTRACKLIB_COMMON_BEAM_ELEMENT_DRIFT_H__

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/impl/be_drift_impl.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

struct NS(Drift);
struct NS(DriftSingle);
struct NS(Particles);

int NS(Drift_create_from_single_drift)( 
    struct NS(Drift)* SIXTRL_RESTRICT drift, 
    struct NS(DriftSingle)* SIXTRL_RESTRICT single_drift );

int NS(Drift_is_valid)( const NS(Drift) *const SIXTRL_RESTRICT drift );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* SIXTRACKLIB_COMMON_BEAM_ELEMENT_DRIFT_H__ */

/* end:  sixtracklib/common/be_drift.h */

