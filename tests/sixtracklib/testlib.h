#ifndef SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_H__
#define SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_H__

#if !defined( __NAMESPACE )
    #define __NAMESPACE st_
    #define __SIXTRACKLIB_TESTS_SIXTRACKLIB_UNDEF_NAMESPACE 1
#endif /* !defined( __NAMESPACE ) */

/* ------------------------------------------------------------------------- */

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/namespace_begin.h"

#include "sixtracklib/testlib/gpu_kernel_tools.h"
#include "sixtracklib/testlib/random.h"
#include "sixtracklib/testlib/test_particles_tools.h"
#include "sixtracklib/testlib/test_track_tools.h"

#include "sixtracklib/testlib/testdata_files.h"

/* ------------------------------------------------------------------------- */

#if defined( __SIXTRACKLIB_TESTS_SIXTRACKLIB_UNDEF_NAMESPACE )
    #include "sixtracklib/_impl/namespace_end.h"

    #undef __NAMESPACE
    #undef __SIXTRACKLIB_TESTS_SIXTRACKLIB_UNDEF_NAMESPACE
#endif /* defined( __SIXTRACKLIB_TESTS_SIXTRACKLIB_UNDEF_NAMESPACE ) */

#endif /* SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_H__ */

/* end: tests/sixtracklib/testlib.h */
