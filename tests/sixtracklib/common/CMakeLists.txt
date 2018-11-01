# sixtracklib/tests/sixtracklib/common/CMakeLists.txt

# -----------------------------------------------------------------------------
# ---- Generate testdata:

if( GTEST_FOUND )
    set( SIXTRACKL_COMMON_TESTS_DEPEND_ON_TRACKING_TESTDATA )

    set( UNIT_TEST_TARGETS )
    set( SIXTRACKL_TEST_LIBRARIES ${SIXTRACKL_TEST_LIBRARIES}
         ${SIXTRACKL_GTEST_LIBRARIES} )

    # --------------------------------------------------------------------------
    # test_tools_c99:

    add_executable( test_tools_common_c99 test_tools_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_tools_common_c99 )
    add_test( C99_CommonToolTests test_tools_common_c99 )

    # --------------------------------------------------------------------------
    # test_managed_buffer_common_c99:

    add_executable( test_managed_buffer_common_c99 test_managed_buffer_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_managed_buffer_common_c99 )
    add_test( C99_CommonManagedBufferTest test_managed_buffer_common_c99 )

    # --------------------------------------------------------------------------
    # test_buffer_common_c99:

    add_executable( test_buffer_common_c99 test_buffer_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_buffer_common_c99 )
    add_test( C99_CommonBufferTest test_buffer_common_c99 )

    # --------------------------------------------------------------------------
    # test_buffer_common_cxx:

    add_executable( test_buffer_common_cxx test_buffer_cxx.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_buffer_common_cxx )
    add_test( CXX_CommonBufferTest test_buffer_common_cxx )

    # --------------------------------------------------------------------------
    # test_gpu_kernel_tools_common_c99:

    add_executable( test_gpu_kernel_tools_common_c99
                    test_gpu_kernel_tools_c99.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_gpu_kernel_tools_common_c99 )

    add_test( C99_CommonGpuKernelTools
              test_gpu_kernel_tools_common_c99 )

    # --------------------------------------------------------------------------
    # test_mem_pool_common_c99:

    add_executable( test_mem_pool_common_c99      test_mem_pool_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}   test_mem_pool_common_c99 )
    add_test( C99_CommonMemPoolTests              test_mem_pool_common_c99 )

    # ==========================================================================
    # test_particles_common_c99:

    add_executable( test_particles_common_c99     test_particles_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}   test_particles_common_c99 )
    add_test( C99_CommonParticlesTests            test_particles_common_c99 )

    # --------------------------------------------------------------------------
    # test_particles_common_cxx:

    add_executable( test_particles_common_cxx     test_particles_cxx.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}   test_particles_common_cxx )
    add_test( CXX_CommonParticlesTests            test_particles_common_cxx )

    # ==========================================================================
    # test_particles_aperature_check_enabled_common_c99:

    add_executable( test_particles_aperature_check_enabled_common_c99
                    test_particles_aperature_c99.cpp )

    target_compile_options( test_particles_aperature_check_enabled_common_c99
        BEFORE PRIVATE -DSIXTRL_ENABLE_APERATURE_CHECK=1
                       -DSIXTRL_APERATURE_X_LIMIT=1.0
                       -DSIXTRL_APERATURE_Y_LIMIT=1.0 )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_particles_aperature_check_enabled_common_c99 )

    add_test( C99_CommonParticlesAperatureEnabledTests
              test_particles_aperature_check_enabled_common_c99 )

    # ------------------------------------------------------------------------
    # test_particles_aperature_check_disabled_common_c99:

    add_executable( test_particles_aperature_check_disabled_common_c99
                    test_particles_aperature_c99.cpp )

    target_compile_options( test_particles_aperature_check_disabled_common_c99
        BEFORE PRIVATE -DSIXTRL_ENABLE_APERATURE_CHECK=0 )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_particles_aperature_check_disabled_common_c99 )

    add_test( C99_CommonParticlesAperatureDisabledTests
              test_particles_aperature_check_disabled_common_c99 )


    # ==========================================================================
    # test_be_drift_common_c99:

    add_executable( test_be_drift_common_c99      test_be_drift_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}   test_be_drift_common_c99 )
    add_test( C99_CommonBeamElementDriftTests     test_be_drift_common_c99 )

    # --------------------------------------------------------------------------
    # test_be_multipole_common_c99:

    add_executable( test_be_multipole_common_c99  test_be_multipole_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}   test_be_multipole_common_c99 )
    add_test( C99_CommonBeamElementMultiPoleTests test_be_multipole_common_c99 )

    # --------------------------------------------------------------------------
    # test_be_xy_shift_common_c99:

    add_executable( test_be_xy_shift_common_c99   test_be_xy_shift_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}   test_be_xy_shift_common_c99 )
    add_test( C99_CommonBeamElementXYShiftTests   test_be_xy_shift_common_c99 )

    # --------------------------------------------------------------------------
    # test_be_srotation_common_c99:

    add_executable( test_be_srotation_common_c99  test_be_srotation_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}   test_be_srotation_common_c99 )
    add_test( C99_CommonBeamElementSRotationTests test_be_srotation_common_c99 )

    # --------------------------------------------------------------------------
    # test_be_cavity_common_c99:

    add_executable( test_be_cavity_common_c99     test_be_cavity_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}   test_be_cavity_common_c99 )
    add_test( C99_CommonBeamElementCavityTests    test_be_cavity_common_c99 )

    # --------------------------------------------------------------------------
    # test_be_monitor_common_c99:

    add_executable( test_be_monitor_common_c99    test_be_monitor_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}   test_be_monitor_common_c99 )
    add_test( C99_CommonBeamMonitorTests          test_be_monitor_common_c99 )

    # ==========================================================================
    # test_be_drift_common_cxx:

    add_executable( test_be_drift_common_cxx    test_be_drift_cxx.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_be_drift_common_cxx )
    add_test( CXX_CommonBeamElementDriftTests   test_be_drift_common_cxx )

    # --------------------------------------------------------------------------
    # test_faddeeva_errf:

#     add_executable( test_faddeeva_errf test_faddeeva_errf.cpp )
#     set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_faddeeva_errf )
#     add_test( CommonFaddeevaErrfnTests test_faddeeva_errf )

    # --------------------------------------------------------------------------
    # test_beam_beam_element_6d:

#     add_executable( test_beam_beam_element_6d test_beam_beam_element_6d.cpp )
#     set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_beam_beam_element_6d )
#     add_test( CommonBeamBeamElement6dTests, test_beam_beam_element_6d )

    # --------------------------------------------------------------------------
    # test_beam_elements:

#     add_executable( test_beam_elements_common test_beam_elements.cpp )
#     set(   UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_beam_elements_common )
#     add_test( CommonBeamElementsTests test_beam_elements_common )

    # ==========================================================================
    # test_track_common_c99:

    add_executable( test_track_common_c99 test_track_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_track_common_c99 )
    add_test( C99_CommonTrackTests test_track_common_c99 )

    # --------------------------------------------------------------------------

    if( SIXTRACKL_ENABLE_AUTOVECTORIZATION )
        add_executable( test_track_common_autovec_c99 test_track_c99.cpp )
        set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_track_common_autovec_c99 )

        target_compile_options( test_track_common_autovec_c99 BEFORE PRIVATE
            -ftree-vectorize -ftree-vectorizer-verbose=6 -fopt-info-loop
            -fno-fast-math --param vect-max-version-for-alias-checks=150 )

        add_test( C99_CommonTrackTestsAutoVec test_track_common_autovec_c99 )

    endif()

    # *************************************************************************
    # Set all properties:

    set_property(
        TARGET ${UNIT_TEST_TARGETS}
        APPEND PROPERTY INCLUDE_DIRECTORIES
        $<BUILD_INTERFACE:${SIXTRACKL_GTEST_INCLUDE_DIRS}>
    )

    set_property( TARGET ${UNIT_TEST_TARGETS}
                  PROPERTY LINK_LIBRARIES ${SIXTRACKL_TEST_LIBRARIES} )

    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD 11 )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD_REQUIRED ON )

    set_property( TARGET ${UNIT_TEST_TARGETS}
                  APPEND PROPERTY COMPILE_OPTIONS
                  ${SIXTRACKLIB_CPU_FLAGS} -Wall -Werror -pedantic )

endif()

# ------------------------------------------------------------------------------
# --- benchmark drifts:

if( SIXTRACKL_ENABLE_BENCHMARKS )


endif()
