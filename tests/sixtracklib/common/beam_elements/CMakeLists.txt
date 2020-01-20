# sixtracklib/tests/sixtracklib/common/beam_elements/CMakeLists.txt

if( GTEST_FOUND )

    set( UNIT_TEST_TARGETS )
    set( SIXTRACKL_TEST_LIBRARIES ${SIXTRACKL_TEST_LIBRARIES}
         ${SIXTRACKL_GTEST_LIBRARIES} )

    # ==========================================================================
    # test_be_drift_common_c99:

    add_executable( test_be_drift_common_c99      test_be_drift_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}   test_be_drift_common_c99 )
    add_test( C99_Common_BeamElements_Drift_Tests test_be_drift_common_c99 )

    # --------------------------------------------------------------------------
    # test_be_drift_common_cxx:

    add_executable( test_be_drift_common_cxx      test_be_drift_cxx.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}   test_be_drift_common_cxx )
    add_test( CXX_Common_BeamElements_Drift_Tests test_be_drift_common_cxx )

    # ==========================================================================
    # test_be_multipole_common_c99:

    add_executable( test_be_multipole_common_c99  test_be_multipole_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}   test_be_multipole_common_c99 )
    add_test( C99_Common_BeamElements_MultiPole_Tests
              test_be_multipole_common_c99 )

    # ==========================================================================
    # test_be_rf_multipole_common_c99:

    add_executable( test_be_rf_multipole_common_c99
            test_be_rf_multipole_c99.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
            test_be_rf_multipole_common_c99 )

    add_test( C99_Common_BeamElements_RFMultiPole_Tests
            test_be_rf_multipole_common_c99 )

    # ==========================================================================
    # test_be_xy_shift_common_c99:

    add_executable( test_be_xy_shift_common_c99   test_be_xy_shift_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}   test_be_xy_shift_common_c99 )
    add_test( C99_Common_BeamElements_XYShift_Tests
              test_be_xy_shift_common_c99 )

    # ==========================================================================
    # test_be_srotation_common_c99:

    add_executable( test_be_srotation_common_c99  test_be_srotation_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}   test_be_srotation_common_c99 )
    add_test( C99_Common_BeamElements_SRotation_Tests
              test_be_srotation_common_c99 )

    # ==========================================================================
    # test_be_cavity_common_c99:

    add_executable( test_be_cavity_common_c99   test_be_cavity_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_be_cavity_common_c99 )
    add_test( C99_Common_BeamElements_Cavity_Tests
              test_be_cavity_common_c99 )

    # ==========================================================================
    # test_be_monitor_common_c99:

    add_executable( test_be_monitor_common_c99   test_be_monitor_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}  test_be_monitor_common_c99 )
    add_test( C99_Common_BeamElements_BeamMonitor_Tests
              test_be_monitor_common_c99 )

    # ==========================================================================
    # test_be_limit_rect_common_c99:

    add_executable( test_be_limit_rect_common_c99  test_be_limit_rect_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_be_limit_rect_common_c99 )

    add_test( C99_Common_BeamElements_LimitRect_Tests
              test_be_limit_rect_common_c99 )

    # --------------------------------------------------------------------------
    # test_be_limit_rect_common_cxx:

    add_executable( test_be_limit_rect_common_cxx  test_be_limit_rect_cxx.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_be_limit_rect_common_cxx )

    add_test( CXX_Common_BeamElements_LimitRect_Tests
              test_be_limit_rect_common_cxx )

    # ==========================================================================
    # test_be_limit_ellipse_common_c99:

    add_executable( test_be_limit_ellipse_common_c99
                    test_be_limit_ellipse_c99.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_be_limit_ellipse_common_c99 )

    add_test( C99_Common_BeamElements_LimitEllipse_Tests
              test_be_limit_ellipse_common_c99 )

    # --------------------------------------------------------------------------
    # test_be_limit_ellipse_common_cxx:

    add_executable( test_be_limit_ellipse_common_cxx
                    test_be_limit_ellipse_cxx.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_be_limit_ellipse_common_cxx )

    add_test( CXX_Common_BeamElements_LimitEllipse_Tests
              test_be_limit_ellipse_common_cxx )

    # ==========================================================================
    # test_be_dipedge_common_c99:

    add_executable( test_be_dipedge_common_c99     test_be_dipedge_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}    test_be_dipedge_common_c99 )
    add_test( C99_Common_BeamElements_DipoleEdge_Tests
              test_be_dipedge_common_c99 )

    # --------------------------------------------------------------------------
    # test_be_dipedge_common_cxx:

    add_executable( test_be_dipedge_common_cxx     test_be_dipedge_cxx.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}    test_be_dipedge_common_cxx )
    add_test( CXX_Common_BeamElements_DipoleEdge_Tests
              test_be_dipedge_common_cxx )

    # *************************************************************************
    # Set all properties:

    set_property( TARGET ${UNIT_TEST_TARGETS}
        APPEND PROPERTY INCLUDE_DIRECTORIES
        $<BUILD_INTERFACE:${SIXTRACKL_GTEST_INCLUDE_DIRS}>
    )

    set_property( TARGET ${UNIT_TEST_TARGETS}
                  PROPERTY LINK_LIBRARIES ${SIXTRACKL_TEST_LIBRARIES} )

    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD 11 )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD_REQUIRED ON )

    set_property( TARGET ${UNIT_TEST_TARGETS}
                  APPEND PROPERTY COMPILE_OPTIONS
                  ${SIXTRACKLIB_CXX_FLAGS} ${SIXTRACKLIB_CPU_FLAGS} )

endif()

