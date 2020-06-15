# sixtracklib/tests/sixtracklib/common/math/CMakeLists.txt

if( GTEST_FOUND )
    set( SIXTRACKL_COMMON_TESTS_DEPEND_ON_TRACKING_TESTDATA )

    set( UNIT_TEST_TARGETS )
    set( SIXTRACKL_TEST_LIBRARIES ${SIXTRACKL_TEST_LIBRARIES}
         ${SIXTRACKL_GTEST_LIBRARIES} )

    # ==========================================================================
    # test_debug_register_common_c99:

    add_executable( test_interpol_linear_common_c99
                    test_interpol_linear_c99.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_interpol_linear_common_c99 )

    add_test( C99_CommonControlDebuggingRegisterTests
              test_interpol_linear_common_c99 )

    # *************************************************************************
    # Set all properties:

    if( UNIT_TEST_TARGETS )
        set_property(
            TARGET ${UNIT_TEST_TARGETS}
            APPEND PROPERTY INCLUDE_DIRECTORIES
            $<BUILD_INTERFACE:${SIXTRACKL_GTEST_INCLUDE_DIRS}>
        )

        set_property( TARGET ${UNIT_TEST_TARGETS}
                      PROPERTY LINK_LIBRARIES ${SIXTRACKL_TEST_LIBRARIES} )

        set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD 11 )
        set_property( TARGET ${UNIT_TEST_TARGETS}
                      PROPERTY CXX_STANDARD_REQUIRED ON )

        set_property( TARGET ${UNIT_TEST_TARGETS}
                      APPEND PROPERTY COMPILE_OPTIONS
                      ${SIXTRACKLIB_CXX_FLAGS} ${SIXTRACKLIB_CPU_FLAGS} )
    endif()
endif()

#end : sixtracklib/tests/sixtracklib/common/control/CMakeLists.txt
