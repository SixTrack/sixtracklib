# tests/sixtracklib/cuda/CMakeLists.txt

if( GTEST_FOUND )

    # ==========================================================================

    set(   UNIT_TEST_TARGETS )
    set(   SIXTRACKL_TEST_LIBRARIES
         ${SIXTRACKL_TEST_LIBRARIES}
         ${SIXTRACKL_GTEST_LIBRARIES}
     )

    # --------------------------------------------------------------------------
    # test_track:

    add_executable( test_track_cuda test_track.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_track_cuda )
    target_link_libraries( test_track_cuda PRIVATE ${SIXTRACKL_TEST_LIBRARIES} )
    add_test( CudaTrackTests test_track_cuda )

    # ==========================================================================
    # ====  set all properties:

    set_property(
        TARGET ${UNIT_TEST_TARGETS}
        APPEND PROPERTY INCLUDE_DIRECTORIES
        $<BUILD_INTERFACE:${SIXTRACKL_GTEST_INCLUDE_DIRS}>
    )

    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD 11 )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD_REQUIRED ON )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY COMPILE_OPTIONS
                  ${SIXTRACKLIB_CPU_FLAGS} -Wall -Werror -pedantic )

endif()
