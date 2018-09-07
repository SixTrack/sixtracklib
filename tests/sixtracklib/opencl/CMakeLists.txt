# tests/sixtracklib/opencl/CMakeLists.txt

if( GTEST_FOUND )

    set(   UNIT_TEST_TARGETS )
    set(   SIXTRACKL_TEST_LIBRARIES ${SIXTRACKL_TEST_LIBRARIES}
         ${SIXTRACKL_GTEST_LIBRARIES} sixtrack_test sixtrack m dl )

    # --------------------------------------------------------------------------
    # test_buffer_opencl_c99:

    add_executable( test_buffer_opencl_c99 test_buffer_opencl_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_buffer_opencl_c99 )
    target_link_libraries( test_buffer_opencl_c99 PRIVATE ${SIXTRACKL_TEST_LIBRARIES} )
    add_test( C99_OpenCL_BufferTests test_buffer_opencl_c99 )

    # --------------------------------------------------------------------------
    # test_track_opencl:

#     add_executable( test_track_opencl test_track.cpp )
#       target_link_libraries( test_track_opencl PRIVATE ${SIXTRACKL_TEST_LIBRARIES} )
#     set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_track_opencl )
#     add_test( OpenCLTrackTests test_track_opencl )

    # *************************************************************************
    # Set all properties:

    set_property(
        TARGET ${UNIT_TEST_TARGETS}
        APPEND PROPERTY INCLUDE_DIRECTORIES
        "${CMAKE_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/tests"
        $<BUILD_INTERFACE:${SIXTRACKL_GTEST_INCLUDE_DIRS}>
    )

    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD 11 )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD_REQUIRED ON )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY COMPILE_OPTIONS
                  ${SIXTRACKLIB_CPU_FLAGS} -Wall -Werror -pedantic )

endif()
