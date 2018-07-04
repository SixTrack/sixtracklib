# tests/sixtracklib/opencl/CMakeLists.txt

if( GTEST_FOUND )

    set(   UNIT_TEST_TARGETS )
    set(   SIXTRACKL_TEST_LIBRARIES
         ${SIXTRACKL_TEST_LIBRARIES}
         ${SIXTRACKL_GTEST_LIBRARIES}
     )

    # --------------------------------------------------------------------------
    # test_ocl_environment_opencl:

    add_executable( test_ocl_environment_opencl test_ocl_environment.cpp )
    target_link_libraries( test_ocl_environment_opencl PRIVATE
                           ${SIXTRACKL_TEST_LIBRARIES} )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_ocl_environment_opencl )
    add_test( OpenCLOclEnvironmentTests test_ocl_environment_opencl )

    # --------------------------------------------------------------------------
    # test_track_opencl:

    add_executable( test_track_opencl test_track.cpp )
    target_link_libraries( test_track_opencl PRIVATE ${SIXTRACKL_TEST_LIBRARIES} )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_track_opencl )
    add_test( OpenCLTrackTests test_track_opencl )

    # *************************************************************************
    # Set all properties:

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
