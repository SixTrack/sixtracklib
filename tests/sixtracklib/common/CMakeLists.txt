# sixtracklib/tests/sixtracklib/common/CMakeLists.txt

# -----------------------------------------------------------------------------
# ---- Generate testdata:

set( SIXTRACKL_COMMON_GENERATE_TESTDATA_TRACKING_HEADERS )

set( SIXTRACKL_COMMON_GENERATE_TESTDATA_TRACKING_SOURCES
     generate_testdata_tracking.cpp
)

# add_executable( generate_testdata_tracking_common
#                 ${SIXTRACKL_COMMON_GENERATE_TESTDATA_TRACKING_HEADERS}
#                 ${SIXTRACKL_COMMON_GENERATE_TESTDATA_TRACKING_SOURCES} )

# target_link_libraries(  generate_testdata_tracking_common
#                         sixtrack_test sixtrack )

# set_property( TARGET generate_testdata_tracking_common
#               PROPERTY LINKER_LANGUAGE CXX )

# set_property( TARGET generate_testdata_tracking_common
#               PROPERTY CXX_STANDARD 11 )

# set_property( TARGET generate_testdata_tracking_common
#               PROPERTY CXX_STANDARD_REQUIRED ON )

# set_property( TARGET generate_testdata_tracking_common
#     PROPERTY COMPILE_OPTIONS ${SIXTRACKLIB_CPU_FLAGS} -Wall -Werror -pedantic )

# add_custom_command(
#     OUTPUT  "${SIXTRACKLIB_TESTDATA_DIR}/test_tracking.generated"
#     DEPENDS generate_testdata_tracking_common
#     COMMAND generate_testdata_tracking_common
#     COMMENT "Generating testdata for tracking from common .... "
#     VERBATIM )

# add_custom_target( testdata_common
#     DEPENDS "${SIXTRACKLIB_TESTDATA_DIR}/test_tracking.generated" )


if( GTEST_FOUND )
    set( SIXTRACKL_COMMON_TESTS_DEPEND_ON_TRACKING_TESTDATA )

    set( UNIT_TEST_TARGETS )
    set( SIXTRACKL_TEST_LIBRARIES ${SIXTRACKL_TEST_LIBRARIES}
         ${SIXTRACKL_GTEST_LIBRARIES} )

    # --------------------------------------------------------------------------
    # test_tools:

    add_executable( test_tools_common test_tools.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_tools_common )
    add_test( CommonToolTests test_tools_common )

    # --------------------------------------------------------------------------
    # test_block:

#     add_executable( test_blocks_common test_blocks.cpp )
#     set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_blocks_common )
#     add_test( CommonBlocksTests test_blocks_common )

    # --------------------------------------------------------------------------
    # test_buffer_mem:

    add_executable( test_managed_buffer_common test_managed_buffer.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_managed_buffer_common )
    add_test( CommonBufferTests test_managed_buffer_common )

    # --------------------------------------------------------------------------
    # test_buffer_common_c99:

    add_executable( test_buffer_common_c99 test_buffer_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_buffer_common_c99 )
    add_test( CommonBufferTests test_buffer_common_c99 )

    # --------------------------------------------------------------------------
    # test_buffer_common_cxx:

    add_executable( test_buffer_common_cxx test_buffer_cxx.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_buffer_common_cxx )
    add_test( CommonBufferTests test_buffer_common_cxx )

    # --------------------------------------------------------------------------
    # test_gpu_kernel_tools:

    add_executable( test_gpu_kernel_tools_common test_gpu_kernel_tools.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_gpu_kernel_tools_common )
    add_test( CommonGpuKernelTools test_gpu_kernel_tools_common )

    # --------------------------------------------------------------------------
    # test_mem_pool:

    add_executable( test_mem_pool_common test_mem_pool.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_mem_pool_common )
    add_test( CommonMemPoolTests test_mem_pool_common )

    # --------------------------------------------------------------------------
    # test_particles:

    add_executable( test_particles_common test_particles.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_particles_common )
    add_test( CommonParticlesTests test_particles_common )

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

    # --------------------------------------------------------------------------
    # test_track:

#     add_executable( test_track_common test_track.cpp )
#     set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_track_common )
#     add_test( CommonTrackTests test_track_common )

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
