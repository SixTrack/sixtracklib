# sixtracklib/tests/sixtracklib/common/CMakeLists.txt

add_subdirectory( buffer )
add_subdirectory( control )
add_subdirectory( math )
add_subdirectory( beam_elements )
add_subdirectory( track )

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
    add_test( C99_Common_Tool test_tools_common_c99 )

    # ==========================================================================
    # test_gpu_kernel_tools_common_c99:

    add_executable( test_gpu_kernel_tools_common_c99
                    test_gpu_kernel_tools_c99.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_gpu_kernel_tools_common_c99 )

    add_test( C99_Common_GpuKernelTools test_gpu_kernel_tools_common_c99 )

    # ==========================================================================
    # test_particles_common_c99:

    add_executable( test_particles_common_c99     test_particles_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}   test_particles_common_c99 )
    add_test( C99_Common_Particles                test_particles_common_c99 )

    # --------------------------------------------------------------------------
    # test_particles_common_cxx:

    add_executable( test_particles_common_cxx     test_particles_cxx.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}   test_particles_common_cxx )
    add_test( CXX_Common_Particles                test_particles_common_cxx )

    # ==========================================================================
    # test_particles_aperture_check_enabled_common_c99:

    add_executable( test_particles_aperture_check_enabled_common_c99
                    test_particles_aperture_c99.cpp )

    target_compile_options( test_particles_aperture_check_enabled_common_c99
        BEFORE PRIVATE -DSIXTRL_APERTURE_CHECK_AT_DRIFT=2
                       -DSIXTRL_APERTURE_X_LIMIT=1.0
                       -DSIXTRL_APERTURE_Y_LIMIT=1.0 )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_particles_aperture_check_enabled_common_c99 )

    add_test( C99_Common_Particles_ApertureEnabled
              test_particles_aperture_check_enabled_common_c99 )

    # ------------------------------------------------------------------------
    # test_particles_aperture_check_disabled_common_c99:

    add_executable( test_particles_aperture_check_disabled_common_c99
                    test_particles_aperture_c99.cpp )

    target_compile_options( test_particles_aperture_check_disabled_common_c99
        BEFORE PRIVATE -DSIXTRL_APERTURE_CHECK_AT_DRIFT=0 )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_particles_aperture_check_disabled_common_c99 )

    add_test( C99_Common_Particles_ApertureDisabled
              test_particles_aperture_check_disabled_common_c99 )

    # --------------------------------------------------------------------------
    # test_elem_by_elem_config_c99:

    add_executable( test_elem_by_elem_config_c99
                    test_elem_by_elem_config_c99.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}  test_elem_by_elem_config_c99 )
    add_test( C99_Common_Output_ElemByElemConfig test_elem_by_elem_config_c99 )

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
    # test_output_buffer_common_c99:

    add_executable( test_output_buffer_common_c99 test_output_buffer_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_output_buffer_common_c99 )
    add_test( C99_Common_Output_OutputBufferCommon
              test_output_buffer_common_c99 )

    # ==========================================================================
    # test_track_line_common_c99:

    add_executable( test_track_line_common_c99 test_track_line_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_track_line_common_c99 )
    add_test( C99_Common_Track_TrackLineCommon test_track_line_common_c99 )

    # ==========================================================================
    # test_track_job_common_c99:

#     add_executable( test_track_job_common_c99 test_track_job_c99.cpp )
#     set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_track_job_common_c99 )
#     add_test( C99_Common_Track_TrackJobCommon, test_track_job_common_c99 )

    # --------------------------------------------------------------------------
    # test_track_job_common_cxx:

    add_executable( test_track_job_common_cxx test_track_job_cxx.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_track_job_common_cxx )
    add_test( CXX_Common_Track_TrackJobCommon, test_track_job_common_cxx )

    # ==========================================================================
    # test_track_job_cpu_common_c99:

    add_executable( test_track_job_cpu_common_c99 test_track_job_cpu_c99.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_track_job_cpu_common_c99 )
    add_test( C99_Common_Track_TrackJobCpu test_track_job_cpu_common_c99 )

    # --------------------------------------------------------------------------
    # test_track_job_cpu_cxx:

    add_executable( test_track_job_cpu_common_cxx test_track_job_cpu_cxx.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_track_job_cpu_common_cxx )
    add_test( CXX_Common_Track_TrackJobCpu test_track_job_cpu_common_cxx )

    # --------------------------------------------------------------------------

    if( SIXTRACKL_ENABLE_AUTOVECTORIZATION )
        add_executable( test_track_common_autovec_c99 test_track_c99.cpp )
        target_compile_options( test_track_common_autovec_c99 BEFORE PRIVATE
            ${SIXTRACKL_C_ENABLED_AUTOVEC_FLAGS} )
        add_test( C99_Common_Track_AutoVec test_track_common_autovec_c99 )

        set_target_properties( test_track_common_autovec_c99 PROPERTIES
            POSITION_INDEPENDENT_CODE ON CXX_STANDARD 11
            CXX_STANDARD_REQUIRED ON )

        target_include_directories( test_track_common_autovec_c99 BEFORE PRIVATE
            $<BUILD_INTERFACE:${SIXTRACKL_GTEST_INCLUDE_DIRS}>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tests> )

        target_link_libraries( test_track_common_autovec_c99
            ${SIXTRACKL_TEST_LIBRARIES} )

        target_compile_options( test_track_common_autovec_c99
            BEFORE PRIVATE ${SIXTRACKLIB_CXX_FLAGS} ${SIXTRACKLIB_CPU_FLAGS}
            ${SIXTRACKL_CXX_ENABLED_AUTOVEC_FLAGS} )
    endif()

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -

    add_executable( test_track_common_c99 test_track_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_track_common_c99 )
    add_test( C99_Common_Track_DisabledAutoVec test_track_common_c99 )

    set_target_properties( test_track_common_c99 PROPERTIES
        POSITION_INDEPENDENT_CODE ON CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON )

    target_include_directories( test_track_common_c99 BEFORE PRIVATE
        $<BUILD_INTERFACE:${SIXTRACKL_GTEST_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tests> )

    target_link_libraries( test_track_common_c99 ${SIXTRACKL_TEST_LIBRARIES} )

    target_compile_options( test_track_common_c99
        BEFORE PRIVATE ${SIXTRACKLIB_CXX_FLAGS} ${SIXTRACKLIB_CPU_FLAGS}
        ${SIXTRACKL_CXX_DISABLED_AUTOVEC_FLAGS} )

    # *************************************************************************
    # Set all properties:

    set_property( TARGET ${UNIT_TEST_TARGETS} APPEND PROPERTY
        INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${SIXTRACKL_GTEST_INCLUDE_DIRS}> )

    set_property( TARGET ${UNIT_TEST_TARGETS}
                  PROPERTY LINK_LIBRARIES ${SIXTRACKL_TEST_LIBRARIES} )

    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD 11 )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD_REQUIRED ON )

    set_property( TARGET ${UNIT_TEST_TARGETS} APPEND PROPERTY COMPILE_OPTIONS
        ${SIXTRACKLIB_CXX_FLAGS} ${SIXTRACKLIB_CPU_FLAGS}
        ${SIXTRACKL_CXX_AUTOVEC_FLAGS} )


endif()
