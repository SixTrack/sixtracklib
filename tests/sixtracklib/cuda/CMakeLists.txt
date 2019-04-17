# tests/sixtracklib/cuda/CMakeLists.txt

if( GTEST_FOUND )

    # ==========================================================================

    set(   UNIT_TEST_TARGETS )

    set(   SIXTRACKL_TEST_LIBRARIES ${SIXTRACKL_TEST_LIBRARIES}
         ${SIXTRACKL_GTEST_LIBRARIES} sixtrack_test sixtrack )

    if( SIXTRACKL_CUDA_COMPUTE_ARCH_LIST )
        foreach( A IN LISTS SIXTRACKL_CUDA_COMPUTE_ARCH_LIST )
            string( APPEND CMAKE_CUDA_FLAGS
                    " -gencode=arch=compute_${A},code=compute_${A}" )
        endforeach()
    endif()

    string( APPEND CMAKE_CUDA_FLAGS " -D_FORCE_INLINES" )

    if( CMAKE_BUILD_TYPE MATCHES "Debug" )
        string( APPEND CMAKE_CUDA_FLAGS " -g -G" )
    else()
        string( APPEND CMAKE_CUDA_FLAGS " -lineinfo" )
    endif()

    # ==========================================================================
    # test_context_cuda_c99

    add_executable( test_context_cuda_c99 test_context_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_context_cuda_c99 )

    target_link_libraries( test_context_cuda_c99
        PRIVATE ${SIXTRACKL_TEST_LIBRARIES} )

    add_test( C99_CudaContextTests test_context_cuda_c99 )

    # --------------------------------------------------------------------------
    # test_context_cuda_cxx

    add_executable( test_context_cuda_cxx test_context_cxx.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_context_cuda_cxx )

    target_link_libraries( test_context_cuda_cxx
        PRIVATE ${SIXTRACKL_TEST_LIBRARIES} )

    add_test( CXX_CudaContextTests test_context_cuda_cxx )

    # ==========================================================================
    # test_argument_cuda_c99

    add_executable( test_argument_cuda_c99 test_argument_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_argument_cuda_c99 )

    target_link_libraries( test_argument_cuda_c99
        PRIVATE ${SIXTRACKL_TEST_LIBRARIES} )

    add_test( C99_CudaArgumentTests test_argument_cuda_c99 )

    # --------------------------------------------------------------------------
    # test_argument_cuda_cxx

    add_executable( test_argument_cuda_cxx test_argument_cxx.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_argument_cuda_cxx )

    target_link_libraries( test_argument_cuda_cxx
        PRIVATE ${SIXTRACKL_TEST_LIBRARIES} )

    add_test( CXX_CudaArgumentTests test_argument_cuda_cxx )

    # --------------------------------------------------------------------------
    # test_single_file_compilation_cuda_cxx:

    add_executable( test_single_file_compilation_cuda_cxx
                    test_single_file_compilation_cuda_cxx.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_single_file_compilation_cuda_cxx )

    target_link_libraries( test_single_file_compilation_cuda_cxx
        PRIVATE ${SIXTRACKL_TEST_LIBRARIES} )

    add_test( CXX_Cuda_SingleFileCompilation
              test_single_file_compilation_cuda_cxx )

    # --------------------------------------------------------------------------
    # test_buffer_cuda_c99:

    add_executable( test_buffer_cuda_c99 test_buffer_cuda_c99.cpp  )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_buffer_cuda_c99 )

    target_link_libraries( test_buffer_cuda_c99 PRIVATE ${SIXTRACKL_TEST_LIBRARIES} )
    add_test( C99_Cuda_BufferTests test_buffer_cuda_c99 )

    # --------------------------------------------------------------------------
    # test_particles_cuda_c99:

    add_executable( test_particles_cuda_c99 test_particles_cuda_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_particles_cuda_c99 )

    target_link_libraries( test_particles_cuda_c99 PRIVATE ${SIXTRACKL_TEST_LIBRARIES} )
    add_test( C99_Cuda_ParticlesBufferTests test_particles_cuda_c99 )

    # --------------------------------------------------------------------------
    # test_be_drift_cuda_c99:

    add_executable( test_be_drift_cuda_c99 test_be_drift_cuda_c99.cpp  )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_be_drift_cuda_c99 )

    target_link_libraries( test_be_drift_cuda_c99 PRIVATE ${SIXTRACKL_TEST_LIBRARIES} )
    add_test( C99_Cuda_BeamElementsDriftTests test_be_drift_cuda_c99 )

    # ==========================================================================
    # test_track_line_cuda_c99:

    add_executable( test_track_line_cuda_c99 test_track_line_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_track_line_cuda_c99 )

    target_link_libraries( test_track_line_cuda_c99 PRIVATE ${SIXTRACKL_TEST_LIBRARIES} )
    add_test( C99_CudaTrackLineTests test_track_line_cuda_c99 )

    # --------------------------------------------------------------------------
    # test_single_file_compilation_cuda_cxx:

    if( SIXTRACKL_ENABLE_CUDA_RTC )

        add_executable( test_runtime_compilation_cuda_cxx
                        test_runtime_compilation_cuda_cxx.cpp )

        set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
             test_runtime_compilation_cuda_cxx )

        target_link_libraries( test_runtime_compilation_cuda_cxx
              PRIVATE nvrtc cuda ${SIXTRACKL_TEST_LIBRARIES} )

        add_test( CXX_Cuda_RunTimeCompilation
                  test_runtime_compilation_cuda_cxx )
    endif()

    # ==========================================================================
    # ====  set all properties:

    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY POSITION_INDEPENDENT_CODE ON )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY LINKER_LANGUAGE C )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD 11 )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD_REQUIRED ON )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CUDA_SEPARABLE_COMPILATION  ON  )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CUDA_STANDARD 11 )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CUDA_STANDARD_REQUIRED 11 )

endif()
