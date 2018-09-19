# tests/sixtracklib/cuda/CMakeLists.txt

if( GTEST_FOUND )

    # ==========================================================================

    set(   UNIT_TEST_TARGETS )
    set(   SIXTRACKL_TEST_LIBRARIES ${SIXTRACKL_TEST_LIBRARIES}
         ${SIXTRACKL_GTEST_LIBRARIES} sixtrack_test sixtrack m dl )


    if( SIXTRACKL_CUDA_COMPUTE_ARCH_LIST )

        foreach( A IN LISTS SIXTRACKL_CUDA_COMPUTE_ARCH_LIST )
            string( APPEND CMAKE_CUDA_FLAGS
                    " -gencode=arch=compute_${A},code=compute_${A}" )
        endforeach()
    endif()

    string( APPEND CMAKE_CUDA_FLAGS " -D_FORCE_INLINES" )
    string( APPEND CMAKE_CUDA_FLAGS " -G -g" )

    # --------------------------------------------------------------------------
    # tests/sixtracklib/cuda: cuda_testlib OBJECT library

    set( SIXTRACKL_CUDA_KERNEL_DEFINITIONS )
    list( APPEND SIXTRACKL_CUDA_KERNEL_DEFINITIONS "_GPUCODE=1" )

    if( SIXTRACKL_C_NAMESPACE_PREFIX )
        list( APPEND SIXTRACKL_CUDA_KERNEL_DEFINITIONS
              "__NAMESPACE=${SIXTRACKL_C_NAMESPACE_PREFIX}" )
    else()
        list( APPEND SIXTRACKL_CUDA_KERNEL_DEFINITIONS
              "__NAMESPACE=st_" )
    endif()

    if( SIXTRACKL_CXX_NAMESPACE )
        list( APPEND SIXTRACKL_CUDA_KERNEL_DEFINITIONS
              "SIXTRL_NAMESPACE=${SIXTRACKL_CXX_NAMESPACE}" )
    else()
        list( APPEND SIXTRACKL_CUDA_KERNEL_DEFINITIONS
              "SIXTRL_NAMESPACE=sixtrack" )
    endif()

    set( TESTS_SIXTRACKLIB_CUDA_TESTLIB_HEADERS
         impl/cuda_test_grid_dimensions.cuh
         impl/cuda_buffer_generic_obj_kernel.cuh
    )

    set( TESTS_SIXTRACKLIB_CUDA_TESTLIB_SOURCES
         details/cuda_grid_dimensions_kernel.cu
         details/cuda_buffer_generic_obj_kernel.cu
    )

    add_library( sixtrack_cuda_testlib_kernels OBJECT
                 ${TESTS_SIXTRACKLIB_CUDA_TESTLIB_HEADERS}
                 ${TESTS_SIXTRACKLIB_CUDA_TESTLIB_SOURCES}
    )

    target_compile_definitions( sixtrack_cuda_testlib_kernels
        PUBLIC ${SIXTRACKL_CUDA_KERNEL_DEFINITIONS} )

    set_property( TARGET sixtrack_cuda_testlib_kernels
                  APPEND PROPERTY INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR} )

    set_property( TARGET sixtrack_cuda_testlib_kernels
                  APPEND PROPERTY INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/tests" )

    set_target_properties(  sixtrack_cuda_testlib_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON )
    set_target_properties(  sixtrack_cuda_testlib_kernels PROPERTIES LINKER_LANGUAGE C )
    set_target_properties(  sixtrack_cuda_testlib_kernels PROPERTIES CXX_STANDARD 11 )
    set_target_properties(  sixtrack_cuda_testlib_kernels PROPERTIES CXX_STANDARD_REQUIRED ON )
    set_target_properties(  sixtrack_cuda_testlib_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
    set_target_properties(  sixtrack_cuda_testlib_kernels PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF )
    set_target_properties(  sixtrack_cuda_testlib_kernels PROPERTIES CUDA_STANDARD 11 )
    set_target_properties(  sixtrack_cuda_testlib_kernels PROPERTIES CUDA_STANDARD_REQUIRED ON )

    # --------------------------------------------------------------------------
    # test_single_file_compilation_cuda_cxx:

    add_executable( test_single_file_compilation_cuda_cxx
                    test_single_file_compilation_cuda_cxx.cpp
                    $<TARGET_OBJECTS:sixtrack_cuda_testlib_kernels> )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_single_file_compilation_cuda_cxx )

    target_link_libraries( test_single_file_compilation_cuda_cxx
        PRIVATE ${SIXTRACKL_TEST_LIBRARIES} )

    add_test( CXX_Cuda_SingleFileCompilation test_single_file_compilation_cuda_cxx )

    # --------------------------------------------------------------------------
    # test_buffer_cuda_c99:

    add_executable( test_buffer_cuda_c99 test_buffer_cuda_c99.cpp
                    $<TARGET_OBJECTS:sixtrack_cuda_testlib_kernels> )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_buffer_cuda_c99 )

    target_link_libraries( test_buffer_cuda_c99 PRIVATE ${SIXTRACKL_TEST_LIBRARIES} )
    add_test( C99_Cuda_BufferTests test_buffer_cuda_c99 )

     # --------------------------------------------------------------------------
    # test_single_file_compilation_cuda_cxx:

    add_executable( test_runtime_compilation_cuda_cxx
                    test_runtime_compilation_cuda_cxx.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_runtime_compilation_cuda_cxx )

    target_link_libraries( test_runtime_compilation_cuda_cxx
        PRIVATE nvrtc cuda ${SIXTRACKL_TEST_LIBRARIES} )

    add_test( CXX_Cuda_RunTimeCompilation test_runtime_compilation_cuda_cxx )

    # ==========================================================================
    # ====  set all properties:

    set_property(
        TARGET ${UNIT_TEST_TARGETS}
        APPEND PROPERTY INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/tests" )

    set_property(
        TARGET ${UNIT_TEST_TARGETS}
        APPEND PROPERTY INCLUDE_DIRECTORIES
        $<BUILD_INTERFACE:${SIXTRACKL_GTEST_INCLUDE_DIRS}>
    )

    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY POSITION_INDEPENDENT_CODE ON )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY LINKER_LANGUAGE C )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD 11 )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD_REQUIRED ON )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CUDA_SEPARABLE_COMPILATION  ON  )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CUDA_STANDARD 11 )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CUDA_STANDARD_REQUIRED 11 )

endif()
