# tests/sixtracklib/testlib/cuda/CMakeLists.txt

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

# -----------------------------------------------------------------------------
# build device part of the submodule library in kernels subdirectory:

add_subdirectory( kernels )

# -----------------------------------------------------------------------------
# build host part of the submodule library:

set( TESTLIB_CUDA_HOST_PART_HEADERS
     cuda_beam_elements_kernel_c_wrapper.h
     cuda_buffer_generic_obj_kernel_c_wrapper.h
     cuda_particles_kernel_c_wrapper.h
)

set( TESTLIB_CUDA_HOST_PART_SOURCES
     cuda_beam_elements_kernel_c_wrapper.cu
     cuda_buffer_generic_obj_kernel_c_wrapper.cu
     cuda_particles_kernel_c_wrapper.cu
)

add_library( sixtrack_test_cuda_host OBJECT
             ${TESTLIB_CUDA_HOST_PART_HEADERS}
             ${TESTLIB_CUDA_HOST_PART_SOURCES}
)

target_include_directories( sixtrack_test_cuda_host
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tests>
)

set_target_properties( sixtrack_test_cuda_host
                       PROPERTIES POSITION_INDEPENDENT_CODE ON )

set_target_properties( sixtrack_test_cuda_host
                       PROPERTIES LINKER_LANGUAGE C )

set_target_properties( sixtrack_test_cuda_host
                       PROPERTIES CXX_STANDARD 11 )

set_target_properties( sixtrack_test_cuda_host
                       PROPERTIES CXX_STANDARD_REQUIRED ON )

# ------------------------------------------------------------------------------
# pass the submodule library on to sixtracklib:

set(  SIXTRACKL_TESTLIB_LIBRARY_MODULES
    ${SIXTRACKL_TESTLIB_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_test_cuda_host> CACHE INTERNAL "" FORCE )

if( SIXTRACKL_ENABLE_CUDA_RTC )
    set(   SIXTRACKL_TESTLIB_LINK_LIBRARIES
           ${SIXTRACKL_TESTLIB_LINK_LIBRARIES} nvrtc CACHE INTERNAL "" FORCE )
endif()

set(   SIXTRACKL_TESTLIB_LINK_LIBRARIES
     ${SIXTRACKL_TESTLIB_LINK_LIBRARIES} cudart CACHE INTERNAL "" FORCE )

# ------------------------------------------------------------------------------
# Install :

if( TESTLIB_CUDA_HOST_PART_HEADERS )

    set( TESTLIB_CUDA_INSTALL_PATH include/sixtracklib/testlib/cuda )

    install( FILES ${TESTLIB_CUDA_INSTALL_PATH}
             DESTINATION ${TESTLIB_CUDA_INSTALL_PATH} )

endif()

#end: tests/sixtracklib/testlib/cuda/CMakeLists.txt
