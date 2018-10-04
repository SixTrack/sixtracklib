# tests/sixtracklib/testlib/cuda/kernels/CMakeLists.txt

# -----------------------------------------------------------------------------
# build device part of the submodule library:

set( TESTLIB_CUDA_DEVICE_PART_DEFINITIONS )
list( APPEND TESTLIB_CUDA_DEVICE_PART_DEFINITIONS "_GPUCODE=1" )

set( TESTLIB_CUDA_DEVICE_PART_HEADERS
     cuda_beam_elements_kernel.cuh
     cuda_buffer_generic_obj_kernel.cuh
     cuda_grid_dimensions_kernel.cuh
     cuda_particles_kernel.cuh
)

set( TESTLIB_CUDA_DEVICE_PART_SOURCES
     cuda_beam_elements_kernel.cu
     cuda_buffer_generic_obj_kernel.cu
     cuda_grid_dimensions_kernel.cu
     cuda_particles_kernel.cu
)

add_library( sixtrack_test_cuda_device OBJECT
             ${TESTLIB_CUDA_DEVICE_PART_HEADERS}
             ${TESTLIB_CUDA_DEVICE_PART_SOURCES}
)

target_compile_definitions( sixtrack_test_cuda_device
                            PUBLIC ${TESTLIB_CUDA_DEVICE_PART_DEFINITIONS} )

target_include_directories( sixtrack_test_cuda_device
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    PRIVATE $<BUILD_INTERFACE:"${CMAKE_SOURCE_DIR}/tests">
)

set_target_properties(  sixtrack_test_cuda_device
                        PROPERTIES POSITION_INDEPENDENT_CODE ON )

set_target_properties(  sixtrack_test_cuda_device
                        PROPERTIES LINKER_LANGUAGE C )

set_target_properties(  sixtrack_test_cuda_device
                        PROPERTIES CXX_STANDARD 11 )

set_target_properties(  sixtrack_test_cuda_device
                        PROPERTIES CXX_STANDARD_REQUIRED ON )

set_target_properties(  sixtrack_test_cuda_device
                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON )

set_target_properties(  sixtrack_test_cuda_device
                        PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF )

set_target_properties(  sixtrack_test_cuda_device
                        PROPERTIES CUDA_STANDARD 11 )

set_target_properties(  sixtrack_test_cuda_device
                        PROPERTIES CUDA_STANDARD_REQUIRED ON )

# ------------------------------------------------------------------------------
# pass the submodule library on to sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES
    ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_test_cuda_device> CACHE INTERNAL "" FORCE )

# ------------------------------------------------------------------------------
# Install :

set(  TESTLIB_CUDA_DEVICE_FILES_TO_INSTALL
    ${TESTLIB_CUDA_DEVICE_PART_HEADERS} )

if( SIXTRACKL_ENABLE_CUDA_RTC )
    set(  TESTLIB_CUDA_DEVICE_FILES_TO_INSTALL
        ${TESTLIB_CUDA_DEVICE_FILES_TO_INSTALL}
        ${TESTLIB_CUDA_DEVICE_PART_SOURCES} )
endif()

if( TESTLIB_CUDA_DEVICE_FILES_TO_INSTALL )

    set( TESTLIB_CUDA_KERNELS_INSTALL_PATH
         include/sixtracklib/testlib/cuda/kernels )

    install( FILES ${TESTLIB_CUDA_DEVICE_FILES_TO_INSTALL}
             DESTINATION ${TESTLIB_CUDA_KERNELS_INSTALL_PATH} )

endif()

#end: tests/sixtracklib/testlib/cuda/kernels/CMakeLists.txt
