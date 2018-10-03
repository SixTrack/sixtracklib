# ------------------------------------------------------------------------------
# --- CUDA Kernel host library:

if( SIXTRACKL_CUDA_COMPUTE_ARCH_LIST )

    foreach( A IN LISTS SIXTRACKL_CUDA_COMPUTE_ARCH_LIST )
        string( APPEND CMAKE_CUDA_FLAGS
                " -gencode=arch=compute_${A},code=compute_${A}" )
    endforeach()
endif()

string( APPEND CMAKE_CUDA_FLAGS " -D_FORCE_INLINES" )
string( APPEND CMAKE_CUDA_FLAGS " -cudart shared" )

if( CMAKE_BUILD_TYPE MATCHES "Debug" )
    string( APPEND CMAKE_CUDA_FLAGS " -g -G" )
else()
    string( APPEND CMAKE_CUDA_FLAGS " -lineinfo" )
endif()

set( SIXTRACKL_CUDA_HOST_DEFINITIONS )
list( APPEND SIXTRACKL_CUDA_HOST_DEFINITIONS ${SIXTRACKLIB_NAMESPACE_FLAGS} )

set( SIXTRACKLIB_CUDA_PART_HOST_HEADERS
     cuda_tools.h
     track_particles_kernel_c_wrapper.h
)

set( SIXTRACKLIB_CUDA_PART_HOST_SOURCES
     internal/track_particles_kernel_c_wrapper.cu
)

add_library( sixtrack_cuda_host OBJECT
             ${SIXTRACKLIB_CUDA_PART_HOST_HEADERS}
             ${SIXTRACKLIB_CUDA_PART_HOST_SOURCES}
)

target_compile_definitions( sixtrack_cuda_host
    PUBLIC ${SIXTRACKL_CUDA_HOST_DEFINITIONS} )

target_include_directories( sixtrack_cuda_host
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> )

set_target_properties( sixtrack_cuda_host PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties( sixtrack_cuda_host PROPERTIES LINKER_LANGUAGE C )
set_target_properties( sixtrack_cuda_host PROPERTIES CXX_STANDARD 11 )
set_target_properties( sixtrack_cuda_host PROPERTIES CXX_STANDARD_REQUIRED ON )

# ------------------------------------------------------------------------------
# pass on sixtrack_cuda_host as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES
    ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_cuda_host> CACHE INTERNAL "" FORCE )

find_library( CUDA_LIBRARY cuda ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} )
find_library( CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} )

if( SIXTRACKL_ENABLE_CUDA_RTC )
    find_library( CUDA_NVRTC_LIBRARY nvrtc ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} )
else()
    set( CUDA_NVRTC_LIBRARY )
endif()

set(   SIXTRACKL_LINK_LIBRARIES
     ${SIXTRACKL_LINK_LIBRARIES}
     ${CUDA_NVRTC_LIBRARY} ${CUDA_LIBRARY} ${CUDART_LIBRARY}
     CACHE INTERNAL "" FORCE )

# ------------------------------------------------------------------------------
# --- CUDA Kernel object library:

set( SIXTRACKL_CUDA_DEVICE_DEFINITIONS )
list( APPEND SIXTRACKL_CUDA_DEVICE_DEFINITIONS "_GPUCODE=1" )
list( APPEND SIXTRACKL_CUDA_DEVICE_DEFINITIONS ${SIXTRACKLIB_NAMESPACE_FLAGS} )

set( SIXTRACKLIB_CUDA_PART_DEVICE_HEADERS
     cuda_tools.h
     kernels/managed_buffer_remap_kernel.cuh
     kernels/track_particles_kernel.cuh
)

set( SIXTRACKLIB_CUDA_PART_DEVICE_SOURCES
     kernels/managed_buffer_remap_kernel.cu
     kernels/track_particles_kernel.cu
)

add_library( sixtrack_cuda_device OBJECT
             ${SIXTRACKLIB_CUDA_PART_DEVICE_HEADERS}
             ${SIXTRACKLIB_CUDA_PART_DEVICE_SOURCES}
)

target_compile_definitions( sixtrack_cuda_device
    PUBLIC ${SIXTRACKL_CUDA_DEVICE_DEFINITIONS} )

set_target_properties( sixtrack_cuda_device PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties( sixtrack_cuda_device PROPERTIES LINKER_LANGUAGE C )
set_target_properties( sixtrack_cuda_device PROPERTIES CXX_STANDARD 11 )
set_target_properties( sixtrack_cuda_device PROPERTIES CXX_STANDARD_REQUIRED ON )
set_target_properties( sixtrack_cuda_device PROPERTIES CUDA_SEPARABLE_COMPILATION  ON  )
set_target_properties( sixtrack_cuda_device PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF )
set_target_properties( sixtrack_cuda_device PROPERTIES CUDA_STANDARD 11 )
set_target_properties( sixtrack_cuda_device PROPERTIES CUDA_STANDARD_REQUIRED 11 )

target_include_directories( sixtrack_cuda_device PRIVATE ${CMAKE_SOURCE_DIR} )

# ------------------------------------------------------------------------------
# pass on sixtrack_cuda_device as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES
    ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_cuda_device> CACHE INTERNAL "" FORCE )

# ------------------------------------------------------------------------------
# install:

if( SIXTRACKLIB_CUDA_PART_HOST_HEADERS )
    set( SIXTRACKLIB_CUDA_HOST_INSTALL_PATH include/sixtracklib/cuda )

    install( FILES ${SIXTRACKLIB_CUDA_PART_HOST_HEADERS}
             DESTINATION ${SIXTRACKLIB_CUDA_HOST_INSTALL_PATH} )
endif()

if( SIXTRACKLIB_CUDA_PART_DEVICE_HEADERS OR
    SIXTRACKLIB_CUDA_PART_DEVICE_SOURCES )

    set( SIXTRACKLIB_CUDA_DEVICE_INSTALL_PATH include/sixtracklib/cuda/kernels )

    install( FILES ${SIXTRACKLIB_CUDA_PART_DEVICE_HEADERS}
                   ${SIXTRACKLIB_CUDA_PART_DEVICE_SOURCES}
             DESTINATION ${SIXTRACKLIB_CUDA_DEVICE_INSTALL_PATH} )
endif()

set( SIXTRACKLIB_CUDA_HOST_INSTALL_PATH include/sixtracklib/cuda )
install( FILES ${SIXTRACKLIB_CUDA_PART_DEVICE_HEADERS}
               ${SIXTRACKLIB_CUDA_PART_DEVICE_SOURCES}
         DESTINATION "${SIXTRACKLIB_CUDA_INSTALL_PATH}/kernels" )

# end: sixtracklib/cuda/CMakeLists.txt
