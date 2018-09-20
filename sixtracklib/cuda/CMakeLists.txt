# ------------------------------------------------------------------------------
# --- CUDA Kernel host library:

if( SIXTRACKL_CUDA_COMPUTE_ARCH_LIST )

    foreach( A IN LISTS SIXTRACKL_CUDA_COMPUTE_ARCH_LIST )
        string( APPEND CMAKE_CUDA_FLAGS
                " -gencode=arch=compute_${A},code=compute_${A}" )
    endforeach()
endif()

string( APPEND CMAKE_CUDA_FLAGS " -D_FORCE_INLINES" )
string( APPEND CMAKE_CUDA_DEBUG_FLAGS " -G" )

set( SIXTRACKL_CUDA_HOST_DEFINITIONS )
list( APPEND SIXTRACKL_CUDA_HOST_DEFINITIONS ${SIXTRACKLIB_NAMESPACE_FLAGS} )

set( SIXTRACKLIB_CUDA_PART_HOST_HEADERS
     impl/cuda_tools.h
     impl/track_particles_kernel_c_wrapper.h
)

set( SIXTRACKLIB_CUDA_PART_HOST_SOURCES
     details/track_particles_kernel_c_wrapper.cu
)

add_library( sixtrack_cuda_host OBJECT
             ${SIXTRACKLIB_CUDA_PART_HOST_HEADERS}
             ${SIXTRACKLIB_CUDA_PART_HOST_SOURCES}
)

target_compile_definitions( sixtrack_cuda_host
    PUBLIC ${SIXTRACKL_CUDA_HOST_DEFINITIONS} )

target_include_directories( sixtrack_cuda_host PUBLIC ${CMAKE_SOURCE_DIR} )

# ------------------------------------------------------------------------------
# --- CUDA Kernel object library:

set( SIXTRACKL_CUDA_DEVICE_DEFINITIONS )
list( APPEND SIXTRACKL_CUDA_DEVICE_DEFINITIONS "_GPUCODE=1" )
list( APPEND SIXTRACKL_CUDA_DEVICE_DEFINITIONS ${SIXTRACKLIB_NAMESPACE_FLAGS} )

set( SIXTRACKLIB_CUDA_PART_DEVICE_HEADERS
     impl/cuda_tools.h
     impl/managed_buffer_remap_kernel.cuh
     impl/track_particles_kernel.cuh
)

set( SIXTRACKLIB_CUDA_PART_DEVICE_SOURCES
     details/managed_buffer_remap_kernel.cu
     details/track_particles_kernel.cu
)

add_library( sixtrack_cuda_device OBJECT
             ${SIXTRACKLIB_CUDA_PART_DEVICE_HEADERS}
             ${SIXTRACKLIB_CUDA_PART_DEVICE_SOURCES}
)

target_compile_definitions( sixtrack_cuda_device
    PUBLIC ${SIXTRACKL_CUDA_DEVICE_DEFINITIONS} )

target_include_directories( sixtrack_cuda_device PUBLIC ${CMAKE_SOURCE_DIR} )

# ------------------------------------------------------------------------------

set( SIXTRACKL_CUDA_OBJECT_LIBS )
set( SIXTRACKL_CUDA_OBJECT_LIBS ${SIXTRACKL_CUDA_OBJECT_LIBS} sixtrack_cuda_host   )
set( SIXTRACKL_CUDA_OBJECT_LIBS ${SIXTRACKL_CUDA_OBJECT_LIBS} sixtrack_cuda_device )


set_property( TARGET ${SIXTRACKL_CUDA_OBJECT_LIBS}
              PROPERTY POSITION_INDEPENDENT_CODE ON )

set_property( TARGET ${SIXTRACKL_CUDA_OBJECT_LIBS}
              PROPERTY LINKER_LANGUAGE C )

set_property( TARGET ${SIXTRACKL_CUDA_OBJECT_LIBS}
              PROPERTY CXX_STANDARD 11 )

set_property( TARGET ${SIXTRACKL_CUDA_OBJECT_LIBS}
              PROPERTY CXX_STANDARD_REQUIRED ON )

set_property( TARGET ${SIXTRACKL_CUDA_OBJECT_LIBS}
              PROPERTY CUDA_SEPARABLE_COMPILATION  ON  )

set_property( TARGET ${SIXTRACKL_CUDA_OBJECT_LIBS}
              PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF )

set_property( TARGET ${SIXTRACKL_CUDA_OBJECT_LIBS}
              PROPERTY CUDA_STANDARD 11 )

set_property( TARGET ${SIXTRACKL_CUDA_OBJECT_LIBS}
              PROPERTY CUDA_STANDARD_REQUIRED 11 )

# ------------------------------------------------------------------------------
# install:

set( SIXTRACKLIB_CUDA_INSTALL_PATH include/sixtracklib/cuda )

install( FILES ${SIXTRACKLIB_CUDA_PART_HOST_HEADERS}
         DESTINATION ${SIXTRACKLIB_CUDA_INSTALL_PATH} )

install( FILES ${SIXTRACKLIB_CUDA_PART_DEVICE_HEADERS}
         DESTINATION "${SIXTRACKLIB_CUDA_INSTALL_PATH}/impl" )

# end: sixtracklib/cuda/CMakeLists.txt
