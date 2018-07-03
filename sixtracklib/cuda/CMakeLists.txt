# ------------------------------------------------------------------------------
# --- CUDA Kernel host library:

if( SIXTRACKL_CUDA_COMPUTE_ARCH_LIST )

    foreach( A IN LISTS SIXTRACKL_CUDA_COMPUTE_ARCH_LIST )
        string( APPEND CMAKE_CUDA_FLAGS
                " -gencode=arch=compute_${A},code=compute_${A}" )
    endforeach()
endif()

set( SIXTRACKL_CUDA_HOST_DEFINITIONS )
list( APPEND SIXTRACKL_CUDA_HOST_DEFINITIONS
      "__NAMESPACE=${SIXTRACKL_C_NAMESPACE_PREFIX}" )

set( SIXTRACKLIB_CUDA_PART_HOST_HEADERS
     cuda_env.h
)

set( SIXTRACKLIB_CUDA_PART_HOST_SOURCES
     details/cuda_env.cu
)

add_library( sixtrack_cuda_host OBJECT
             ${SIXTRACKLIB_CUDA_PART_HOST_HEADERS}
             ${SIXTRACKLIB_CUDA_PART_HOST_SOURCES}
)

target_compile_definitions( sixtrack_cuda_host
    PUBLIC ${SIXTRACKL_CUDA_HOST_DEFINITIONS} )

# ------------------------------------------------------------------------------
# --- CUDA Kernel object library:

set( SIXTRACKL_CUDA_KERNEL_DEFINITIONS )
list( APPEND SIXTRACKL_CUDA_KERNEL_DEFINITIONS "_GPUCODE=1" )
list( APPEND SIXTRACKL_CUDA_KERNEL_DEFINITIONS
      "__NAMESPACE=${SIXTRACKL_C_NAMESPACE_PREFIX}" )


set( SIXTRACKLIB_CUDA_PART_KERNEL_HEADERS
     impl/track_particles_kernel.cuh
)

set( SIXTRACKLIB_CUDA_PART_KERNEL_SOURCES
     details/track_particles_kernel.cu
)

add_library( sixtrack_cuda_kernel OBJECT
             ${SIXTRACKLIB_CUDA_PART_KERNEL_HEADERS}
             ${SIXTRACKLIB_CUDA_PART_KERNEL_SOURCES}
)

target_compile_definitions( sixtrack_cuda_kernel
    PUBLIC ${SIXTRACKL_CUDA_KERNEL_DEFINITIONS} )

# ------------------------------------------------------------------------------

set( SIXTRACKL_CUDA_OBJECT_LIBS sixtrack_cuda_host sixtrack_cuda_kernel )

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

if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )
    add_subdirectory( tests )
endif()

# ------------------------------------------------------------------------------
# set( CUDA_PROPAGATE_HOST_FLAGS ${SAVED_CUDA_PROPAGATE_HOST_FLAGS} )

