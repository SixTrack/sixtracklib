
set( SAVED_CUDA_PROPAGATE_HOST_FLAGS ${CUDA_PROPAGATE_HOST_FLAGS} )
set( CUDA_PROPAGATE_HOST_FLAGS  OFF )
set( CUDA_SEPARABLE_COMPILATION ON  )

cuda_select_nvcc_arch_flags( SIXTRACKL_CUDA_ARCH_FLAGS 3.0 )

set( SIXTRACKLIB_CUDA_PART_HEADERS
     cuda_env.h
     impl/track_particles_kernel.cuh
)

set( SIXTRACKLIB_CUDA_PART_SOURCES 
     details/cuda_env.cu
     details/track_particles_kernel.cu
)

set( CXX_LANG  $<COMPILE_LANGUAGE:CXX>  )
set( CUDA_LANG $<COMPILE_LANGUAGE:CUDA> )

set( SIXTRACKL_CUDA_PART_CXX_DEFS ${SIXTRACKLIB_NAMESPACE_FLAGS} )
set( SIXTRACKL_CUDA_PART_CXX_FLAGS 
    "-Wall -Werror -pedantic ${SIXTRACKLIB_CPU_FLAGS}" )

set( SIXTRACKL_CUDA_PART_CUDA_FLAGS "${SIXTRACKL_CUDA_ARCH_FLAGS}" )
set( SIXTRACKL_CUDA_PART_CUDA_DEFS "_GPUCODE=1" )

set( SIXTRACKL_CUDA_PART_CUDA_DEFS ${SIXTRACKL_CUDA_PART_CUDA_DEFS} 
                                   ${SIXTRACKLIB_NAMESPACE_FLAGS} )

set( SIXTRACKL_CUDA_PART_CUDA_DEFS ${SIXTRACKL_CUDA_PART_CUDA_DEFS} 
                                   "_FORCE_INLINES=1" )
                                   
add_library( sixtrack_cuda_part OBJECT 
             ${SIXTRACKLIB_CUDA_PART_HEADERS}
             ${SIXTRACKLIB_CUDA_PART_SOURCES} 
)

set_target_properties( sixtrack_cuda_part PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
set_target_properties( sixtrack_cuda_part PROPERTIES CUDA_STANDARD 11 )

target_include_directories( sixtrack_cuda_part PUBLIC 
    ${CMAKE_SOURCE_DIR} ${SIXTRACKL_CUDA_INCLUDE_DIRS} )

set_target_properties( sixtrack_cuda_part PROPERTIES LINKER_LANGUAGE C )

set_target_properties( sixtrack_cuda_part 
    PROPERTIES POSITION_INDEPENDENT_CODE True )
    
set_target_properties( sixtrack_cuda_part PROPERTIES CXX_STANDARD  11 )
set_target_properties( sixtrack_cuda_part PROPERTIES C_STANDARD    99 )

target_compile_options( sixtrack_cuda_part 
    PUBLIC
    $<${CXX_LANG}:$<BUILD_INTERFACE:${SIXTRACKL_CUDA_PART_CXX_FLAGS}>>
    PRIVATE
    $<${CUDA_LANG}:$<BUILD_INTERFACE:${SIXTRACKL_CUDA_PART_CUDA_FLAGS}>>
)

target_compile_definitions( sixtrack_cuda_part
    PUBLIC
    $<${CXX_LANG}:$<BUILD_INTERFACE:${SIXTRACKL_CUDA_PART_CXX_DEFS}>>
    PUBLIC
    $<${CUDA_LANG}:$<BUILD_INTERFACE:${SIXTRACKL_CUDA_PART_CUDA_DEFS}>>
)

# ------------------------------------------------------------------------------

if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )
    add_subdirectory( tests )
endif()

# ------------------------------------------------------------------------------

set( CUDA_PROPAGATE_HOST_FLAGS ${SAVED_CUDA_PROPAGATE_HOST_FLAGS} )
