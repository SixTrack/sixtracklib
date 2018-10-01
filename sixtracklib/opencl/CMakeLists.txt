set( SIXTRACKLIB_OPENCL_HEADERS
     argument.h
     context.h
)

set( SIXTRACKLIB_OPENCL_INTERNAL_HEADERS
     internal/base_context.h
)

set( SIXTRACKLIB_OPENCL_SOURCES
     internal/argument.cpp
     internal/base_context.cpp
     internal/context.cpp
)

set( SIXTRACKLIB_OPENCL_KERNEL_SOURCES
     kernels/managed_buffer_remap_kernel.cl
     kernels/track_particles_kernel.cl
     kernels/track_particles_priv_particles_optimized_kernel.cl
)

add_library( sixtrack_opencl OBJECT
             ${SIXTRACKLIB_OPENCL_HEADERS}
             ${SIXTRACKLIB_OPENCL_SOURCES}
)

target_include_directories( sixtrack_opencl PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${SIXTRACKL_OPENCL_INCLUDE_DIR} )

set_target_properties( sixtrack_opencl PROPERTIES LINKER_LANGUAGE C )
set_target_properties( sixtrack_opencl PROPERTIES POSITION_INDEPENDENT_CODE True )
set_target_properties( sixtrack_opencl PROPERTIES CXX_STANDARD 11 )
set_target_properties( sixtrack_opencl PROPERTIES CXX_STANDARD_REQUIRED ON   )

target_compile_definitions( sixtrack_opencl PUBLIC
        ${SIXTRACKLIB_NAMESPACE_FLAGS} -DCL_USE_DEPRECATED_OPENCL_1_2_APIS )

target_compile_options( sixtrack_opencl BEFORE PUBLIC
        ${SIXTRACKLIB_CPU_FLAGS} -Wall -Werror -pedantic -ansi )


# ------------------------------------------------------------------------------
# install :

set( SIXTRACKLIB_OPENCL_INSTALL_PATH include/sixtracklib/opencl )

install( FILES ${SIXTRACKLIB_OPENCL_HEADERS}
         DESTINATION ${SIXTRACKLIB_OPENCL_INSTALL_PATH} )

install( FILES ${SIXTRACKLIB_OPENCL_INTERNAL_HEADERS}
         DESTINATION "${SIXTRACKLIB_OPENCL_INSTALL_PATH}/internal" )

install( FILES ${SIXTRACKLIB_OPENCL_KERNEL_SOURCES}
         DESTINATION "${SIXTRACKLIB_OPENCL_INSTALL_PATH}/kernels" )

# end: sixtracklib/opencl/CMakeLists.txt
