set( SIXTRACKLIB_OPENCL_HEADERS
     argument.h
     context.h
)

set( SIXTRACKLIB_OPENCL_INTERNAL_HEADERS
     internal/default_compile_options.h
     internal/base_context.h
)

set( SIXTRACKLIB_OPENCL_SOURCES
     internal/argument.cpp
     internal/base_context.cpp
     internal/context.cpp
)

set( SIXTRACKLIB_OPENCL_C99_HEADERS
     track.h
)

set( SIXTRACKLIB_OPENCL_C99_SOURCES
     internal/track.c
)

set( SIXTRACKLIB_OPENCL_KERNEL_SOURCES
     kernels/track_particles.cl
     kernels/track_particles_debug.cl
     kernels/track_particles_optimized_priv_particles.cl
     kernels/track_particles_optimized_priv_particles_debug.cl
     kernels/be_monitors_assign_out_buffer.cl
     kernels/be_monitors_assign_out_buffer_debug.cl
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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

add_library(
    sixtrack_opencl_c99 OBJECT
    ${SIXTRACKLIB_OPENCL_C99_HEADERS}
    ${SIXTRACKLIB_OPENCL_C99_SOURCES}
)

target_include_directories( sixtrack_opencl_c99 PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${SIXTRACKL_OPENCL_INCLUDE_DIR} )

set_target_properties( sixtrack_opencl_c99 PROPERTIES LINKER_LANGUAGE C )
set_target_properties( sixtrack_opencl_c99 PROPERTIES POSITION_INDEPENDENT_CODE True )
set_target_properties( sixtrack_opencl_c99 PROPERTIES C_STANDARD 99 )
set_target_properties( sixtrack_opencl_c99 PROPERTIES C_STANDARD_REQUIRED ON   )

target_compile_options( sixtrack_opencl_c99 BEFORE PUBLIC
        ${SIXTRACKLIB_CPU_FLAGS} -Wall -Werror -pedantic -ansi )

# ------------------------------------------------------------------------------
# pass on sixtrack_opencl as a module for sixtracklib:

set( SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
     $<TARGET_OBJECTS:sixtrack_opencl>
     $<TARGET_OBJECTS:sixtrack_opencl_c99> CACHE INTERNAL "" FORCE )

set( SIXTRACKL_LINK_LIBRARIES ${SIXTRACKL_LINK_LIBRARIES}
     ${SIXTRACKL_OPENCL_LIBRARY} CACHE INTERNAL "" FORCE )

# ------------------------------------------------------------------------------
# install :

set( SIXTRACKLIB_OPENCL_INSTALL_PATH include/sixtracklib/opencl )

install( FILES ${SIXTRACKLIB_OPENCL_HEADERS}
         DESTINATION ${SIXTRACKLIB_OPENCL_INSTALL_PATH} )

install( FILES ${SIXTRACKLIB_OPENCL_INTERNAL_HEADERS}
         DESTINATION "${SIXTRACKLIB_OPENCL_INSTALL_PATH}/internal" )

install( FILES ${SIXTRACKLIB_OPENCL_KERNEL_SOURCES}
         DESTINATION "${SIXTRACKLIB_OPENCL_INSTALL_PATH}/kernels" )

install( FILES ${SIXTRACKLIB_OPENCL_C99_HEADERS}
         DESTINATION ${SIXTRACKLIB_OPENCL_INSTALL_PATH} )

# end: sixtracklib/opencl/CMakeLists.txt
