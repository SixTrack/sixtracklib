# ==============================================================================
# OpenCL 1.x sixtracklib submodule

message( STATUS "---- processing sixtracklib/opencl/CMakeLists.txt" )

set( C99_HEADERS_INSTALL_PATH "include/sixtracklib/opencl" )
set( CXX_HEADERS_INSTALL_PATH ${C99_HEADERS_INSTALL_PATH}  )

set( CL_H_IN_FILE  "${CMAKE_CURRENT_SOURCE_DIR}/cl.h.template" )
set( CL_H_OUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cl.h" )
configure_file( ${CL_H_IN_FILE} ${CL_H_OUT_FILE} @ONLY )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Handle install of cl.h

set( CL_H_INSTALL_PATH
     "${CMAKE_INSTALL_PREFIX}/${CXX_HEADERS_INSTALL_PATH}/${CL_H_OUT_FILE}" )

set( CL_H_INSTALL_INFO
    "set( SIXTRACKL_MIN_OPENCL_VERSION    ${SIXTRACKL_MIN_OPENCL_VERSION} )
     set( SIXTRACKL_TARGET_OPENCL_VERSION ${SIXTRACKL_TARGET_OPENCL_VERSION} )
     set( SIXTRL_OPENCL_CL_HPP ${SIXTRL_OPENCL_CL_HPP} )
     set( SIXTRL_USES_CL2_HPP  ${SIXTRL_USES_CL2_HPP}  )
     set( SIXTRL_USES_CL_HPP   ${SIXTRL_USES_CL_HPP}   )
     set( SIXTRL_OPENCL_ENABLE_EXCEPTION_STR
          ${SIXTRL_OPENCL_ENABLE_EXCEPTION_STR} )
     set( SIXTRL_OPENCL_ENABLES_EXCEPTION_FLAG
          ${SIXTRL_OPENCL_ENABLES_EXCEPTION_FLAG} ) " )

install( CODE "
         ${CL_H_INSTALL_INFO}
         message( STATUS \"Generating: ${CL_H_INSTALL_PATH}\" )
         configure_file( \"${CL_H_IN_FILE}\" \"${CL_H_INSTALL_PATH}\" @ONLY)" )

# ------------------------------------------------------------------------------

set( SIXTRACKLIB_OPENCL_HEADERS
     ${CL_H_OUT_FILE} argument.h context.h track_job_cl.h make_track_job.h )

set( SIXTRACKLIB_OPENCL_INTERNAL_HEADERS
     internal/default_compile_options.h internal/base_context.h )

set( SIXTRACKLIB_OPENCL_SOURCES internal/argument.cpp internal/base_context.cpp
    internal/context.cpp internal/track_job_cl.cpp internal/make_track_job.cpp )

set( SIXTRACKLIB_OPENCL_C99_HEADERS )
set( SIXTRACKLIB_OPENCL_C99_SOURCES )

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

target_include_directories( sixtrack_opencl
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    PUBLIC $<BUILD_INTERFACE:${SIXTRACKL_OPENCL_INCLUDE_DIR}>
)

set_target_properties( sixtrack_opencl PROPERTIES LINKER_LANGUAGE C
    POSITION_INDEPENDENT_CODE True CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON )

target_compile_definitions( sixtrack_opencl PUBLIC
        ${SIXTRACKLIB_NAMESPACE_FLAGS} -DCL_USE_DEPRECATED_OPENCL_1_2_APIS )

target_compile_options( sixtrack_opencl BEFORE PUBLIC
        ${SIXTRACKLIB_CPU_FLAGS} ${SIXTRACKLIB_CXX_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_opencl as a module for sixtracklib:

set( SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
     $<TARGET_OBJECTS:sixtrack_opencl> CACHE INTERNAL "" FORCE )

set( SIXTRACKL_LINK_LIBRARIES ${SIXTRACKL_LINK_LIBRARIES}
     ${SIXTRACKL_OPENCL_LIBRARY} CACHE INTERNAL "" FORCE )

# ------------------------------------------------------------------------------
# install :

install( FILES ${SIXTRACKLIB_OPENCL_HEADERS}
         DESTINATION ${CXX_HEADERS_INSTALL_PATH} )

install( FILES ${SIXTRACKLIB_OPENCL_INTERNAL_HEADERS}
         DESTINATION "${CXX_HEADERS_INSTALL_PATH}/internal" )

install( FILES ${SIXTRACKLIB_OPENCL_KERNEL_SOURCES}
         DESTINATION "${C99_HEADERS_INSTALL_PATH}/kernels" )

# end: sixtracklib/opencl/CMakeLists.txt
