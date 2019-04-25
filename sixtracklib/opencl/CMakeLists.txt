# ==============================================================================
# OpenCL 1.x sixtracklib submodule

message( STATUS "---- processing sixtracklib/opencl/CMakeLists.txt" )

set( C99_HEADERS_INSTALL_PATH "include/sixtracklib/opencl" )
set( CXX_HEADERS_INSTALL_PATH ${C99_HEADERS_INSTALL_PATH}  )

#-------------------------------------------------------------------------------
# Prepare the internal cl.h header according to entries from Settings

set( SIXTRL_OPENCL_ENABLE_HOST_EXCEPTIONS "" )

if( SIXTRACKL_USE_LEGACY_CL_HPP )
    set( SIXTRL_OPENCL_CL_HPP "CL/cl.hpp" )
    set( SIXTRL_USES_CL2_HPP 0 )
    set( SIXTRL_USES_CL_HPP  1 )

    if( SIXTRL_OPENCL_ENABLE_HOST_EXCEPTIONS )
        string( APPEND SIXTRL_OPENCL_ENABLE_HOST_EXCEPTIONS
                "    #if !defined( __CL_ENABLE_EXCEPTIONS )\r\n" )
        string( APPEND SIXTRL_OPENCL_ENABLE_HOST_EXCEPTIONS
                "        #define __CL_ENABLE_EXCEPTIONS  \r\n" )
        string( APPEND SIXTRL_OPENCL_ENABLE_HOST_EXCEPTIONS
                "    #endif /* !defined( __CL_ENABLE_EXCEPTIONS ) */ \r\n" )
    endif()
else()
    set( SIXTRL_OPENCL_CL_HPP "CL/cl2.hpp" )
    set( SIXTRL_USES_CL2_HPP 1 )
    set( SIXTRL_USES_CL_HPP  0 )

    if( SIXTRACKL_OPENCL_CXX_ENABLE_EXCEPTIONS )
        string( APPEND SIXTRL_OPENCL_ENABLE_HOST_EXCEPTIONS
                "    #if !defined( CL_HPP_ENABLE_EXCEPTIONS )\r\n" )
        string( APPEND SIXTRL_OPENCL_ENABLE_HOST_EXCEPTIONS
                "        #define CL_HPP_ENABLE_EXCEPTIONS \r\n" )
        string( APPEND SIXTRL_OPENCL_ENABLE_HOST_EXCEPTIONS
                "    #endif /* !defined( CL_HPP_ENABLE_EXCEPTIONS ) */ \r\n" )
    endif()
endif()

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
     set( SIXTRL_OPENCL_ENABLE_HOST_EXCEPTIONS
          ${SIXTRL_OPENCL_ENABLE_HOST_EXCEPTIONS} ) " )

install( CODE "
         ${CL_H_INSTALL_INFO}
         message( STATUS \"Generating: ${CL_H_INSTALL_PATH}\" )
         configure_file( \"${CL_H_IN_FILE}\" \"${CL_H_INSTALL_PATH}\" @ONLY)" )

# ------------------------------------------------------------------------------

set( SIXTRACKLIB_OPENCL_HEADERS
     ${CL_H_OUT_FILE}
     argument.h
     context.h
     track_job_cl.h
)

set( SIXTRACKLIB_OPENCL_INTERNAL_HEADERS
     internal/default_compile_options.h
     internal/base_context.h
)

set( SIXTRACKLIB_OPENCL_SOURCES
     internal/argument.cpp
     internal/base_context.cpp
     internal/context.cpp
     internal/track_job_cl.cpp
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

install( FILES ${SIXTRACKLIB_OPENCL_HEADERS}
         DESTINATION ${CXX_HEADERS_INSTALL_PATH} )

install( FILES ${SIXTRACKLIB_OPENCL_INTERNAL_HEADERS}
         DESTINATION "${CXX_HEADERS_INSTALL_PATH}/internal" )

install( FILES ${SIXTRACKLIB_OPENCL_KERNEL_SOURCES}
         DESTINATION "${C99_HEADERS_INSTALL_PATH}/kernels" )

install( FILES ${SIXTRACKLIB_OPENCL_C99_HEADERS}
         DESTINATION ${C99_HEADERS_INSTALL_PATH} )

# end: sixtracklib/opencl/CMakeLists.txt
