# sitracklib/_impl/CMakeLists.txt

set( IMPL_INSTALL_PATH "include/sixtracklib/_impl" )

# ------------------------------------------------------------------------------
# modules.h is a header file with pre-defined constants, allownig the user
# of the library to query which modules are present in from C/C++ code

if( SIXTRACKL_ENABLE_MANUAL_SIMD )
    set( SIXTRACKL_ENABLE_MODULE_SIMD 1 )
else()
    set( SIXTRACKL_ENABLE_MODULE_SIMD 0 )
endif()

if( SIXTRACKL_ENABLE_OPENCL )
    set( SIXTRACKL_ENABLE_MODULE_OPENCL 1 )
else()
    set( SIXTRACKL_ENABLE_MODULE_OPENCL 0 )
endif()


if( SIXTRACKL_ENABLE_CUDA )
    set( SIXTRACKL_ENABLE_MODULE_CUDA 1 )
else()
    set( SIXTRACKL_ENABLE_MODULE_CUDA 0 )
endif()


if( SIXTRACKL_ENABLE_MPFR4 )
    set( SIXTRACKL_ENABLE_MODULE_MPFR4 1 )
else()
    set( SIXTRACKL_ENABLE_MODULE_MPFR4 0 )
endif()


set( MODULES_H_IN_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/modules.h.template" )
set( MODULES_H_OUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/modules.h" )

configure_file( ${MODULES_H_IN_PATH} ${MODULES_H_OUT_PATH} @ONLY )

set( SIXTRACKL_HEADERS ${SIXTRACKL_HEADERS} ${MODULES_H_OUT_PATH} PARENT_SCOPE )


set( MODULES_H_INSTALL_PATH
     "${CMAKE_INSTALL_PREFIX}/${IMPL_INSTALL_PATH}/modules.h" )

set( MODULES_ENABLE_DISABLE_INFO
    "set( SIXTRACKL_ENABLE_MODULE_OPENCL ${SIXTRACKL_ENABLE_MODULE_OPENCL} )
     set( SIXTRACKL_ENABLE_MODULE_CUDA   ${SIXTRACKL_ENABLE_MODULE_CUDA}   )
     set( SIXTRACKL_ENABLE_MODULE_SIMD   ${SIXTRACKL_ENABLE_MODULE_SIMD}   )
     set( SIXTRACKL_ENABLE_MODULE_MPFR4  ${SIXTRACKL_ENABLE_MODULE_MPFR4}  )" )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# install:

install( CODE "
    ${MODULES_ENABLE_DISABLE_INFO}
    message( STATUS \"Generating: ${MODULES_H_INSTALL_PATH}\" )
    configure_file( \"${MODULES_H_IN_PATH}\" \"${MODULES_H_INSTALL_PATH}\"
        @ONLY ) " )



# ------------------------------------------------------------------------------
# create a header file _impl/path.h with the current, local absolute path to
# this directory. This is used as prefix for all source files which have to
# be processed during run-time (e.g. GPU Kernels, etc.)

# Note: Paths in CMake always use a slash as separator and always lack a
#       trailing slashes -> add them to make it easier to use the
#       macro in _impl/path.h as a prefix for filename and directory paths!

set( SIXTRACKLIB_BASE_DIR "${CMAKE_SOURCE_DIR}/" )

set( PATH_H_IN_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/path.h.template" )
set( PATH_H_OUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/path.h" )

configure_file( ${PATH_H_IN_PATH} ${PATH_H_OUT_PATH} @ONLY )

set( SIXTRACKL_HEADERS ${SIXTRACKL_HEADERS} ${PATH_H_OUT_PATH} PARENT_SCOPE )

set( PATH_H_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/${IMPL_INSTALL_PATH}/path.h" )


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# install:

install( CODE "
         set( SIXTRACKLIB_BASE_DIR \"${CMAKE_INSTALL_PREFIX}/\" )
         message( STATUS \"Generating: ${PATH_H_INSTALL_PATH}\" )
         configure_file( \"${PATH_H_IN_PATH}\" \"${PATH_H_INSTALL_PATH}\"
            @ONLY ) " )

# ------------------------------------------------------------------------------
# Generate the SixTracklibConfig.cmake package configuration file to the
# lib/cmake directory inside CMAKE_INSTALL_PREFIX:

set( SIXTRACKL_PKG_CONFIG_OPTIONS "" )

if( SIXTRACKL_ENABLE_PROGRAMM_TESTS OR SIXTRACKL_BUILD_EXAMPLES )
    set(   SIXTRACKL_PKG_CONFIG_OPTIONS
         ${SIXTRACKL_PKG_CONFIG_OPTIONS} "set( SixTrackTestlib_FOUND On )" )
else()
    set(   SIXTRACKL_PKG_CONFIG_OPTIONS
         ${SIXTRACKL_PKG_CONFIG_OPTIONS} "set( SixTrackTestlib_FOUND Off )" )
endif()

set( SIXTRACKL_PKG_IN_CONFIG
     "${CMAKE_CURRENT_SOURCE_DIR}/SixTracklibConfig.cmake.template" )

set( SIXTRACKL_PKG_INSTALL_CONFIG
     "${CMAKE_INSTALL_PREFIX}/lib/cmake/SixTracklibConfig.cmake" )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# install:

install( CODE "
         ${SIXTRACKL_PKG_CONFIG_OPTIONS}
         message( STATUS \"Generating: ${SIXTRACKL_PKG_INSTALL_CONFIG}\" )
         configure_file( \"${SIXTRACKL_PKG_IN_CONFIG}\"
                         \"${SIXTRACKL_PKG_INSTALL_CONFIG}\" @ONLY )" )

# ------------------------------------------------------------------------------

set( SIXTRACKL_IMPL_HEADERS
     namespace_begin.h
     definitions.h
     namespace_end.h
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# install:

install( FILES ${SIXTRACKL_IMPL_HEADERS} DESTINATION ${IMPL_INSTALL_PATH} )

# ------------------------------------------------------------------------------

string( REGEX REPLACE "([^;]+)" "_impl/\\1" SIXTRACKL_IMPL_HEADERS
                      "${SIXTRACKL_IMPL_HEADERS}" )

set( SIXTRACKL_HEADERS
     ${SIXTRACKL_HEADERS} ${SIXTRACKL_IMPL_HEADERS} PARENT_SCOPE )

set( SIXTRACKL_HEADERS
     ${SIXTRACKL_HEADERS}
     ${MODULES_H_OUT_PATH}
     ${PATH_H_OUT_PATH}
     PARENT_SCOPE
)
