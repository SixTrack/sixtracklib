# ==============================================================================
# common/generated sixtracklib submodule

set( GENERATED_INSTALL_PATH "include/sixtracklib/common/generated" )
set( SIXTRACKL_COMMON_GENERATED_HEADERS )

# ------------------------------------------------------------------------------
# modules.h is a header file with pre-defined constants, allownig the user
# of the library to query which modules are present in from C/C++ code

if( SIXTRACKL_ENABLE_MANUAL_SIMD )
    set( SIXTRACKL_ENABLE_MODULE_SIMD 1 )
    set( PY_SIXTRACKL_MODULE_SIMD "True" )
else()
    set( SIXTRACKL_ENABLE_MODULE_SIMD 0 )
    set( PY_SIXTRACKL_MODULE_SIMD "False" )
endif()

if( SIXTRACKL_ENABLE_OPENCL )
    set( SIXTRACKL_ENABLE_MODULE_OPENCL 1 )
    set( PY_SIXTRACKL_MODULE_OPENCL "True" )
else()
    set( SIXTRACKL_ENABLE_MODULE_OPENCL 0 )
    set( PY_SIXTRACKL_MODULE_OPENCL "False" )
endif()


if( SIXTRACKL_ENABLE_CUDA )
    set( SIXTRACKL_ENABLE_MODULE_CUDA 1 )
    set( PY_SIXTRACKL_MODULE_CUDA "True" )
else()
    set( SIXTRACKL_ENABLE_MODULE_CUDA 0 )
    set( PY_SIXTRACKL_MODULE_CUDA "False" )
endif()


if( SIXTRACKL_ENABLE_MPFR4 )
    set( SIXTRACKL_ENABLE_MODULE_MPFR4 1 )
    set( PY_SIXTRACKL_MODULE_MPFRA "True" )
else()
    set( SIXTRACKL_ENABLE_MODULE_MPFR4 0 )
    set( PY_SIXTRACKL_MODULE_MPFRA "False" )
endif()


set( MODULES_H_IN_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/modules.h.template" )
set( MODULES_H_OUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/modules.h" )

configure_file( ${MODULES_H_IN_PATH} ${MODULES_H_OUT_PATH} @ONLY )
set(   SIXTRACKL_COMMON_GENERATED_HEADERS
     ${SIXTRACKL_COMMON_GENERATED_HEADERS} "generated/modules.h" )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# prepare python related module list

if( SIXTRACKL_ENABLE_PYTHON )
    set( PY_SIXTRACKLIB_MODULES_STR
        "SIXTRACKLIB_MODULES = {}"
        "SIXTRACKLIB_MODULES[\"simd\"]   = ${PY_SIXTRACKL_MODULE_SIMD}"
        "SIXTRACKLIB_MODULES[\"opencl\"] = ${PY_SIXTRACKL_MODULE_OPENCL}"
        "SIXTRACKLIB_MODULES[\"cuda\"]   = ${PY_SIXTRACKL_MODULE_CUDA}"
        "SIXTRACKLIB_MODULES[\"mpfr4\"]  = ${PY_SIXTRACKL_MODULE_MPFRA}"
         CACHE INTERNAL "" FORCE )
endif()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# install:

set( MODULES_H_INSTALL_PATH
     "${CMAKE_INSTALL_PREFIX}/${GENERATED_INSTALL_PATH}/modules.h" )

set( MODULES_ENABLE_DISABLE_INFO
    "set( SIXTRACKL_ENABLE_MODULE_OPENCL ${SIXTRACKL_ENABLE_MODULE_OPENCL} )
     set( SIXTRACKL_ENABLE_MODULE_CUDA   ${SIXTRACKL_ENABLE_MODULE_CUDA}   )
     set( SIXTRACKL_ENABLE_MODULE_SIMD   ${SIXTRACKL_ENABLE_MODULE_SIMD}   )
     set( SIXTRACKL_ENABLE_MODULE_MPFR4  ${SIXTRACKL_ENABLE_MODULE_MPFR4}  )" )

install( CODE "
    ${MODULES_ENABLE_DISABLE_INFO}
    message( STATUS \"Generating: ${MODULES_H_INSTALL_PATH}\" )
    configure_file( \"${MODULES_H_IN_PATH}\" \"${MODULES_H_INSTALL_PATH}\"
        @ONLY ) " )

# ------------------------------------------------------------------------------
# create a header file common/generated/path.h with the current,
# local absolute path to this directory. This is used as prefix for all
# source files which have to be processed during run-time
# (e.g. GPU Kernels, etc.)

# Note: Paths in CMake always use a slash as separator and always lack a
#       trailing slashes -> add them to make it easier to use the
#       macro in common/generated/path.h as a prefix for
#       filename and directory paths!

set( SIXTRACKLIB_BASE_DIR     "${CMAKE_SOURCE_DIR}/" )
set( SIXTRACKLIB_INCLUDE_DIR  ${SIXTRACKLIB_BASE_DIR} )

set( PATH_H_IN_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/path.h.template" )
set( PATH_H_OUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/path.h" )

configure_file( ${PATH_H_IN_PATH} ${PATH_H_OUT_PATH} @ONLY )

set(  SIXTRACKL_COMMON_GENERATED_HEADERS
    ${SIXTRACKL_COMMON_GENERATED_HEADERS} "generated/path.h" )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# install:

set( PATH_H_INSTALL_PATH
     "${CMAKE_INSTALL_PREFIX}/${GENERATED_INSTALL_PATH}/path.h" )

install( CODE "
         set( SIXTRACKLIB_BASE_DIR \"${CMAKE_INSTALL_PREFIX}/\" )
         set( SIXTRACKLIB_INCLUDE_DIR \"${CMAKE_INSTALL_PREFIX}/include\" )
         message( STATUS \"Generating: ${PATH_H_INSTALL_PATH}\" )
         configure_file( \"${PATH_H_IN_PATH}\" \"${PATH_H_INSTALL_PATH}\"
            @ONLY ) " )


# ------------------------------------------------------------------------------
# create a header file common/generated/config.h with encoding top-level
# configuration options from Settings.cmake:

if( SIXTRACKL_ENABLE_APERTURE_CHECK )
    set( SIXTRACKL_ENABLE_APERTURE_CHECK_FLAG "1" )
else()
    set( SIXTRACKL_ENABLE_APERTURE_CHECK_FLAG "0" )
endif()

set( CONFIG_H_IN_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/config.h.template" )
set( CONFIG_H_OUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/config.h" )

configure_file( ${CONFIG_H_IN_PATH} ${CONFIG_H_OUT_PATH} @ONLY )

set(  SIXTRACKL_COMMON_GENERATED_HEADERS
    ${SIXTRACKL_COMMON_GENERATED_HEADERS} "generated/config.h" )


if( SIXTRACKL_ENABLE_PYTHON )
    if( SIXTRACKL_ENABLE_APERATURE_CHECK )
        set( PY_SIXTRL_ENABLE_APERTURE_CHECK "True" CACHE INTERNAL "" FORCE )
    else()
        set( PY_SIXTRL_ENABLE_APERTURE_CHECK "False" CACHE INTERNAL "" FORCE )
    endif()

    set( PY_SIXTRL_APERTURE_X_LIMIT "${SIXTRACKL_APERTURE_X_LIMIT_VALUE}"
         CACHE INTERNAL "" FORCE )

    set( PY_SIXTRL_APERTURE_Y_LIMIT "${SIXTRACKL_APERTURE_Y_LIMIT_VALUE}"
         CACHE INTERNAL "" FORCE )
endif()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# install:

set( CONFIG_H_INSTALL_PATH
     "${CMAKE_INSTALL_PREFIX}/${GENERATED_INSTALL_PATH}/config.h" )

install( CODE "
         set( SIXTRACKL_ENABLE_APERTURE_CHECK_FLAG
              \"${SIXTRACKL_ENABLE_APERTURE_CHECK_FLAG}\" )
         set( SIXTRACKL_APERTURE_X_LIMIT_VALUE ${SIXTRACKL_APERTURE_X_LIMIT_VALUE} )
         set( SIXTRACKL_APERTURE_Y_LIMIT_VALUE ${SIXTRACKL_APERTURE_Y_LIMIT_VALUE} )
         message( STATUS \"Generating: ${CONFIG_H_INSTALL_PATH}\" )
         configure_file( \"${CONFIG_H_IN_PATH}\" \"${CONFIG_H_INSTALL_PATH}\" @ONLY ) " )

# ------------------------------------------------------------------------------
# Create a header file encapsulating the choices for the C and C++
# namespaces. This moves namespaces away from being passed down via
# command line flags (altough overriding them via command line defines
# is still possible) and encapsulates everything within the library itself ->
# this should hopefully reduce the hassles for users of the library and
# reduce the problems if users want to, positively, not use CMake

if( SIXTRACKL_C_NAMESPACE_PREFIX )
    set( SIXTRL_GENERATED_C99_NAMESPACE ${SIXTRACKL_C_NAMESPACE_PREFIX} )
else()
    set( SIXTRL_GENERATED_C99_NAMESPACE "st_" )
endif()

if( SIXTRACKL_CXX_NAMESPACE )
    set( SIXTRL_GENERATED_CXX_NAMESPACE ${SIXTRACKL_CXX_NAMESPACE} )
else()
    set( SIXTRL_GENERATED_CXX_NAMESPACE "sixtrack" )
endif()

set( NAMESPACE_H_IN_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/namespace.h.template" )
set( NAMESPACE_H_OUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/namespace.h" )

configure_file( ${NAMESPACE_H_IN_PATH} ${NAMESPACE_H_OUT_PATH} @ONLY )

set(  SIXTRACKL_COMMON_GENERATED_HEADERS
    ${SIXTRACKL_COMMON_GENERATED_HEADERS} "generated/namespace.h" )


if( SIXTRACKL_ENABLE_PYTHON )

    set( PY_SIXTRL_C99_NAMESPACE_STR "${SIXTRL_GENERATED_C99_NAMESPACE}"
         CACHE INTERNAL "" FORCE )

    set( PY_SIXTRL_C99_NAMESPACE_PREFIX_STR "${SIXTRL_GENERATED_C99_NAMESPACE}"
         CACHE INTERNAL "" FORCE )

    set( PY_SIXTRL_CXX_NAMESPACE_STR "${SIXTRL_GENERATED_CXX_NAMESPACE}"
         CACHE INTERNAL "" FORCE )

    set( PY_SIXTRL_CXX_NAMESPACE_PREFIX_STR
         "${SIXTRL_GENERATED_CXX_NAMESPACE}::" CACHE INTERNAL "" FORCE )

endif()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# install:

set( NAMESPACE_H_INSTALL_PATH
     "${CMAKE_INSTALL_PREFIX}/${GENERATED_INSTALL_PATH}/namespace.h" )

install( CODE "
         set( SIXTRL_GENERATED_C99_NAMESPACE \"${SIXTRL_GENERATED_C99_NAMESPACE}\" )
         set( SIXTRL_GENERATED_CXX_NAMESPACE \"${SIXTRL_GENERATED_CXX_NAMESPACE}\" )
         message( STATUS \"Generating: ${NAMESPACE_H_INSTALL_PATH}\" )
         configure_file( \"${NAMESPACE_H_IN_PATH}\"
                         \"${NAMESPACE_H_INSTALL_PATH}\" @ONLY ) " )

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

install( CODE "
         ${SIXTRACKL_PKG_CONFIG_OPTIONS}
         message( STATUS \"Generating: ${SIXTRACKL_PKG_INSTALL_CONFIG}\" )
         configure_file( \"${SIXTRACKL_PKG_IN_CONFIG}\"
                         \"${SIXTRACKL_PKG_INSTALL_CONFIG}\" @ONLY )" )

# ------------------------------------------------------------------------------

set(   SIXTRACKLIB_COMMON_C99_HEADERS
     ${SIXTRACKLIB_COMMON_C99_HEADERS}
     ${SIXTRACKL_COMMON_GENERATED_HEADERS} PARENT_SCOPE )


#end: sixtracklib/common/generated/CMakeLists.txt
