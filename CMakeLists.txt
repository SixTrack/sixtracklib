# SixTrackLib
# Authors: R. De Maria, G. Iadarola, D. Pellegrini, H. Jasim, M. Schwinzerl
#
# Copyright 2018 CERN. This software is distributed under the terms of the GNU
# Lesser General Public License version 2.1, copied verbatim in the file
#`COPYING''.
#
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

cmake_minimum_required( VERSION 3.8 FATAL_ERROR )
project( sixtracklib LANGUAGES C CXX )

message( STATUS "---- Project sixtracklib" )
message( STATUS "---- Inside main CMakeLists.txt" )

# ------------------------------------------------------------------------------
# load local settings file -> this is optional, if not loaded then the default
# settings file will be loaded:

set( PATH_TO_DEFAULT_SETTINGS_FILE
     "${CMAKE_CURRENT_SOURCE_DIR}/Settings.cmake.default" )

set( PATH_TO_LOCAL_SETTINGS_FILE
     "${CMAKE_CURRENT_SOURCE_DIR}/Settings.cmake" )

if( EXISTS ${PATH_TO_LOCAL_SETTINGS_FILE} )

    include( ${PATH_TO_LOCAL_SETTINGS_FILE} OPTIONAL
             RESULT_VARIABLE STATUS_INCLUDE_LOCAL_SETTINGS_FILE )

    if( STATUS_INCLUDE_LOCAL_SETTINGS_FILE )
        message( STATUS "---- Loaded local settings file ${PATH_TO_LOCAL_SETTINGS_FILE}" )
    endif()

elseif( EXISTS ${PATH_TO_DEFAULT_SETTINGS_FILE} )

    include( ${PATH_TO_DEFAULT_SETTINGS_FILE} OPTIONAL
             RESULT_VARIABLE STATUS_INCLUDE_DEFAULT_SETTINGS_FILE )

    if( STATUS_INCLUDE_DEFAULT_SETTINGS_FILE )
        message( STATUS "---- Loaded default settings file ${PATH_TO_DEFAULT_SETTINGS_FILE}" )
    endif()

endif()

if( NOT STATUS_INCLUDE_LOCAL_SETTINGS_FILE AND
    NOT STATUS_INCLUDE_DEFAULT_SETTINGS_FILE )

    message( WARNING "No settings file (i.e. neither settings.cmake nor "
             "settings.cmake.default) included" )

endif()

# ==============================================================================
# handle dependencies and external libraries/toolkits:

# ------------------------------------------------------------------------------
# place all contributed cmake modules and snipplets into the cmake subdirectory
# of the project's root folder:

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )
message( STATUS "cmake_module_path: ${CMAKE_MODULE_PATH}" )

# ------------------------------------------------------------------------------
# Unit-testing:

include( SetupUnitTesting )

# ------------------------------------------------------------------------------
# MPFR4:

if( SIXTRACKL_ENABLE_MPFR4 OR SIXTRACKL_ENABLE_MPFR_ACCURACY_TESTS )
    include( SetupMultiPrecision )
endif()


# ------------------------------------------------------------------------------
# OpenCL:

if( SIXTRACKL_ENABLE_OPENCL )
    include( SetupOpenCL )
endif()

# ------------------------------------------------------------------------------
# CUDA:

if( SIXTRACKL_ENABLE_CUDA )
    enable_language( CUDA )
endif()

# ------------------------------------------------------------------------------
# OpenMP:

if( SIXTRACKL_ENABLE_OPENMP )
    include( SetupOpenMP )
endif()

# ==============================================================================
# build libraries and modules:

# ------------------------------------------------------------------------------
# sixtracklib: C/C++ library

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# CPU/System architecture settings:

set( SIXTRACKLIB_CPU_FLAGS )

if( SIXTRACKL_CPU_ARCH MATCHES "avx" )
    message( STATUS "---- Optimizing for AVX architecture" )
    set( SIXTRACKLIB_CPU_FLAGS ${SIXTRACKLIB_CPU_FLAGS} -mavx )

elseif( SIXTRACKL_CPU_ARCH MATCHES "sse2" )
    message( STATUS "---- Optimizing for SSE2 architecture" )
    set( SIXTRACKLIB_CPU_FLAGS ${SIXTRACKLIB_CPU_FLAGS} -msse2  )

elseif( SIXTRACKL_CPU_ARCH MATCHES "native" )
    message( STATUS "---- Optimizing for native environment of the CPU" )
    set( SIXTRACKLIB_CPU_FLAGS ${SIXTRACKLIB_CPU_FLAGS} -march=native  )

endif()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# C Namespace handling:

if( SIXTRACKL_C_NAMESPACE_PREFIX )
    set( SIXTRACKLIB_NAMESPACE_FLAGS
         "-D__NAMESPACE=${SIXTRACKL_C_NAMESPACE_PREFIX}" )
else()
    set( SIXTRACKLIB_NAMESPACE_FLAGS "-D__NAMESPACE=st_" )
endif()

add_subdirectory( sixtracklib )

# ------------------------------------------------------------------------------

if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )
    add_subdirectory( tests )
endif()

# ------------------------------------------------------------------------------

if( SIXTRACKL_BUILD_EXAMPLES )
    add_subdirectory( examples )
endif()

# ------------------------------------------------------------------------------
#
# end: sixtracklib/CMakeLists.txt
#
