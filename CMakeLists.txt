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

# Set a default build type if none was specified
set( SIXTRL_DEFAULT_BUILD_TYPE "Release" )

if( NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES )

  message( STATUS "---- Setting build type to '${SIXTRL_DEFAULT_BUILD_TYPE}' as none was specified." )
  set( CMAKE_BUILD_TYPE "${SIXTRL_DEFAULT_BUILD_TYPE}"
       CACHE STRING "Choose the type of build." FORCE)

  # Set the possible values of build type for cmake-gui
  set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo" )

endif()

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
        message( STATUS "---- Loaded local settings file
                 ${PATH_TO_LOCAL_SETTINGS_FILE}" )
    endif()

elseif( EXISTS ${PATH_TO_DEFAULT_SETTINGS_FILE} )

    include( ${PATH_TO_DEFAULT_SETTINGS_FILE} OPTIONAL
             RESULT_VARIABLE STATUS_INCLUDE_DEFAULT_SETTINGS_FILE )

    if( STATUS_INCLUDE_DEFAULT_SETTINGS_FILE )
        message( STATUS "---- Loaded default settings file
                 ${PATH_TO_DEFAULT_SETTINGS_FILE}" )
    endif()

endif()

if( NOT STATUS_INCLUDE_LOCAL_SETTINGS_FILE AND
    NOT STATUS_INCLUDE_DEFAULT_SETTINGS_FILE )

    message( WARNING "---- -> No settings file (i.e. neither Settings.cmake nor"
             " Settings.cmake.default) included" )

endif()

set( SIXTRACKL_VERSION_STR "0.1.0" )

# ==============================================================================
# handle dependencies and external libraries/toolkits:

# ------------------------------------------------------------------------------
# These two abstract lists contain the name of the modules that are know to
# SixTrackLib and "1" and "0" for representing the enabled and disabled state
# of these modules. This is used to generate headers, list, install files, etc.

set( SIXTRACKLIB_SUPPORTED_MODULES        )
set( SIXTRACKLIB_SUPPORTED_MODULES_VALUES )

set( SIXTRACKLIB_SUPPORTED_BINDINGS        )
set( SIXTRACKLIB_SUPPORTED_BINDINGS_VALUES )

set( SIXTRACKLIB_TESTDATA_DIR
     "${CMAKE_SOURCE_DIR}/tests/testdata" CACHE STRING "" FORCE )

set( SIXTRACKLIB_CPU_FLAGS )
set( SIXTRACKLIB_C99_FLAGS )
set( SIXTRACKLIB_CXX_FLAGS )

# ------------------------------------------------------------------------------
# place all contributed cmake modules and snipplets into the cmake subdirectory
# of the project's root folder:

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )
message( STATUS "---- cmake_module_path: ${CMAKE_MODULE_PATH}" )

# ------------------------------------------------------------------------------
# Setup language bindings - we have to be first to pull in python (if needed)
# to prevent any module picking a python version we do not wish to support:

include(CheckLanguage)

include( SetupC99 )
include( SetupCxx )
include( SetupPython )
include( SetupGit )

# ------------------------------------------------------------------------------
# Unit-testing:

include( SetupUnitTesting )

# ------------------------------------------------------------------------------
# MPFR4:

include( SetupMultiPrecision )

# ------------------------------------------------------------------------------
# Autovectorization and SIMD:

include( SetupAutoVecSIMD )

# ------------------------------------------------------------------------------
# OpenMP:

include( SetupOpenMP )

# ------------------------------------------------------------------------------
# OpenCL:

include( SetupOpenCL )

# ------------------------------------------------------------------------------
# Cuda:

if( SIXTRACKL_ENABLE_CUDA )
    check_language( CUDA )
    if( CMAKE_CUDA_COMPILER )
        message( STATUS "---- enabling CUDA language" )
        enable_language( CUDA )
    endif()
endif()

include( SetupCuda )

# ------------------------------------------------------------------------------
# Namespace handling:

set( SIXTRACKLIB_NAMESPACE_FLAGS )

if( SIXTRACKL_C_NAMESPACE_PREFIX )
    list( APPEND SIXTRACKLIB_NAMESPACE_FLAGS
          "__NAMESPACE=${SIXTRACKL_C_NAMESPACE_PREFIX}" )
endif()

if( SIXTRACKL_CXX_NAMESPACE )
    list( APPEND SIXTRACKLIB_NAMESPACE_FLAGS
          "SIXTRL_NAMESPACE=${SIXTRACKL_CXX_NAMESPACE}" )
endif()

# ==============================================================================
# sixtracklib: C/C++ library ; sixtracklib python bindings

add_subdirectory( external )
add_subdirectory( sixtracklib )

# ------------------------------------------------------------------------------
# Add tools

if( SIXTRACKL_BUILD_TOOLS )
    add_subdirectory( tools )
endif()

# ------------------------------------------------------------------------------
# Add sixtracklib Python 3.x bindings

if( SIXTRACKL_ENABLE_PYTHON )
    add_subdirectory( python )
endif()

# ==============================================================================
# examples: c99, cxx, python

if( SIXTRACKL_BUILD_EXAMPLES )
    add_subdirectory( examples )
endif()

# ==============================================================================
# tests: unit-tests, benchmarks, testdata generators:

if( SIXTRACKL_ENABLE_PROGRAMM_TESTS OR SIXTRACKL_ENABLE_BENCHMARK_TESTS OR
    SIXTRACKL_BUILD_EXAMPLES OR SIXTRACKL_BUILD_TOOLS )
    add_subdirectory( tests )
endif()

# ------------------------------------------------------------------------------
#
# end: sixtracklib/CMakeLists.txt
#
