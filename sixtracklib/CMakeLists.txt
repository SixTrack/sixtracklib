# sixtracklib/sixtracklib
#
# Authors: R. De Maria, G. Iadarola, D. Pellegrini, H. Jasim, M. Schwinzerl
#
# Copyright 2018 CERN. This software is distributed under the terms of the GNU
# Lesser General Public License version 2.1, copied verbatim in the file
#`COPYING''.
#
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# ------------------------------------------------------------------------------

set( SIXTRACKL_LINK_LIBRARIES )

# ------------------------------------------------------------------------------

if( SIXTRACKL_C_NAMESPACE_PREFIX )
    add_definitions( "-D__NAMESPACE=${SIXTRACKL_C_NAMESPACE_PREFIX}" )
else()
    add_definitions( "-D__NAMESPACE=st_" )
endif()

# ------------------------------------------------------------------------------
# create a header file _impl/path.h with the current, local absolute path to 
# this directory. This is used as prefix for all source files which have to 
# be processed during run-time (e.g. GPU Kernels, etc.) 

# Note: Paths in CMake always use a slash as separator and always lack a 
#       trailing slashes -> add them to make it easier to use the 
#       macro in _impl/path.h as a prefix for filename and directory paths!

set( SIXTRACKLIB_BASE_DIR "${CMAKE_SOURCE_DIR}/" )
configure_file( _impl/path.h.template "${CMAKE_CURRENT_SOURCE_DIR}/_impl/path.h"
                @ONLY )


# ------------------------------------------------------------------------------
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
                
# ------------------------------------------------------------------------------

set( SIXTRACKL_LIBRARY_MODULES )
set( SIXTRACKL_TEST_LIBRARIES sixtrack )

set( SIXTRACKL_HEADERS 
     _impl/namespace_begin.h 
     _impl/namespace_end.h
)

# ------------------------------------------------------------------------------

add_subdirectory( common )
set( SIXTRACKL_LIBRARY_MODULES $<TARGET_OBJECTS:sixtrack_common> )

# ------------------------------------------------------------------------------

if( SIXTRACKL_ENABLE_MANUAL_SIMD )    
    
    add_subdirectory( simd )
    
    set( SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
         $<TARGET_OBJECTS:sixtrack_simd> )    
         
    set( SIXTRACKL_ENABLE_MODULE_SIMD 1 )
else()
    set( SIXTRACKL_ENABLE_MODULE_SIMD 0 )
endif()

# ------------------------------------------------------------------------------

if( SIXTRACKL_ENABLE_OPENCL )
    add_subdirectory( opencl )
    
    set( SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
         $<TARGET_OBJECTS:sixtrack_opencl> )
         
    set( SIXTRACKL_LINK_LIBRARIES ${SIXTRACKL_LINK_LIBRARIES} 
                                  ${SIXTRACKL_OPENCL_LIBRARY} )
                                  
    set( SIXTRACKL_ENABLE_MODULE_OPENCL 1 )
    
else()
    
    set( SIXTRACKL_ENABLE_MODULE_OPENCL 0 )

endif()

# ------------------------------------------------------------------------------

configure_file( _impl/modules.h.template 
                "${CMAKE_CURRENT_SOURCE_DIR}/_impl/modules.h" @ONLY )

# ------------------------------------------------------------------------------

set( SIXTRACKL_HEADERS
     _impl/definitions.h
     _impl/modules.h 
     _impl/namespace_begin.h
     _impl/namespace_end.h 
     _impl/path.h )

set( SIXTRACKL_SOURCES )

add_library( sixtrack SHARED 
             ${SIXTRACKL_HEADERS} ${SIXTRACKL_SOURCES} 
             ${SIXTRACKL_LIBRARY_MODULES} 
)

if( SIXTRACKL_LINK_LIBRARIES )
    target_link_libraries( sixtrack ${SIXTRACKL_LINK_LIBRARIES} )
endif()

set_target_properties( sixtrack PROPERTIES LINKER_LANGUAGE C )
set_target_properties( sixtrack PROPERTIES POSITION_INDEPENDENT_CODE True )
set_target_properties( sixtrack PROPERTIES C_STANDARD 99 )

target_compile_options( sixtrack PUBLIC 
    ${SIXTRACKLIB_CPU_FLAGS} -Wall -Werror -pedantic -ansi )

# ------------------------------------------------------------------------------

if( SIXTRACKL_BUILD_EXAMPLES )
    add_subdirectory( examples )
endif()

# ------------------------------------------------------------------------------

