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
# C Namespace handling:

if( SIXTRACKL_C_NAMESPACE_PREFIX )
    set( SIXTRACKLIB_NAMESPACE_FLAGS 
         "-D__NAMESPACE=${SIXTRACKL_C_NAMESPACE_PREFIX}" )
else()
    set( SIXTRACKLIB_NAMESPACE_FLAGS "-D__NAMESPACE=st_" )
endif()

# ------------------------------------------------------------------------------

set( SIXTRACKL_LIBRARY_MODULES )
set( SIXTRACKL_TEST_LIBRARIES sixtrack )

set( SIXTRACKL_HEADERS sixtracklib.h )
set( SIXTRACKL_SOURCES )

set( SIXTRACKLIB_TESTDATA_DIR "${CMAKE_SOURCE_DIR}/testdata" )

add_subdirectory( _impl )

# ------------------------------------------------------------------------------

add_subdirectory( common )
set( SIXTRACKL_LIBRARY_MODULES $<TARGET_OBJECTS:sixtrack_common> )

# ------------------------------------------------------------------------------

if( SIXTRACKL_ENABLE_MANUAL_SIMD )    
     
    add_subdirectory( simd )
    
    set( SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
         $<TARGET_OBJECTS:sixtrack_simd> )

endif()

# ------------------------------------------------------------------------------

if( SIXTRACKL_ENABLE_OPENCL )
    
    set( SIXTRACKL_LINK_LIBRARIES ${SIXTRACKL_LINK_LIBRARIES} 
                                  ${SIXTRACKL_OPENCL_LIBRARY} )
    add_subdirectory( opencl )
    
    set( SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
         $<TARGET_OBJECTS:sixtrack_opencl> )
    
endif()

if( SIXTRACKL_ENABLE_CUDA )

#     set( SIXTRACKL_LINK_LIBRARIES ${SIXTRACKL_LINK_LIBRARIES} 
#                                   ${SIXTRACKL_CUDA_LIBRARIES} )
        
    add_subdirectory( cuda )

    set( SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
         $<TARGET_OBJECTS:sixtrack_cuda> )
    
endif()
                
# ------------------------------------------------------------------------------

add_library( sixtrack SHARED 
             ${SIXTRACKL_HEADERS} ${SIXTRACKL_SOURCES} 
             ${SIXTRACKL_LIBRARY_MODULES} 
)

if( SIXTRACKL_LINK_LIBRARIES )
    target_link_libraries( sixtrack ${SIXTRACKL_LINK_LIBRARIES} )
endif()

target_include_directories( 
    sixtrack PUBLIC $<INSTALL_INTERFACE:include>
                    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> )

set_target_properties( sixtrack PROPERTIES LINKER_LANGUAGE C )
set_target_properties( sixtrack PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties( sixtrack PROPERTIES C_STANDARD 99 )
set_target_properties( sixtrack PROPERTIES DEBUG_POSTFIX d )

target_compile_definitions( sixtrack PUBLIC ${SIXTRACKLIB_NAMESPACE_FLAGS} )

target_compile_options( sixtrack PRIVATE -Wall -Werror -pedantic -ansi ${SIXTRACKLIB_CPU_FLAGS} )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
install( TARGETS sixtrack 
         EXPORT  SixTracklib-targets 
         RUNTIME DESTINATION bin 
         LIBRARY DESTINATION lib 
         ARCHIVE DESTINATION lib )
    
install( EXPORT SixTracklib-targets 
         FILE   SixTracklibTargets.cmake 
         NAMESPACE SixTracklib::
         DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake
)
    
# ------------------------------------------------------------------------------

# if( SIXTRACKL_BUILD_EXAMPLES )
#     add_subdirectory( examples )
# endif()

# ------------------------------------------------------------------------------
