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

set( SIXTRACKL_LIBRARY_MODULES )
set( SIXTRACKL_TOP_LEVEL_HEADERS sixtracklib.h )
set( SIXTRACKL_HEADERS )
set( SIXTRACKL_SOURCES )

add_subdirectory( _impl )

# ------------------------------------------------------------------------------

add_subdirectory( common )
set( SIXTRACKL_LIBRARY_MODULES )
list( APPEND SIXTRACKL_LIBRARY_MODULES $<TARGET_OBJECTS:sixtrack_common> )

# ------------------------------------------------------------------------------

# if( SIXTRACKL_ENABLE_MANUAL_SIMD )
#
#     add_subdirectory( simd )
#
#     set( SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
#          $<TARGET_OBJECTS:sixtrack_simd> )
#
# endif()

# ------------------------------------------------------------------------------

if( SIXTRACKL_ENABLE_OPENCL )

    set( SIXTRACKL_LINK_LIBRARIES ${SIXTRACKL_LINK_LIBRARIES}
                                  ${SIXTRACKL_OPENCL_LIBRARY} )
    add_subdirectory( opencl )

    set( SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
         $<TARGET_OBJECTS:sixtrack_opencl> )

endif()

if( SIXTRACKL_ENABLE_CUDA )

    add_subdirectory( cuda )

    list( APPEND SIXTRACKL_LIBRARY_MODULES $<TARGET_OBJECTS:sixtrack_cuda_device> )
    list( APPEND SIXTRACKL_LIBRARY_MODULES $<TARGET_OBJECTS:sixtrack_cuda_host> )

    set( SIXTRACKL_LINK_LIBRARIES ${SIXTRACKL_LINK_LIBRARIES} cudart )
endif()

# ------------------------------------------------------------------------------

set( SIXTRACKL_LINK_LIBRARIES ${SIXTRACKL_LINK_LIBRARIES} m )
set( SIXTRACKL_HEADERS ${SIXTRACKL_HEADERS} ${SIXTRACKL_TOP_LEVEL_HEADERS} )

add_library( sixtrack SHARED
             ${SIXTRACKL_HEADERS}
             ${SIXTRACKL_SOURCES}
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

target_compile_options( sixtrack
                        PRIVATE -Wall -Werror -pedantic -ansi
                        ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# install :

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

set( SIXTRACKLIB_TLD_INSTALL_PATH include/sixtracklib )

install( FILES ${SIXTRACKL_TOP_LEVEL_HEADERS}
         DESTINATION ${SIXTRACKLIB_TLD_INSTALL_PATH} )

# ------------------------------------------------------------------------------
