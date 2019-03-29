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

set( SIXTRACKL_LINK_LIBRARIES  "" CACHE INTERNAL "" FORCE )
set( SIXTRACKL_LIBRARY_MODULES "" CACHE INTERNAL "" FORCE )

# ------------------------------------------------------------------------------

set( SIXTRACKLIB_C99_HEADERS )
set( SIXTRACKLIB_CXX_HEADERS )

set( SIXTRACKL_LOCAL_C99_HEADERS
     sixtracklib.h
)

set( SIXTRACKL_LOCAL_CXX_HEADERS
     sixtracklib.hpp
)

add_subdirectory( common )

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
    add_subdirectory( opencl )
endif()

if( SIXTRACKL_ENABLE_CUDA )
    add_subdirectory( cuda )
endif()

# ------------------------------------------------------------------------------

set( SIXTRACKL_LINK_LIBRARIES ${SIXTRACKL_LINK_LIBRARIES} m )

add_library( sixtrack SHARED
             ${SIXTRACKLIB_C99_HEADERS}
             ${SIXTRACKL_LOCAL_C99_HEADERS}
             ${SIXTRACKLIB_CXX_HEADERS}
             ${SIXTRACKL_LOCAL_CXX_HEADERS}
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
set_target_properties( sixtrack PROPERTIES C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack
                        PRIVATE -Wall -Werror -pedantic -ansi
                        ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# provide a copy of the library to the python module if it is defined!

if( SIXTRACKL_ENABLE_PYTHON )
    get_target_property( OUT_SIXTRL_PREFIX  sixtrack PREFIX )
    get_target_property( OUT_SIXTRL_POSTFIX sixtrack RELEASE_POSTFIX )
    get_target_property( OUT_SIXTRL_NAME    sixtrack OUTPUT_NAME )

    set( LIB_NAME )

    if( OUT_SIXTRL_PREFIX )
        string( APPEND LIB_NAME  ${OUT_SIXTRL_PREFIX} )
    else()
        string( APPEND LIB_NAME  "lib" )
    endif()

    if( OUT_SIXTRL_NAME )
        string( APPEND LIB_NAME ${OUT_SIXTRL_NAME} )
    else()
        string( APPEND LIB_NAME "sixtrack" )
    endif()

    if( OUT_SIXTRL_POSTFIX )
        string( APPEND LIB_NAME ${OUT_SIXTRL_POSTFIX} )
    endif()

    string( APPEND LIB_NAME ${CMAKE_SHARED_LIBRARY_SUFFIX} )

    if( EXISTS ${CMAKE_SOURCE_DIR}/python/pysixtracklib )
        set(  PY_SIXTRACKLIB_SHARED_LIB ${LIB_NAME} CACHE INTERNAL "" FORCE )
        set( IN_PATH    $<TARGET_FILE:sixtrack> )
        set( DEST_PATH "${CMAKE_SOURCE_DIR}/python/pysixtracklib/${LIB_NAME}" )

        message( STATUS "---- Providing a copy of sixtracklib to pysixtacklib "
                        "directory: ${DEST_PATH}" )

        add_custom_command( TARGET sixtrack POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${IN_PATH} ${DEST_PATH} )
    endif()
endif()

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

if( SIXTRACKL_LOCAL_C99_HEADERS )

    set( SIXTRACKLIB_C99_INSTALL_PATH include/sixtracklib )

    install( FILES ${SIXTRACKL_LOCAL_C99_HEADERS}
             DESTINATION ${SIXTRACKLIB_C99_INSTALL_PATH} )
endif()

if( SIXTRACKL_LOCAL_CXX_HEADERS )

    set( SIXTRACKLIB_CXX_INSTALL_PATH include/sixtracklib )

    install( FILES ${SIXTRACKL_LOCAL_CXX_HEADERS}
             DESTINATION ${SIXTRACKLIB_CXX_INSTALL_PATH} )
endif()

# ------------------------------------------------------------------------------
