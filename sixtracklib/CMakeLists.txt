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

if( SIXTRACKL_C_NAMESPACE_PREFIX )
    add_definitions( "-D__NAMESPACE=${SIXTRACKL_C_NAMESPACE_PREFIX}" )
else()
    add_definitions( "-D__NAMESPACE=st_" )
endif()

# ------------------------------------------------------------------------------
# create a header file _impl/path.h with the current, local absolute path to 
# this directory. This is used as prefix for all source files which have to 
# be processed during run-time (e.g. GPU Kernels, etc.) 

set( SIXTRACKLIB_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" )
configure_file( _impl/path.h.template "${CMAKE_CURRENT_SOURCE_DIR}/_impl/path.h"
                @ONLY )

# ------------------------------------------------------------------------------

set( SIXTRACKL_TEST_LIBRARIES sixtrack )

set( SIXTRACKL_HEADERS 
     _impl/namespace_begin.h 
     _impl/namespace_end.h
)

set( SIXTRACKL_SOURCES )

# ------------------------------------------------------------------------------

add_subdirectory( common )

# ------------------------------------------------------------------------------

add_library( sixtrack SHARED ${SIXTRACKL_HEADERS} ${SIXTRACKL_SOURCES} 
             $<TARGET_OBJECTS:sixtrack_common>
)

set_target_properties( sixtrack PROPERTIES LINKER_LANGUAGE C )
set_target_properties( sixtrack PROPERTIES POSITION_INDEPENDENT_CODE True )
set_target_properties( sixtrack PROPERTIES C_STANDARD 99 )

# ------------------------------------------------------------------------------

if( SIXTRACKL_BUILD_EXAMPLES )
    add_subdirectory( examples )
endif()

# ------------------------------------------------------------------------------

