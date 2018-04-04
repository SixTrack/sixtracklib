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

set( SIXTRACKL_HEADERS )
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
